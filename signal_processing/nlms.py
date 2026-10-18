"""Normalized Least Mean Squares (NLMS) adaptive filter in the STFT domain.

Conforms to DRDO-ANC AI Coding Contract v1.
Maintains one filter per frequency bin, with length L = NLMS_TAPS frames.
Uses ai_engine.models.stft.STFT for time-frequency boundary conversions.
"""

from typing import Any

import numpy as np
import torch

from ai_engine.models.stft import STFT
from common.constants import N_FFT, N_FREQ, NLMS_EPS, NLMS_MU, NLMS_TAPS, SAMPLE_RATE

DEFAULT_PARAMS: dict[str, Any] = {
    "taps": NLMS_TAPS,
    "mu": NLMS_MU,
    "eps": NLMS_EPS,
}


class NLMSFilter:
    """Frequency-domain Normalized Least Mean Squares (NLMS) filter.

    Maintains one independent adaptive FIR filter for each of the `n_freq`
    frequency bins. Operates on complex STFT frames.

    Algorithm per frame t:
        y[k, t] = sum_{l=0..L-1} W[k, l] * X[k, t - l]
        E[k, t] = D[k, t] - y[k, t]
        W[k, l] += mu * conj(X[k, t - l]) * E[k, t] / (eps + sum_m |X[k, t - m]|^2)

    Attributes:
        n_freq: Number of frequency bins (defaults to N_FREQ = 257).
        taps: Filter length L in frames (defaults to NLMS_TAPS = 4).
        mu: Step size parameter (defaults to NLMS_MU = 0.5).
        eps: Regularization parameter to avoid division by zero (defaults to NLMS_EPS = 1e-6).
        w: Complex filter weights of shape [n_freq, taps].
    """

    def __init__(
        self,
        n_freq: int = N_FREQ,
        taps: int = NLMS_TAPS,
        mu: float = NLMS_MU,
        eps: float = NLMS_EPS,
    ) -> None:
        """Initialize NLMS filter and reset weights."""
        self.n_freq = n_freq
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self.w = np.zeros((self.n_freq, self.taps), dtype=np.complex64)
        self.reset()

    def reset(self) -> None:
        """Reset adaptive filter weights to zero."""
        self.w.fill(0.0)

    def process(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Process reference (X) and primary/desired (D) complex spectra.

        Vectorized over frequency bins [n_freq] and filter taps [taps],
        looping sequentially across time frames [F].

        Args:
            X: Reference complex spectrum of shape [n_freq, F].
            D: Primary/noisy complex spectrum of shape [n_freq, F].

        Returns:
            np.ndarray: Error signal spectrum E of shape [n_freq, F] with dtype complex64.
        """
        n_freq, n_frames = D.shape
        if X.shape != D.shape:
            raise ValueError(
                f"Shape mismatch: reference X {X.shape} must match primary D {D.shape}"
            )
        if n_freq != self.n_freq:
            raise ValueError(
                f"Frequency dimension mismatch: expected {self.n_freq}, got {n_freq}"
            )

        E = np.zeros((n_freq, n_frames), dtype=np.complex64)

        # Buffer for past reference frames across taps: shape [n_freq, taps]
        x_buf = np.zeros((n_freq, self.taps), dtype=np.complex64)

        for t in range(n_frames):
            # Shift buffer right and insert current reference frame at index 0
            x_buf = np.roll(x_buf, shift=1, axis=1)
            x_buf[:, 0] = X[:, t]

            # Filter output: y[k, t] = sum_l W[k, l] * X[k, t - l]
            y = np.sum(self.w * x_buf, axis=1)

            # Error signal: E[k, t] = D[k, t] - y[k, t]
            e = D[:, t] - y
            E[:, t] = e

            # Reference power across taps per frequency bin: sum_l |X[k, t - l]|^2
            x_power = np.sum(np.abs(x_buf) ** 2, axis=1, keepdims=True)  # [n_freq, 1]

            # Normalized weight update
            denom = x_power + self.eps
            step = (self.mu / denom) * e[:, None]  # [n_freq, 1]
            self.w += step * np.conj(x_buf)

        return E


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    *,
    taps: int = DEFAULT_PARAMS["taps"],
    mu: float = DEFAULT_PARAMS["mu"],
    eps: float = DEFAULT_PARAMS["eps"],
) -> np.ndarray:
    """Enhance noisy audio using Normalized Least Mean Squares (NLMS) filtering.

    Requires a time-aligned reference microphone signal `reference`.
    If `reference` is None, adaptive filtering cannot be performed and a copy
    of `noisy` is returned.

    Args:
        noisy: 1D NumPy array [T] of primary noisy audio samples.
        reference: Optional 1D NumPy array [T] of reference noise samples.
            Must match `len(noisy)` when provided.
        sr: Sampling rate in Hz (defaults to SAMPLE_RATE = 16000).
        taps: Filter length in STFT frames (default: NLMS_TAPS = 4).
        mu: NLMS step size (default: NLMS_MU = 0.5).
        eps: Regularization parameter (default: NLMS_EPS = 1e-6).

    Returns:
        np.ndarray: Enhanced 1D audio array [T] with dtype np.float32.
    """
    del sr
    noisy_arr = np.asarray(noisy, dtype=np.float32)
    t_len = len(noisy_arr)

    # Edge cases: no reference, too short for STFT, or silent signal
    if reference is None:
        return noisy_arr.copy()

    ref_arr = np.asarray(reference, dtype=np.float32)
    if len(ref_arr) != t_len:
        raise ValueError(
            f"Reference length ({len(ref_arr)}) must equal noisy length ({t_len})"
        )

    min_length = N_FFT // 2 + 1
    if t_len < min_length or np.all(noisy_arr == 0):
        return noisy_arr.copy()

    # Time-frequency boundary: NumPy -> PyTorch [1, T] -> STFT
    stft_module = STFT()
    with torch.no_grad():
        d_tensor = stft_module.forward(torch.from_numpy(noisy_arr).unsqueeze(0))
        x_tensor = stft_module.forward(torch.from_numpy(ref_arr).unsqueeze(0))

    # Boundary: PyTorch -> NumPy complex64 [257, F]
    D = d_tensor.squeeze(0).cpu().numpy().astype(np.complex64)
    X = x_tensor.squeeze(0).cpu().numpy().astype(np.complex64)

    # Adaptive filtering in STFT domain
    nlms_filter = NLMSFilter(n_freq=N_FREQ, taps=taps, mu=mu, eps=eps)
    E = nlms_filter.process(X, D)

    # Boundary: NumPy -> PyTorch [1, 257, F] -> iSTFT
    e_tensor = torch.from_numpy(E).unsqueeze(0)
    with torch.no_grad():
        recon_tensor = stft_module.inverse(e_tensor, length=t_len)

    # Boundary: PyTorch -> NumPy float32 [T]
    enhanced_audio = recon_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    if not np.all(np.isfinite(enhanced_audio)):
        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)

    return enhanced_audio
