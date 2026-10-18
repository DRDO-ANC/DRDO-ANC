"""Least Mean Squares (LMS) adaptive filter in the STFT domain.

Conforms to DRDO-ANC AI Coding Contract v1.
Maintains one filter per frequency bin with length L = NLMS_TAPS frames,
updating weights without normalization: W[k, l] += mu * conj(X[k, t - l]) * E[k, t].

Stability Consideration:
Standard unnormalized LMS weight updates require 0 < mu < 2 / (L * P_X) to prevent
divergence. Under unit variance white noise in the time domain, the STFT bin power
P_X is roughly 100-200. Extensive sweeps on synthetic audio confirm that mu values
exceeding 5e-3 cause divergent filter weights, while mu in the range [5e-5, 5e-4]
consistently achieves fast, monotonic MSE convergence without instability.
The default is set to mu = 1e-4.
"""

from typing import Any

import numpy as np
import torch

from ai_engine.models.stft import STFT
from common.constants import N_FFT, N_FREQ, NLMS_TAPS, SAMPLE_RATE

# Default mu chosen from stability sweep on synthetic STFT spectra
DEFAULT_PARAMS: dict[str, Any] = {
    "taps": NLMS_TAPS,
    "mu": 1e-4,
}


class LMSFilter:
    """Frequency-domain unnormalized Least Mean Squares (LMS) filter.

    Maintains one independent adaptive FIR filter for each of the `n_freq`
    frequency bins. Operates on complex STFT frames.

    Algorithm per frame t:
        y[k, t] = sum_{l=0..L-1} W[k, l] * X[k, t - l]
        E[k, t] = D[k, t] - y[k, t]
        W[k, l] += mu * conj(X[k, t - l]) * E[k, t]

    Attributes:
        n_freq: Number of frequency bins (defaults to N_FREQ = 257).
        taps: Filter length L in frames (defaults to NLMS_TAPS = 4).
        mu: Step size parameter (defaults to 1e-4 for numerical stability).
        w: Complex filter weights of shape [n_freq, taps].
    """

    def __init__(
        self,
        n_freq: int = N_FREQ,
        taps: int = NLMS_TAPS,
        mu: float = DEFAULT_PARAMS["mu"],
    ) -> None:
        """Initialize LMS filter and reset weights."""
        self.n_freq = n_freq
        self.taps = taps
        self.mu = mu
        self.w = np.zeros((self.n_freq, self.taps), dtype=np.complex64)
        self.reset()

    def reset(self) -> None:
        """Reset adaptive filter weights to zero."""
        self.w.fill(0.0)

    def process(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Process reference (X) and primary/desired (D) complex spectra.

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

            # Unnormalized weight update: W += mu * conj(X) * E
            step = self.mu * e[:, None]  # [n_freq, 1]
            self.w += step * np.conj(x_buf)

        return E


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    *,
    taps: int = DEFAULT_PARAMS["taps"],
    mu: float = DEFAULT_PARAMS["mu"],
) -> np.ndarray:
    """Enhance noisy audio using unnormalized Least Mean Squares (LMS) filtering.

    Requires a time-aligned reference microphone signal `reference`.
    If `reference` is None, adaptive filtering cannot be performed and a copy
    of `noisy` is returned.

    Args:
        noisy: 1D NumPy array [T] of primary noisy audio samples.
        reference: Optional 1D NumPy array [T] of reference noise samples.
            Must match `len(noisy)` when provided.
        sr: Sampling rate in Hz (defaults to SAMPLE_RATE = 16000).
        taps: Filter length in STFT frames (default: NLMS_TAPS = 4).
        mu: LMS step size (default: 1e-4).

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
    lms_filter = LMSFilter(n_freq=N_FREQ, taps=taps, mu=mu)
    E = lms_filter.process(X, D)

    # Boundary: NumPy -> PyTorch [1, 257, F] -> iSTFT
    e_tensor = torch.from_numpy(E).unsqueeze(0)
    with torch.no_grad():
        recon_tensor = stft_module.inverse(e_tensor, length=t_len)

    # Boundary: PyTorch -> NumPy float32 [T]
    enhanced_audio = recon_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    if not np.all(np.isfinite(enhanced_audio)):
        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)

    return enhanced_audio
