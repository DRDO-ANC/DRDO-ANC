"""Wiener filter speech enhancement baseline with decision-directed a priori SNR tracking.

Conforms to DRDO-ANC AI Coding Contract v1.
Uses ai_engine.models.stft.STFT for time-frequency boundary conversions.
"""

from typing import Any

import numpy as np
import torch

from ai_engine.models.stft import STFT
from common.constants import N_FFT, SAMPLE_RATE

# Immutable default hyperparameters
DEFAULT_PARAMS: dict[str, Any] = {
    "a": 0.98,
    "xi_min_db": -25.0,
    "percentile": 10.0,
    "eps": 1e-10,
}


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    *,
    a: float = DEFAULT_PARAMS["a"],
    xi_min_db: float = DEFAULT_PARAMS["xi_min_db"],
    percentile: float = DEFAULT_PARAMS["percentile"],
    eps: float = DEFAULT_PARAMS["eps"],
) -> np.ndarray:
    """Enhance a single-channel noisy audio signal using decision-directed Wiener filtering.

    Wiener gain is G = xi / (1 + xi), where a priori SNR (xi) is tracked across
    frames using the Ephraim-Malah decision-directed approach:
        xi_k = a * (|X_{k-1}|^2 / (N + eps)) + (1 - a) * max(gamma_k - 1, 0)
    with a floor applied at xi_min_db (default -25 dB).

    Note:
        The `reference` argument is ignored as this is a single-channel baseline.

    Args:
        noisy: 1D NumPy array [T] of noisy speech samples.
        reference: Ignored (single-channel baseline).
        sr: Sampling rate in Hz (defaults to SAMPLE_RATE=16000).
        a: Smoothing constant for decision-directed a priori SNR (default: 0.98).
        xi_min_db: Minimum a priori SNR floor in dB (default: -25.0 dB).
        percentile: Percentile for noise PSD estimation over time (default: 10.0).
        eps: Small epsilon for numerical stability.

    Returns:
        np.ndarray: Enhanced 1D audio array [T] with dtype np.float32.
    """
    del reference, sr  # Unused in single-channel baseline

    noisy_arr = np.asarray(noisy, dtype=np.float32)
    t_len = noisy_arr.shape[0]

    # Edge cases: empty, very short clip (STFT requires T > N_FFT // 2), or all zeros
    min_length = N_FFT // 2 + 1
    if t_len < min_length or np.all(noisy_arr == 0):
        return noisy_arr.copy()

    # Time-frequency boundary: NumPy -> PyTorch [1, T] -> STFT
    wave_tensor = torch.from_numpy(noisy_arr).unsqueeze(0)
    stft_module = STFT()
    with torch.no_grad():
        spec_tensor = stft_module.forward(wave_tensor)  # [1, N_FREQ, F]

    # Boundary: PyTorch -> NumPy complex64 [257, F]
    y_spec = spec_tensor.squeeze(0).cpu().numpy().astype(np.complex64)
    n_freq, n_frames = y_spec.shape

    y_power = np.abs(y_spec) ** 2

    # Noise PSD: 10th percentile over time frames per frequency bin [N_FREQ]
    noise_psd = np.percentile(y_power, percentile, axis=-1)  # shape [N_FREQ]
    noise_psd = np.maximum(noise_psd, eps)

    xi_min = 10.0 ** (xi_min_db / 10.0)

    # Frame-by-frame loop over F for decision-directed tracking
    enhanced_spec = np.zeros_like(y_spec)
    prev_clean_mag_sq = np.zeros(n_freq, dtype=np.float64)

    for k in range(n_frames):
        y_k_power = y_power[:, k].astype(np.float64)
        # A posteriori SNR: gamma_k = |Y_k|^2 / noise_psd
        gamma_k = y_k_power / noise_psd

        # A priori SNR estimation (decision-directed)
        if k == 0:
            xi_k = np.maximum(gamma_k - 1.0, 0.0)
        else:
            prior_est = prev_clean_mag_sq / noise_psd
            xi_k = a * prior_est + (1.0 - a) * np.maximum(gamma_k - 1.0, 0.0)

        # Floor at xi_min
        xi_k = np.maximum(xi_k, xi_min)

        # Wiener gain: G_k = xi_k / (1 + xi_k)
        gain_k = xi_k / (1.0 + xi_k)

        # Apply gain
        enhanced_k = gain_k * y_spec[:, k]
        enhanced_spec[:, k] = enhanced_k

        # Update previous frame power estimate
        prev_clean_mag_sq = np.abs(enhanced_k) ** 2

    # Boundary: NumPy -> PyTorch [1, 257, F] -> iSTFT
    enhanced_tensor = torch.from_numpy(enhanced_spec).unsqueeze(0)
    with torch.no_grad():
        recon_tensor = stft_module.inverse(enhanced_tensor, length=t_len)

    # Boundary: PyTorch -> NumPy float32 [T]
    enhanced_audio = recon_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    # Replace any unexpected non-finite numbers safely
    if not np.all(np.isfinite(enhanced_audio)):
        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)

    return enhanced_audio
