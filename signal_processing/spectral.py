"""Magnitude spectral subtraction speech enhancement baseline.

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
    "alpha": 2.0,
    "beta": 0.02,
    "percentile": 10.0,
    "eps": 1e-10,
}


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    *,
    alpha: float = DEFAULT_PARAMS["alpha"],
    beta: float = DEFAULT_PARAMS["beta"],
    percentile: float = DEFAULT_PARAMS["percentile"],
    eps: float = DEFAULT_PARAMS["eps"],
) -> np.ndarray:
    """Enhance a single-channel noisy audio signal using magnitude spectral subtraction.

    Noise PSD is estimated as the 10th percentile of |Y|^2 over time frames per
    frequency bin. Clean PSD is estimated as:
        clean_psd = max(|Y|^2 - alpha * N, beta * N)
    and resynthesized using the noisy phase.

    Note:
        The `reference` argument is ignored as this is a single-channel baseline.

    Args:
        noisy: 1D NumPy array [T] of noisy speech samples.
        reference: Ignored (single-channel baseline).
        sr: Sampling rate in Hz (defaults to SAMPLE_RATE=16000).
        alpha: Over-subtraction factor (default: 2.0).
        beta: Spectral floor factor (default: 0.02).
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

    # Compute magnitude and phase
    y_mag = np.abs(y_spec)
    y_power = y_mag**2
    y_phase = np.angle(y_spec)

    # Estimate noise PSD: 10th percentile over time frames (axis -1) per bin
    noise_psd = np.percentile(y_power, percentile, axis=-1, keepdims=True)

    # Spectral subtraction with over-subtraction and floor
    clean_psd = np.maximum(y_power - alpha * noise_psd, beta * noise_psd)
    clean_mag = np.sqrt(np.maximum(clean_psd, eps))

    # Reconstruct complex spectrum using original noisy phase
    enhanced_spec = clean_mag * np.exp(1j * y_phase)

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
