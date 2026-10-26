"""Spectral subtraction speech/audio enhancer."""

from __future__ import annotations

import numpy as np

from signal_processing._common import (
    EPS,
    from_tf,
    noise_psd,
    to_tf,
    validate_sr,
    validate_waveform,
)


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = 16000,
    alpha: float = 2.0,
    beta: float = 0.02,
    noise_mode: str = "initial",
    **params: object,
) -> np.ndarray:
    """Enhance noisy audio using spectral subtraction.

    Noise PSD estimation modes (`noise_mode`):
    - "initial": Average power over the first `n_frames` (default 25).
    - "percentile": Per-bin percentile across all frames (default 10th percentile).
    - "min_stats": Per-bin minimum power across the entire clip. This is a simplification
      (no sliding window or running minimum tracker) that suits stationary noise but can
      underestimate non-stationary noise.
    """
    _ = reference, params
    noisy_arr = validate_waveform(noisy, name="noisy")
    sample_rate = validate_sr(sr)

    _, _, z = to_tf(noisy_arr, sr=sample_rate)
    p = np.abs(z) ** 2
    pn = noise_psd(p, mode=noise_mode)

    clean_p = np.maximum(p - alpha * pn, beta * pn)
    gain = np.sqrt(clean_p) / (np.sqrt(p) + EPS)

    z_clean = gain * z
    return from_tf(z_clean, length=len(noisy_arr), sr=sample_rate)
