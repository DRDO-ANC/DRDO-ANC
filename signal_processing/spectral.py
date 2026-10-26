"""Spectral subtraction speech/audio enhancer."""

from __future__ import annotations

import numpy as np

from signal_processing._common import EPS, from_tf, noise_psd, to_tf


def enhance(
    noisy: np.ndarray,
    reference: np.ndarray | None = None,
    sr: int = 16000,
    alpha: float = 2.0,
    beta: float = 0.02,
    noise_mode: str = "initial",
    **params: object,
) -> np.ndarray:
    """Enhance noisy audio using spectral subtraction."""
    _ = reference, params
    if noisy.size == 0:
        return np.empty(0, dtype=np.float32)

    _, _, z = to_tf(noisy, sr=sr)
    p = np.abs(z) ** 2
    pn = noise_psd(p, mode=noise_mode)

    clean_p = np.maximum(p - alpha * pn, beta * pn)
    gain = np.sqrt(clean_p) / (np.sqrt(p) + EPS)

    z_clean = gain * z
    return from_tf(z_clean, length=len(noisy), sr=sr)
