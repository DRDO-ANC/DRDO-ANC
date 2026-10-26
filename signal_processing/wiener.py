"""Wiener filter speech/audio enhancer."""

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
    xi_min: float = 10 ** (-25 / 10),
    noise_mode: str = "initial",
    decision_directed: bool = False,
    dd_alpha: float = 0.98,
    **params: object,
) -> np.ndarray:
    """Enhance noisy audio using Wiener filtering.

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

    if not decision_directed:
        gamma = p / (pn + EPS)
        xi = np.maximum(gamma - 1.0, xi_min)
        gain = xi / (1.0 + xi)
    else:
        num_freq, num_frames = p.shape
        gain = np.zeros_like(p, dtype=np.float32)
        prev_s_p = np.zeros(num_freq, dtype=np.float32)

        for l_idx in range(num_frames):
            p_l = p[:, l_idx]
            pn_l = pn[:, 0]
            gamma_l = p_l / (pn_l + EPS)

            if l_idx == 0:
                xi_l = np.maximum(gamma_l - 1.0, xi_min)
            else:
                xi_dd = dd_alpha * (prev_s_p / (pn_l + EPS)) + (1.0 - dd_alpha) * np.maximum(
                    gamma_l - 1.0, 0.0
                )
                xi_l = np.maximum(xi_dd, xi_min)

            gain_l = xi_l / (1.0 + xi_l)
            gain[:, l_idx] = gain_l
            prev_s_p = (gain_l**2) * p_l

    z_clean = gain * z
    return from_tf(z_clean, length=len(noisy_arr), sr=sample_rate)
