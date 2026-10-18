"""Audio mixing module for DRDO-ANC synthetic dataset generation.

Conforms to DRDO-ANC AI Coding Contract v1 (Section E).
Takes clean speech and noise NumPy arrays, adjusts duration to CLIP_SAMPLES,
scales noise to the exact target SNR, and applies joint peak normalisation.
"""

import numpy as np

from common.constants import CLIP_SAMPLES, PEAK_NORM


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix clean speech and noise at a specific SNR level in dB.

    Steps:
      1. Crop or zero-pad clean signal to CLIP_SAMPLES. Random crop uses `rng`.
      2. If noise is shorter than CLIP_SAMPLES, tile cyclically to length CLIP_SAMPLES.
         If longer, random crop to CLIP_SAMPLES using `rng`.
      3. Scale noise to match exact target SNR relative to clean speech.
         Guarded against silent noise/speech using eps.
      4. Joint peak normalisation: If max(|mixed|) > PEAK_NORM (0.99), multiply
         mixed, clean, and noise by the exact same scale factor, perfectly
         preserving the target SNR.

    Args:
        clean: 1D NumPy array representing clean speech.
        noise: 1D NumPy array representing background noise.
        snr_db: Target Signal-to-Noise Ratio in decibels (dB).
        rng: NumPy random generator instance for reproducible cropping.
        eps: Epsilon to prevent division by zero on silent signals.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (mixed, clean, noise) as 1D arrays of shape (CLIP_SAMPLES,) and dtype np.float32.
    """
    clean_arr = np.asarray(clean, dtype=np.float64)
    noise_arr = np.asarray(noise, dtype=np.float64)

    # 1. Process clean signal to CLIP_SAMPLES
    len_clean = len(clean_arr)
    if len_clean > CLIP_SAMPLES:
        start = int(rng.integers(0, len_clean - CLIP_SAMPLES + 1))
        clean_proc = clean_arr[start : start + CLIP_SAMPLES].copy()
    elif len_clean < CLIP_SAMPLES:
        clean_proc = np.pad(clean_arr, (0, CLIP_SAMPLES - len_clean), mode="constant")
    else:
        clean_proc = clean_arr.copy()

    # 2. Process noise signal to CLIP_SAMPLES
    len_noise = len(noise_arr)
    if len_noise < CLIP_SAMPLES:
        if len_noise == 0:
            noise_proc = np.zeros(CLIP_SAMPLES, dtype=np.float64)
        else:
            repeats = int(np.ceil(CLIP_SAMPLES / len_noise))
            tiled = np.tile(noise_arr, repeats)
            noise_proc = tiled[:CLIP_SAMPLES].copy()
    elif len_noise > CLIP_SAMPLES:
        start_n = int(rng.integers(0, len_noise - CLIP_SAMPLES + 1))
        noise_proc = noise_arr[start_n : start_n + CLIP_SAMPLES].copy()
    else:
        noise_proc = noise_arr.copy()

    # 3. Scale noise to target SNR
    clean_power = np.mean(clean_proc**2)
    noise_power = np.mean(noise_proc**2)

    # Guard against silent signals
    if noise_power > eps and clean_power > eps:
        target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
        scale_factor = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise_proc * scale_factor
    else:
        scaled_noise = noise_proc

    mixed = clean_proc + scaled_noise

    # 4. Joint peak normalisation
    peak = float(np.max(np.abs(mixed)))
    if peak > PEAK_NORM:
        norm_factor = PEAK_NORM / peak
        mixed = mixed * norm_factor
        clean_proc = clean_proc * norm_factor
        scaled_noise = scaled_noise * norm_factor

    return (
        mixed.astype(np.float32),
        clean_proc.astype(np.float32),
        scaled_noise.astype(np.float32),
    )
