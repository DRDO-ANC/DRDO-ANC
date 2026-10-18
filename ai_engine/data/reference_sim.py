"""Reference microphone signal simulation for dual-channel ANC.

Conforms to DRDO-ANC AI Coding Contract v1 (Section D).
Generates an acoustic reference signal from the background noise source
using a short random delay, decaying FIR impulse response, and independent sensor noise.
Clean speech NEVER enters the reference microphone.
"""

import numpy as np

from common.constants import PEAK_NORM

# Module-level physical and simulation bounds
DELAY_MIN_SAMPLES: int = 0
DELAY_MAX_SAMPLES: int = 80  # 0 to 5 ms at 16000 Hz
TAPS_MIN: int = 8
TAPS_MAX: int = 32
INDEPENDENT_NOISE_SNR_MIN_DB: float = 20.0
INDEPENDENT_NOISE_SNR_MAX_DB: float = 30.0


def simulate_reference(
    noise: np.ndarray,
    rng: np.random.Generator,
    *,
    delay_min: int = DELAY_MIN_SAMPLES,
    delay_max: int = DELAY_MAX_SAMPLES,
    taps_min: int = TAPS_MIN,
    taps_max: int = TAPS_MAX,
    noise_snr_min_db: float = INDEPENDENT_NOISE_SNR_MIN_DB,
    noise_snr_max_db: float = INDEPENDENT_NOISE_SNR_MAX_DB,
    eps: float = 1e-10,
) -> np.ndarray:
    """Simulate a reference microphone signal from the noise source.

    Recipe:
      1. Acoustic delay: A random delay of 0 to 80 samples (0 to 5 ms at 16 kHz).
         This delay is far below the 32 ms STFT analysis window, ensuring
         that STFT-domain adaptive filters (e.g. NLMS) can model the path causally.
      2. Multi-path FIR filter: A short FIR filter of 8 to 32 taps with decaying
         magnitude: h[n] = exp(-decay * n) * v[n], normalized to unit energy.
      3. Sensor noise: Independent additive Gaussian sensor noise scaled to be
         20 to 30 dB below the filtered noise signal.
      4. Strict isolation: Clean speech must NEVER be passed or leaked into
         the reference microphone.

    Args:
        noise: 1D NumPy array representing the acoustic noise source at the primary mic.
        rng: NumPy random generator instance for reproducible sampling.
        delay_min: Minimum acoustic delay in samples (default: 0).
        delay_max: Maximum acoustic delay in samples (default: 80).
        taps_min: Minimum FIR filter taps (default: 8).
        taps_max: Maximum FIR filter taps (default: 32).
        noise_snr_min_db: Lower bound for independent sensor noise SNR in dB (default: 20.0).
        noise_snr_max_db: Upper bound for independent sensor noise SNR in dB (default: 30.0).
        eps: Small epsilon to prevent division by zero on silent inputs.

    Returns:
        np.ndarray: Simulated reference signal of same length as noise, dtype np.float32.
    """
    noise_arr = np.asarray(noise, dtype=np.float64)
    length = len(noise_arr)

    if length == 0:
        return np.zeros(0, dtype=np.float32)

    noise_pow = np.mean(noise_arr**2)
    if noise_pow <= eps:
        return np.zeros(length, dtype=np.float32)

    # 1. Random acoustic delay
    delay = int(rng.integers(delay_min, delay_max + 1))

    # 2. Random decaying FIR filter
    n_taps = int(rng.integers(taps_min, taps_max + 1))
    decay = rng.uniform(0.05, 0.2)
    t = np.arange(n_taps, dtype=np.float64)
    envelope = np.exp(-decay * t)
    raw_fir = envelope * rng.normal(0, 1.0, n_taps)
    # Energy normalization
    fir_energy = np.sum(raw_fir**2)
    if fir_energy > eps:
        fir = raw_fir / np.sqrt(fir_energy)
    else:
        fir = np.zeros(n_taps, dtype=np.float64)
        fir[0] = 1.0

    # Filter noise
    filtered = np.convolve(noise_arr, fir, mode="full")[:length]

    # Apply delay
    if delay > 0:
        delayed = np.pad(filtered, (delay, 0), mode="constant")[:length]
    else:
        delayed = filtered

    # 3. Add low-level independent sensor noise (20 to 30 dB below signal)
    sensor_snr_db = rng.uniform(noise_snr_min_db, noise_snr_max_db)
    delayed_pow = np.mean(delayed**2)
    if delayed_pow > eps:
        sensor_noise_target_pow = delayed_pow / (10.0 ** (sensor_snr_db / 10.0))
        raw_sensor_noise = rng.normal(0, 1.0, length)
        sensor_noise_scale = np.sqrt(sensor_noise_target_pow / (np.mean(raw_sensor_noise**2) + eps))
        sensor_noise = raw_sensor_noise * sensor_noise_scale
        reference = delayed + sensor_noise
    else:
        reference = delayed

    # 4. Rescale reference only if it would exceed PEAK_NORM
    ref_peak = float(np.max(np.abs(reference)))
    if ref_peak > PEAK_NORM:
        reference = reference * (PEAK_NORM / ref_peak)

    return reference.astype(np.float32)
