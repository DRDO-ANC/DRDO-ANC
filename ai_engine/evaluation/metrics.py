"""Speech enhancement and evaluation metrics.

Provides standard evaluation metrics for DRDO-ANC conforming to AI Coding Contract v1.
Functions operate on 1D NumPy float arrays, compute with float64 precision,
and return standard Python float values.
"""

import logging

import numpy as np
import pystoi

try:
    import pesq
except ImportError:
    pesq = None  # type: ignore

from common.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)


def snr(clean: np.ndarray, enhanced: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels (dB).

    Formula:
        10 * log10(sum(clean^2) / (sum((clean - enhanced)^2) + eps))

    Args:
        clean: 1D array of ground-truth clean audio signal.
        enhanced: 1D array of processed/enhanced audio signal.
        eps: Small epsilon to prevent division by zero or log of zero.

    Returns:
        float: SNR value in dB.
    """
    clean_64 = np.asarray(clean, dtype=np.float64)
    enhanced_64 = np.asarray(enhanced, dtype=np.float64)

    signal_power = np.sum(clean_64**2)
    noise_power = np.sum((clean_64 - enhanced_64) ** 2)

    val = (signal_power + eps) / (noise_power + eps)
    return float(10.0 * np.log10(val))


def si_snr(clean: np.ndarray, enhanced: np.ndarray, eps: float = 1e-10) -> float:
    """Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in decibels (dB).

    Both signals are zero-meaned prior to orthogonal projection.
    The enhanced signal is decomposed into parallel (target) and orthogonal (error)
    components relative to the clean reference:
        e_target = (<s_clean, s_enhanced> / (||s_clean||^2 + eps)) * s_clean
        e_noise = s_enhanced - e_target
        SI-SNR = 10 * log10(||e_target||^2 / (||e_noise||^2 + eps))

    Args:
        clean: 1D array of ground-truth clean audio signal.
        enhanced: 1D array of processed/enhanced audio signal.
        eps: Small epsilon to prevent division by zero or log of zero.

    Returns:
        float: SI-SNR value in dB.
    """
    s = np.asarray(clean, dtype=np.float64)
    s_hat = np.asarray(enhanced, dtype=np.float64)

    # Zero-mean both signals
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)

    dot = np.dot(s, s_hat)
    s_norm_sq = np.sum(s**2)

    target = (dot / (s_norm_sq + eps)) * s
    noise = s_hat - target

    target_pow = np.sum(target**2)
    noise_pow = np.sum(noise**2)

    val = (target_pow + eps) / (noise_pow + eps)
    return float(10.0 * np.log10(val))


def ssnr(
    clean: np.ndarray,
    enhanced: np.ndarray,
    frame_length: int | None = None,
    min_snr_db: float = -10.0,
    max_snr_db: float = 35.0,
    eps: float = 1e-10,
) -> float:
    """Compute Segmental Signal-to-Noise Ratio (SSNR) in decibels (dB).

    The signals are divided into non-overlapping frames of 20 ms duration
    (320 samples at SAMPLE_RATE=16000 Hz by default). For each frame, the local
    SNR in dB is evaluated and clipped to the interval [min_snr_db, max_snr_db]
    (default: [-10 dB, +35 dB]) to avoid biases from silence or near-zero noise.
    The final score is the arithmetic mean across all valid complete frames.

    Args:
        clean: 1D array of ground-truth clean audio signal.
        enhanced: 1D array of processed/enhanced audio signal.
        frame_length: Number of samples per frame. Defaults to 20 ms
            (int(0.020 * SAMPLE_RATE) = 320 samples).
        min_snr_db: Lower bound for clipping per-frame SNR in dB (-10.0 dB).
        max_snr_db: Upper bound for clipping per-frame SNR in dB (+35.0 dB).
        eps: Small epsilon to prevent division by zero or log of zero.

    Returns:
        float: Segmental SNR in dB averaged over frames.
    """
    clean_64 = np.asarray(clean, dtype=np.float64)
    enhanced_64 = np.asarray(enhanced, dtype=np.float64)

    if frame_length is None:
        frame_length = int(0.020 * SAMPLE_RATE)

    n_samples = min(len(clean_64), len(enhanced_64))
    n_frames = n_samples // frame_length

    if n_frames == 0:
        return snr(clean_64, enhanced_64, eps=eps)

    frame_snrs: list[float] = []
    for i in range(n_frames):
        start = i * frame_length
        end = start + frame_length
        c_frame = clean_64[start:end]
        e_frame = enhanced_64[start:end]

        sig_pow = np.sum(c_frame**2)
        noise_pow = np.sum((c_frame - e_frame) ** 2)

        ratio = (sig_pow + eps) / (noise_pow + eps)
        frame_snr = 10.0 * np.log10(ratio)
        frame_snr_clipped = np.clip(frame_snr, min_snr_db, max_snr_db)
        frame_snrs.append(float(frame_snr_clipped))

    return float(np.mean(frame_snrs))


def stoi_score(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Short-Time Objective Intelligibility (STOI) score.

    Uses `pystoi.stoi` with sample rate SAMPLE_RATE and standard (non-extended) STOI.

    Args:
        clean: 1D array of clean audio signal.
        enhanced: 1D array of enhanced audio signal.

    Returns:
        float: STOI score between 0.0 and 1.0.
    """
    clean_64 = np.asarray(clean, dtype=np.float64)
    enhanced_64 = np.asarray(enhanced, dtype=np.float64)
    return float(pystoi.stoi(clean_64, enhanced_64, SAMPLE_RATE, extended=False))


def pesq_score(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Perceptual Evaluation of Speech Quality (PESQ) score (wideband).

    Uses `pesq.pesq(SAMPLE_RATE, clean, enhanced, 'wb')`. If the evaluation fails
    (e.g., due to short audio, silence, or alignment issues), failures are logged
    via `logging.warning` and `float('nan')` is returned without raising an exception.

    Args:
        clean: 1D array of clean audio signal.
        enhanced: 1D array of enhanced audio signal.

    Returns:
        float: PESQ wideband score (typically between 1.0 and 4.5), or NaN on error.
    """
    clean_64 = np.asarray(clean, dtype=np.float64)
    enhanced_64 = np.asarray(enhanced, dtype=np.float64)

    if pesq is None:
        logger.warning("PESQ calculation failed: pesq package is not installed.")
        return float("nan")

    try:
        score = pesq.pesq(SAMPLE_RATE, clean_64, enhanced_64, "wb")
        return float(score)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PESQ calculation failed: %s", exc)
        return float("nan")


def rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Compute Real-Time Factor (RTF).

    RTF < 1.0 indicates that processing is faster than real time.

    Args:
        processing_seconds: Elapsed processing wall-clock time in seconds.
        audio_seconds: Length of processed audio signal in seconds.

    Returns:
        float: Real-Time Factor (processing_seconds / audio_seconds).
    """
    return float(processing_seconds / audio_seconds)
