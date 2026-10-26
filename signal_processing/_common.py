"""Common STFT utilities and noise PSD estimation for classical DSP."""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from scipy import signal

N_FFT: int = 512
HOP: int = 160
WIN: int = 400
EPS: float = 1e-10


def validate_waveform(x: Any, name: str = "noisy") -> np.ndarray:
    """Validate a 1-D integer or floating-point waveform and return float32."""
    arr = np.asarray(x)

    if not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(
            f"{name} must be a real numeric array (int or float), got dtype {arr.dtype}"
        )

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")

    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    with np.errstate(over="ignore"):
        out = arr.astype(np.float32)

    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"{name} contains non-finite values (NaN, inf, or values too large for float32)"
        )

    return out


def validate_sr(sr: object) -> int:
    """Validate a positive integer sample rate in Hz."""
    if (
        isinstance(sr, (bool, np.bool_))
        or not isinstance(sr, (int, np.integer))
        or sr <= 0
    ):
        raise ValueError(f"sr must be a positive integer (Hz), got {sr!r}")
    return int(sr)


def to_tf(x: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform of 1-D waveform using a Hann window."""
    f, t, z = signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=WIN,
        noverlap=WIN - HOP,
        nfft=N_FFT,
        boundary="zeros",
        padded=True,
    )
    return f, t, z


def from_tf(z: np.ndarray, length: int, sr: int = 16000) -> np.ndarray:
    """Invert STFT back to 1-D waveform, trimmed or padded to length in float32."""
    _, x_rec = signal.istft(
        z,
        fs=sr,
        window="hann",
        nperseg=WIN,
        noverlap=WIN - HOP,
        nfft=N_FFT,
        boundary="zeros",
    )
    x_rec = np.asarray(x_rec, dtype=np.float32)
    if len(x_rec) < length:
        out = np.zeros(length, dtype=np.float32)
        out[: len(x_rec)] = x_rec
        return out
    return x_rec[:length].astype(np.float32)


def noise_psd(
    p: np.ndarray,
    mode: str = "initial",
    n_frames: int = 25,
    percentile: float = 10.0,
) -> np.ndarray:
    """Estimate noise power spectral density across frequency bins.

    Supported modes:
    - "initial": Average power over the first `n_frames` (clamped to available frames).
    - "percentile": Per-bin percentile across all frames specified by `percentile`.
    - "min_stats": Per-bin minimum power across the entire clip. This is a simplification
      (no sliding window or running minimum tracker) that suits stationary noise but can
      underestimate non-stationary noise.
    """
    if not isinstance(p, np.ndarray) or p.ndim != 2:
        shape_info = getattr(p, "shape", None)
        raise ValueError(f"Expected 2-D power spectrogram [F, T], got shape {shape_info}")

    num_bins, num_frames = p.shape
    if num_bins == 0 or num_frames == 0:
        raise ValueError(
            f"Power spectrogram must have non-zero bins and frames, got shape {p.shape}"
        )

    if not np.all(np.isfinite(p)):
        raise ValueError("Power spectrogram contains non-finite values (NaN or inf)")

    if np.any(p < 0.0):
        raise ValueError("Power spectrogram cannot contain negative values")

    # Validate n_frames: must be int (not bool) >= 1
    if isinstance(n_frames, bool) or not isinstance(n_frames, numbers.Integral) or n_frames < 1:
        raise ValueError(
            f"n_frames must be an integer (not bool) >= 1, got {n_frames!r} of type {type(n_frames).__name__}"
        )

    # Validate percentile: must be a finite number with 0 <= percentile <= 100
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, numbers.Real)
        or not np.isfinite(percentile)
        or not (0.0 <= float(percentile) <= 100.0)
    ):
        raise ValueError(
            f"percentile must be a finite real number between 0 and 100, got {percentile!r}"
        )

    if mode == "initial":
        k = max(1, min(int(n_frames), num_frames))
        pn = np.mean(p[:, :k], axis=1, keepdims=True)
    elif mode == "percentile":
        pn = np.percentile(p, float(percentile), axis=1, keepdims=True)
    elif mode == "min_stats":
        pn = np.min(p, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown noise PSD estimation mode: {mode}")

    return pn.astype(np.float32)
