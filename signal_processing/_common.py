"""Common STFT utilities and noise PSD estimation for classical DSP."""

from __future__ import annotations

import numpy as np
from scipy import signal

N_FFT: int = 512
HOP: int = 160
WIN: int = 400
EPS: float = 1e-10


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
    """Estimate noise power spectral density across frequency bins."""
    if p.ndim != 2:
        raise ValueError(f"Expected 2-D power spectrogram [F, T], got shape {p.shape}")
    num_frames = p.shape[1]

    if mode == "initial":
        k = max(1, min(n_frames, num_frames))
        pn = np.mean(p[:, :k], axis=1, keepdims=True)
    elif mode == "percentile":
        pn = np.percentile(p, percentile, axis=1, keepdims=True)
    elif mode == "min_stats":
        pn = np.min(p, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown noise PSD estimation mode: {mode}")

    return pn.astype(np.float32)
