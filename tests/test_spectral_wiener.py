"""Tests for classical spectral subtraction and Wiener filtering enhancers."""

from __future__ import annotations

import numpy as np
import pytest

from signal_processing import spectral, wiener
from signal_processing._common import from_tf, noise_psd, to_tf


def compute_snr_db(clean: np.ndarray, noisy_or_enhanced: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in decibels."""
    err = noisy_or_enhanced - clean
    s_pwr = float(np.mean(clean**2))
    n_pwr = float(np.mean(err**2))
    return float(10.0 * np.log10(s_pwr / (n_pwr + 1e-12)))


def generate_test_audio(
    sr: int = 16000,
    tone_duration: float = 3.0,
    lead_in: float = 0.5,
    tone_freq: float = 1000.0,
    snr_db: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 1 kHz tone with noise-only lead-in and 0 dB white noise."""
    rng = np.random.default_rng(seed)
    total_duration = lead_in + tone_duration
    total_samples = int(total_duration * sr)
    lead_in_samples = int(lead_in * sr)

    t = np.arange(total_samples, dtype=np.float32) / sr
    clean = np.zeros(total_samples, dtype=np.float32)
    clean[lead_in_samples:] = 0.5 * np.sin(2.0 * np.pi * tone_freq * t[lead_in_samples:])

    noise = rng.normal(0.0, 1.0, size=total_samples).astype(np.float32)
    s_pwr = np.mean(clean[lead_in_samples:] ** 2)
    n_pwr = np.mean(noise**2)
    scale = np.sqrt(s_pwr / (n_pwr * (10.0 ** (snr_db / 10.0))))
    noise = (noise * scale).astype(np.float32)

    noisy = clean + noise
    # Keep audio within [-1, 1] bounds
    max_val = max(float(np.max(np.abs(noisy))), 1.0)
    clean = (clean / max_val).astype(np.float32)
    noisy = (noisy / max_val).astype(np.float32)
    return clean, noisy


@pytest.mark.parametrize("module", [spectral, wiener])
def test_snr_improvement(module) -> None:
    """Verify that enhance achieves > 3 dB SNR improvement over input."""
    clean, noisy = generate_test_audio()
    in_snr = compute_snr_db(clean, noisy)

    enhanced = module.enhance(noisy, sr=16000)
    out_snr = compute_snr_db(clean, enhanced)

    assert out_snr > in_snr + 3.0, f"{module.__name__} SNR gain {out_snr - in_snr:.2f} dB <= 3 dB"


@pytest.mark.parametrize("module", [spectral, wiener])
@pytest.mark.parametrize("noise_mode", ["initial", "percentile", "min_stats"])
def test_all_noise_modes_run(module, noise_mode: str) -> None:
    """Verify that all noise modes run without error and return matching shape/dtype."""
    _clean, noisy = generate_test_audio()
    enhanced = module.enhance(noisy, sr=16000, noise_mode=noise_mode)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape


@pytest.mark.parametrize("module", [spectral, wiener])
@pytest.mark.parametrize("length", [1000, 16000, 40000])
def test_output_dtype_and_length(module, length: int) -> None:
    """Verify enhance returns float32 and exact matching length for various input sizes."""
    rng = np.random.default_rng(42)
    noisy = rng.uniform(-0.9, 0.9, size=length).astype(np.float32)

    enhanced = module.enhance(noisy, sr=16000)
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == (length,)


@pytest.mark.parametrize("module", [spectral, wiener])
def test_bad_noise_mode_raises_value_error(module) -> None:
    """Verify bad noise mode raises ValueError in enhance."""
    noisy = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown noise PSD estimation mode"):
        module.enhance(noisy, noise_mode="unsupported_mode")


def test_noise_psd_unknown_mode_raises_value_error() -> None:
    """Verify noise_psd directly raises ValueError on unknown mode."""
    p = np.ones((257, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown noise PSD estimation mode"):
        noise_psd(p, mode="invalid")


def test_wiener_decision_directed() -> None:
    """Verify Wiener enhancer with decision-directed approach improves SNR and preserves shape."""
    clean, noisy = generate_test_audio()
    in_snr = compute_snr_db(clean, noisy)

    enhanced = wiener.enhance(noisy, sr=16000, decision_directed=True)
    out_snr = compute_snr_db(clean, enhanced)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert out_snr > in_snr + 3.0


def test_stft_istft_roundtrip() -> None:
    """Verify to_tf and from_tf reconstruct the waveform with low error."""
    rng = np.random.default_rng(42)
    for length in (1000, 16000, 40000):
        x = rng.uniform(-0.8, 0.8, size=length).astype(np.float32)
        _, _, z = to_tf(x)
        rec = from_tf(z, length=length)
        assert rec.dtype == np.float32
        assert rec.shape == (length,)
        assert np.max(np.abs(x - rec)) < 1e-4
