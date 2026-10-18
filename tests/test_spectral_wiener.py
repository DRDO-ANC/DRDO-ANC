"""Tests for spectral subtraction and Wiener filtering enhancement algorithms.

All synthetic signals use reproducible RNG seeded with SEED from common.constants.
SNR is measured using ai_engine.evaluation.metrics.snr.
"""

import numpy as np
import pytest

from ai_engine.evaluation.metrics import snr
from common.constants import CLIP_SAMPLES, SAMPLE_RATE, SEED
from signal_processing.spectral import DEFAULT_PARAMS as SPECTRAL_DEFAULT_PARAMS
from signal_processing.spectral import enhance as spectral_enhance
from signal_processing.wiener import DEFAULT_PARAMS as WIENER_DEFAULT_PARAMS
from signal_processing.wiener import enhance as wiener_enhance


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_noisy_speech(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create a 440 Hz tone with speech-like activity + white noise at 0 dB SNR (4 s).

    Because noise PSD estimation relies on the 10th percentile over time frames per bin,
    the speech/tone signal has active and inactive periods (speech-like voice activity).
    """
    t = np.linspace(0, CLIP_SAMPLES / SAMPLE_RATE, CLIP_SAMPLES, endpoint=False)
    # 50% duty cycle activity envelope (1 Hz modulation)
    activity = (np.sin(2 * np.pi * 1.0 * t) > 0).astype(np.float32)
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float32) * activity

    # 0 dB SNR: noise power equals signal power
    signal_power = np.mean(clean**2)
    raw_noise = rng.normal(0, 1.0, CLIP_SAMPLES).astype(np.float32)
    noise_power = np.mean(raw_noise**2)
    noise = raw_noise * np.sqrt(signal_power / noise_power)

    noisy = clean + noise
    return clean, noisy


def test_default_params_immutable() -> None:
    """Ensure default param dictionaries exist with expected keys."""
    assert "alpha" in SPECTRAL_DEFAULT_PARAMS
    assert "beta" in SPECTRAL_DEFAULT_PARAMS
    assert "a" in WIENER_DEFAULT_PARAMS
    assert "xi_min_db" in WIENER_DEFAULT_PARAMS


def test_spectral_enhancement_improves_snr(
    synthetic_noisy_speech: tuple[np.ndarray, np.ndarray],
) -> None:
    """Spectral subtraction should improve SNR on 0 dB tone + noise."""
    clean, noisy = synthetic_noisy_speech
    input_snr = snr(clean, noisy)

    enhanced = spectral_enhance(noisy)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))

    output_snr = snr(clean, enhanced)
    assert output_snr > input_snr


def test_wiener_enhancement_improves_snr(
    synthetic_noisy_speech: tuple[np.ndarray, np.ndarray],
) -> None:
    """Wiener filtering should improve SNR on 0 dB tone + noise."""
    clean, noisy = synthetic_noisy_speech
    input_snr = snr(clean, noisy)

    enhanced = wiener_enhance(noisy)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))

    output_snr = snr(clean, enhanced)
    assert output_snr > input_snr


def test_edge_cases_silence_and_short_clips() -> None:
    """Ensure both methods handle silent and very short inputs safely."""
    # Silent input
    silent = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    out_spec = spectral_enhance(silent)
    out_wiener = wiener_enhance(silent)

    assert out_spec.shape == silent.shape
    assert out_wiener.shape == silent.shape
    assert np.all(np.isfinite(out_spec))
    assert np.all(np.isfinite(out_wiener))

    # Very short input (< 256 samples)
    short_signal = np.ones(100, dtype=np.float32) * 0.1
    short_spec = spectral_enhance(short_signal)
    short_wiener = wiener_enhance(short_signal)

    assert short_spec.shape == (100,)
    assert short_wiener.shape == (100,)
    assert np.all(np.isfinite(short_spec))
    assert np.all(np.isfinite(short_wiener))
