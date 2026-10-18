"""Tests for STFT-domain LMS and NLMS adaptive filtering algorithms.

All synthetic tests use reproducible RNG seeded with SEED from common.constants.
Does not write or commit any WAV or image files.
"""

import numpy as np
import pytest

from common.constants import N_FREQ, NLMS_TAPS, SAMPLE_RATE, SEED
from signal_processing.lms import LMSFilter
from signal_processing.lms import enhance as lms_enhance
from signal_processing.nlms import NLMSFilter
from signal_processing.nlms import enhance as nlms_enhance


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG seeded with SEED."""
    return np.random.default_rng(SEED)


def test_nlms_filter_process_direct(rng: np.random.Generator) -> None:
    """Test NLMSFilter.process directly on random complex inputs.

    Output shape must be [257, F], dtype complex64, no NaN.
    With X = 0, the output must equal D identically.
    """
    n_freq = N_FREQ
    n_frames = 20

    # Random complex input
    X = (rng.normal(0, 1.0, (n_freq, n_frames)) + 1j * rng.normal(0, 1.0, (n_freq, n_frames))).astype(np.complex64)
    D = (rng.normal(0, 1.0, (n_freq, n_frames)) + 1j * rng.normal(0, 1.0, (n_freq, n_frames))).astype(np.complex64)

    filter_nlms = NLMSFilter(n_freq=n_freq, taps=NLMS_TAPS)
    E = filter_nlms.process(X, D)

    assert E.shape == (n_freq, n_frames)
    assert E.dtype == np.complex64
    assert np.all(np.isfinite(E))

    # Test with X = 0: output must equal D
    filter_nlms.reset()
    X_zero = np.zeros_like(D)
    E_zero = filter_nlms.process(X_zero, D)
    np.testing.assert_allclose(E_zero, D, atol=1e-6)


def test_lms_filter_process_direct(rng: np.random.Generator) -> None:
    """Test LMSFilter.process directly on random complex inputs."""
    n_freq = N_FREQ
    n_frames = 20

    D = (rng.normal(0, 1.0, (n_freq, n_frames)) + 1j * rng.normal(0, 1.0, (n_freq, n_frames))).astype(np.complex64)
    X_zero = np.zeros_like(D)

    filter_lms = LMSFilter(n_freq=n_freq, taps=NLMS_TAPS)
    E_zero = filter_lms.process(X_zero, D)
    np.testing.assert_allclose(E_zero, D, atol=1e-6)


def test_nlms_convergence_synthetic_noise(rng: np.random.Generator) -> None:
    """Test NLMS adaptive noise cancellation on synthetic tone + correlated noise.

    Primary = tone + (delay + 8-tap FIR) * noise.
    Reference = noise.
    Assert after convergence (second half of audio), residual power is clearly
    below initial noisy power, and output is float32 of matching length.
    """
    duration_s = 4.0
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Clean tone
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Synthetic white noise reference
    ref_noise = rng.normal(0, 0.4, n_samples).astype(np.float32)

    # 8-tap FIR filter + 16 sample delay simulating acoustic path to primary mic
    fir = np.array([0.05, 0.1, 0.2, 0.35, 0.25, 0.15, 0.08, 0.04], dtype=np.float32)
    filtered_noise = np.convolve(ref_noise, fir, mode="full")[:n_samples]
    delay = 16
    delayed_noise = np.pad(filtered_noise, (delay, 0))[:n_samples]

    noisy = tone + delayed_noise

    enhanced = nlms_enhance(noisy, reference=ref_noise, mu=0.1)

    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))

    # Evaluate residual in second half (after adaptive filter has converged)
    second_half = slice(n_samples // 2, n_samples)
    noisy_error_power = np.mean((noisy[second_half] - tone[second_half]) ** 2)
    enhanced_error_power = np.mean((enhanced[second_half] - tone[second_half]) ** 2)

    # Significant attenuation of correlated noise
    assert enhanced_error_power < 0.6 * noisy_error_power


def test_reference_none_returns_noisy_copy() -> None:
    """When reference is None, enhance returns an exact copy of noisy."""
    noisy = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    out_nlms = nlms_enhance(noisy, reference=None)
    out_lms = lms_enhance(noisy, reference=None)

    np.testing.assert_array_equal(out_nlms, noisy)
    np.testing.assert_array_equal(out_lms, noisy)


def test_mismatched_reference_length_raises() -> None:
    """Mismatch in reference length should raise ValueError."""
    noisy = np.zeros(1000, dtype=np.float32)
    ref = np.zeros(800, dtype=np.float32)

    with pytest.raises(ValueError, match="Reference length"):
        nlms_enhance(noisy, reference=ref)

    with pytest.raises(ValueError, match="Reference length"):
        lms_enhance(noisy, reference=ref)
