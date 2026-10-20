"""Integration test for all classical speech enhancement methods.

Verifies that enhance() from each classical method (spectral, wiener, lms, nlms):
- accepts noisy audio (and reference audio when applicable)
- returns float32 1D numpy array of the exact same length as input
- produces all finite numbers
- passes common.interface.check_enhanced
"""

import numpy as np
import pytest

from common.constants import CLIP_SAMPLES, SAMPLE_RATE, SEED
from common.interface import check_enhanced


@pytest.fixture
def synthetic_audio():
    """Create a synthetic test signal with pure tone + noise."""
    rng = np.random.default_rng(SEED)
    t = np.arange(CLIP_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    clean = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = (0.1 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)
    noisy = (clean + noise).astype(np.float32)

    # Reference signal correlated with noise for adaptive filtering
    ref = (0.9 * noise + 0.01 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)
    return clean, noisy, ref


def test_integration_spectral_subtraction(synthetic_audio):
    """Test spectral subtraction enhance integration."""
    try:
        from signal_processing.spectral import enhance
    except ImportError as e:
        pytest.skip(f"spectral module not available: {e}")

    _, noisy, _ = synthetic_audio
    enhanced = enhance(noisy, sr=SAMPLE_RATE, alpha=2.0, beta=0.02)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))
    # Must pass contract check function
    check_enhanced(noisy, enhanced)


def test_integration_wiener_filter(synthetic_audio):
    """Test Wiener filter enhance integration."""
    try:
        from signal_processing.wiener import enhance
    except ImportError as e:
        pytest.skip(f"wiener module not available: {e}")

    _, noisy, _ = synthetic_audio
    enhanced = enhance(noisy, sr=SAMPLE_RATE, a=0.98)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))
    check_enhanced(noisy, enhanced)


def test_integration_lms_filter(synthetic_audio):
    """Test STFT-domain LMS filter enhance integration."""
    try:
        from signal_processing.lms import enhance
    except ImportError as e:
        pytest.skip(f"lms module not available: {e}")

    _, noisy, ref = synthetic_audio
    enhanced = enhance(noisy, reference=ref, sr=SAMPLE_RATE, mu=1e-4)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))
    check_enhanced(noisy, enhanced)


def test_integration_nlms_filter(synthetic_audio):
    """Test STFT-domain NLMS filter enhance integration."""
    try:
        from signal_processing.nlms import enhance
    except ImportError as e:
        pytest.skip(f"nlms module not available: {e}")

    _, noisy, ref = synthetic_audio
    enhanced = enhance(noisy, reference=ref, sr=SAMPLE_RATE, taps=4, mu=0.5, eps=1e-6)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    assert np.all(np.isfinite(enhanced))
    check_enhanced(noisy, enhanced)


def test_integration_noisy_identity(synthetic_audio):
    """Test noisy (identity) method behaves as expected."""
    _, noisy, _ = synthetic_audio
    enhanced = noisy.astype(np.float32)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == np.float32
    assert enhanced.shape == noisy.shape
    check_enhanced(noisy, enhanced)
