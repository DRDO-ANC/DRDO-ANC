"""Tests for audio mixer in ai_engine/data/mixer.py.

All tests use synthetic NumPy arrays (no WAV files or file I/O).
Imports constants directly from common.constants.
"""

import numpy as np
import pytest

from ai_engine.data.mixer import mix_at_snr
from common.constants import CLIP_SAMPLES, PEAK_NORM, SEED, SNR_LEVELS_DB


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG seeded with SEED."""
    return np.random.default_rng(SEED)


@pytest.fixture
def clean_synthetic(rng: np.random.Generator) -> np.ndarray:
    """Fixture providing synthetic clean speech-like signal."""
    t = np.linspace(0, 5.0, 5 * 16000, endpoint=False)  # 5 seconds (> CLIP_SAMPLES)
    signal = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
    return signal.astype(np.float32)


@pytest.fixture
def noise_synthetic(rng: np.random.Generator) -> np.ndarray:
    """Fixture providing synthetic background noise."""
    return rng.normal(0, 0.3, 6 * 16000).astype(np.float32)  # 6 seconds


@pytest.mark.parametrize("target_snr", SNR_LEVELS_DB)
def test_mix_at_snr_exact_db(
    clean_synthetic: np.ndarray,
    noise_synthetic: np.ndarray,
    target_snr: int,
    rng: np.random.Generator,
) -> None:
    """Verify measured SNR of (mixed - clean) vs clean is within 0.01 dB of target."""
    mixed, clean_out, noise_out = mix_at_snr(
        clean_synthetic, noise_synthetic, float(target_snr), rng
    )

    # Validate output length, shape, and dtype
    assert mixed.shape == (CLIP_SAMPLES,)
    assert clean_out.shape == (CLIP_SAMPLES,)
    assert noise_out.shape == (CLIP_SAMPLES,)
    assert mixed.dtype == np.float32
    assert clean_out.dtype == np.float32
    assert noise_out.dtype == np.float32

    # Verify linear combination: mixed == clean + noise
    np.testing.assert_allclose(mixed, clean_out + noise_out, atol=1e-5)

    # Compute actual achieved SNR in double precision
    c_pow = np.mean(clean_out.astype(np.float64) ** 2)
    diff = (mixed - clean_out).astype(np.float64)
    diff_pow = np.mean(diff**2)

    achieved_snr = 10.0 * np.log10(c_pow / diff_pow)
    assert achieved_snr == pytest.approx(float(target_snr), abs=0.01)


def test_mix_at_snr_peak_normalization(rng: np.random.Generator) -> None:
    """Peak-normalise: if peak of mixed > 0.99, scale so max peak <= 0.99."""
    # Create high-amplitude signals that definitely clip without normalization
    clean = np.ones(CLIP_SAMPLES, dtype=np.float32) * 0.9
    noise = np.ones(CLIP_SAMPLES, dtype=np.float32) * 0.9

    mixed, clean_out, noise_out = mix_at_snr(clean, noise, 0.0, rng)

    assert np.max(np.abs(mixed)) <= PEAK_NORM + 1e-6
    assert np.max(np.abs(clean_out)) <= PEAK_NORM + 1e-6
    assert np.max(np.abs(noise_out)) <= PEAK_NORM + 1e-6


def test_clean_padding_when_shorter(rng: np.random.Generator) -> None:
    """Verify clean signal is zero-padded if shorter than CLIP_SAMPLES."""
    short_clean = np.ones(1000, dtype=np.float32) * 0.5
    noise = rng.normal(0, 0.1, CLIP_SAMPLES).astype(np.float32)

    _mixed, clean_out, _noise_out = mix_at_snr(short_clean, noise, 5.0, rng)

    assert clean_out.shape == (CLIP_SAMPLES,)
    # Tail of clean_out should be zero
    np.testing.assert_allclose(clean_out[1000:], 0.0)


def test_noise_tiling_when_shorter(rng: np.random.Generator) -> None:
    """Verify noise signal is tiled cyclically if shorter than CLIP_SAMPLES."""
    clean = rng.normal(0, 0.1, CLIP_SAMPLES).astype(np.float32)
    short_noise = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)

    mixed, _clean_out, noise_out = mix_at_snr(clean, short_noise, 10.0, rng)

    assert noise_out.shape == (CLIP_SAMPLES,)
    assert mixed.shape == (CLIP_SAMPLES,)
    assert np.all(np.isfinite(noise_out))


def test_reproducibility_with_same_seed() -> None:
    """Verify same rng seed gives identical output."""
    rng1 = np.random.default_rng(SEED)
    clean1 = rng1.normal(0, 0.2, CLIP_SAMPLES * 2).astype(np.float32)
    noise1 = rng1.normal(0, 0.2, CLIP_SAMPLES * 2).astype(np.float32)

    rng2 = np.random.default_rng(SEED)
    clean2 = rng2.normal(0, 0.2, CLIP_SAMPLES * 2).astype(np.float32)
    noise2 = rng2.normal(0, 0.2, CLIP_SAMPLES * 2).astype(np.float32)

    out1_mix, out1_clean, out1_noise = mix_at_snr(clean1, noise1, 0.0, rng1)
    out2_mix, out2_clean, out2_noise = mix_at_snr(clean2, noise2, 0.0, rng2)

    np.testing.assert_array_equal(out1_mix, out2_mix)
    np.testing.assert_array_equal(out1_clean, out2_clean)
    np.testing.assert_array_equal(out1_noise, out2_noise)


def test_peak_norm_negative_snr_loud_noise(rng: np.random.Generator) -> None:
    """Regression test: loud noise at negative SNR where max|noise| > max|mixed|.

    Verifies all three signals (mixed, clean, noise) have peak <= PEAK_NORM (0.99)
    and the SNR error is < 0.01 dB.
    """
    # Create signals where noise amplitude exceeds mixed due to destructive cancellation
    t = np.linspace(0, 4.0, CLIP_SAMPLES, endpoint=False)
    clean = (0.2 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    noise = rng.normal(0, 0.5, CLIP_SAMPLES).astype(np.float32)

    snr_db = -5.0
    mixed, clean_out, noise_out = mix_at_snr(clean, noise, snr_db, rng)

    assert np.max(np.abs(mixed)) <= PEAK_NORM + 1e-6
    assert np.max(np.abs(clean_out)) <= PEAK_NORM + 1e-6
    assert np.max(np.abs(noise_out)) <= PEAK_NORM + 1e-6

    # Verify SNR precision
    clean_pow = np.mean(clean_out.astype(np.float64) ** 2)
    noise_pow = np.mean(noise_out.astype(np.float64) ** 2)
    actual_snr = 10.0 * np.log10(clean_pow / noise_pow)
    assert abs(actual_snr - snr_db) < 0.01

