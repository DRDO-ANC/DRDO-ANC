"""Tests for speech enhancement metrics in ai_engine/evaluation/metrics.py.

All tests use synthetic signals (sine waves, pseudo-random noise) with no WAV files.
Constants are imported directly from common.constants.
"""

from pathlib import Path

import numpy as np
import pytest

from ai_engine.evaluation.metrics import (
    pesq_score,
    rtf,
    si_snr,
    snr,
    ssnr,
    stoi_score,
)
from common.constants import SAMPLE_RATE, SEED


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG seeded with SEED."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_speech(rng: np.random.Generator) -> np.ndarray:
    """Fixture providing a synthetic speech-like signal (sum of harmonics + noise)."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * rng.normal(0, 0.05, len(t))
    )
    return signal.astype(np.float64)


def test_identical_signals_high_snr_and_si_snr(synthetic_speech: np.ndarray) -> None:
    """Identical signals should produce very high SNR and SI-SNR (> 80 dB)."""
    clean = synthetic_speech
    enhanced = synthetic_speech.copy()

    val_snr = snr(clean, enhanced)
    val_si_snr = si_snr(clean, enhanced)
    val_ssnr = ssnr(clean, enhanced)

    assert val_snr > 80.0
    assert val_si_snr > 80.0
    assert val_ssnr == pytest.approx(35.0, abs=1e-3)


def test_scaling_leaves_si_snr_unchanged_but_lowers_snr(
    synthetic_speech: np.ndarray,
) -> None:
    """Scaling estimate by 0.5 leaves SI-SNR unchanged but lowers SNR."""
    clean = synthetic_speech
    scaled_enhanced = 0.5 * clean

    val_si_snr = si_snr(clean, scaled_enhanced)
    val_snr = snr(clean, scaled_enhanced)

    # Scale invariant SNR should remain very high
    assert val_si_snr > 80.0
    # Standard SNR with clean - 0.5*clean = 0.5*clean:
    # 10*log10(sum(clean^2) / sum((0.5*clean)^2)) = 10*log10(4) = ~6.02 dB
    assert val_snr == pytest.approx(6.0206, abs=0.1)


def test_stoi_identical_signals(synthetic_speech: np.ndarray) -> None:
    """STOI of identical signals should be approximately 1.0."""
    clean = synthetic_speech
    enhanced = synthetic_speech.copy()

    val_stoi = stoi_score(clean, enhanced)
    assert val_stoi == pytest.approx(1.0, abs=1e-2)


def test_pesq_failure_on_short_clip_returns_nan() -> None:
    """PESQ failure on very short clip returns NaN and does not raise."""
    # A 100-sample clip (~6 ms) is far too short for wideband PESQ (needs ~1/4 s min)
    short_clean = np.zeros(100, dtype=np.float64)
    short_enhanced = np.zeros(100, dtype=np.float64)

    score = pesq_score(short_clean, short_enhanced)
    assert np.isnan(score)


def test_rtf_helper() -> None:
    """RTF calculation helper test."""
    val = rtf(0.5, 2.0)
    assert val == pytest.approx(0.25)


def test_compare_with_torch_si_snr(synthetic_speech: np.ndarray) -> None:
    """Compare NumPy si_snr with P1's PyTorch si_snr in ai_engine/training/losses.py."""
    losses_path = Path("ai_engine/training/losses.py")
    if not losses_path.exists():
        pytest.skip("ai_engine/training/losses.py does not exist yet (P1 module)")

    import torch
    from ai_engine.training.losses import si_snr as torch_si_snr

    clean_np = synthetic_speech
    rng = np.random.default_rng(SEED)
    noisy_np = clean_np + rng.normal(0, 0.1, len(clean_np))

    numpy_result = si_snr(clean_np, noisy_np)

    clean_t = torch.from_numpy(clean_np).unsqueeze(0)
    noisy_t = torch.from_numpy(noisy_np).unsqueeze(0)
    torch_result = torch_si_snr(clean_t, noisy_t).item()

    # Accommodate potential sign conventions (loss = -si_snr vs metric = +si_snr)
    if torch_result < 0 and numpy_result > 0:
        torch_result = -torch_result

    assert numpy_result == pytest.approx(torch_result, abs=1e-3)
