"""Tests for reference microphone simulation and manifest builder/validator.

Uses temporary directories and synthetic arrays in memory; no WAV files are committed.
Imports constants directly from common.constants.
"""

from pathlib import Path

import numpy as np
import pytest

from ai_engine.data.build_manifest import build_manifest, validate_manifest
from ai_engine.data.reference_sim import (
    DELAY_MAX_SAMPLES,
    DELAY_MIN_SAMPLES,
    simulate_reference,
)
from common.constants import (
    CLIP_SAMPLES,
    MANIFEST_COLUMNS,
    PEAK_NORM,
    SAMPLE_RATE,
    SEED,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture providing reproducible PRNG seeded with SEED."""
    return np.random.default_rng(SEED)


def test_simulate_reference_properties(rng: np.random.Generator) -> None:
    """Verify simulate_reference returns float32, same length, finite values."""
    noise = rng.normal(0, 0.5, CLIP_SAMPLES).astype(np.float32)
    ref = simulate_reference(noise, rng)

    assert isinstance(ref, np.ndarray)
    assert ref.dtype == np.float32
    assert ref.shape == (CLIP_SAMPLES,)
    assert np.all(np.isfinite(ref))


def test_no_clean_speech_leakage_in_reference(rng: np.random.Generator) -> None:
    """Verify that clean speech does not leak into the simulated reference.

    The correlation between reference and clean speech must be statistically near zero.
    """
    t = np.linspace(0, CLIP_SAMPLES / SAMPLE_RATE, CLIP_SAMPLES, endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
    noise = rng.normal(0, 0.4, CLIP_SAMPLES).astype(np.float32)

    ref = simulate_reference(noise, rng)

    # Normalized cross-correlation at lag 0
    clean_norm = clean / np.linalg.norm(clean)
    ref_norm = ref / np.linalg.norm(ref)
    corr = np.abs(np.dot(clean_norm, ref_norm))

    assert corr < 0.05  # Negligible correlation with clean speech


def test_reference_delay_bounds(rng: np.random.Generator) -> None:
    """Verify module delay constants conform to the contract."""
    assert DELAY_MIN_SAMPLES == 0
    assert DELAY_MAX_SAMPLES == 80  # 5 ms at 16000 Hz, far below 32 ms STFT window


def test_build_and_validate_manifest_small(tmp_path: Path) -> None:
    """Test generating a small manifest and running full validation."""
    data_root = tmp_path / "data"
    manifest_csv = tmp_path / "manifest.csv"

    rows = build_manifest(
        data_root=data_root,
        output_csv=manifest_csv,
        small=True,
        seed=SEED,
    )

    # 1. Check columns and order
    assert list(rows[0].keys()) == list(MANIFEST_COLUMNS)

    # 2. Check full validation passes
    validate_manifest(rows, data_root)


def test_validate_catches_speaker_overlap(tmp_path: Path) -> None:
    """Test validate_manifest catches an injected speaker overlap."""
    data_root = tmp_path / "data"
    manifest_csv = tmp_path / "manifest.csv"

    rows = build_manifest(
        data_root=data_root,
        output_csv=manifest_csv,
        small=True,
        seed=SEED,
    )

    # Inject speaker overlap between train and val
    train_row = next(r for r in rows if r["split"] == "train")
    val_row = next(r for r in rows if r["split"] == "val")
    val_row["speaker_id"] = train_row["speaker_id"]

    with pytest.raises(ValueError, match="Speaker overlap"):
        validate_manifest(rows, data_root)


def test_validate_catches_unseen_noise_leakage(tmp_path: Path) -> None:
    """Test validate_manifest catches unseen noise appearing in train split."""
    data_root = tmp_path / "data"
    manifest_csv = tmp_path / "manifest.csv"

    rows = build_manifest(
        data_root=data_root,
        output_csv=manifest_csv,
        small=True,
        seed=SEED,
    )

    # Inject unseen noise into a train row
    train_row = next(r for r in rows if r["split"] == "train")
    unseen_row = next(r for r in rows if r["split"] == "test_unseen_noise")
    train_row["noise_type"] = unseen_row["noise_type"]

    with pytest.raises(ValueError, match="Unseen-noise leakage"):
        validate_manifest(rows, data_root)


def test_validate_catches_snr_discrepancy(tmp_path: Path) -> None:
    """Test validate_manifest catches an SNR discrepancy > 0.01 dB."""
    data_root = tmp_path / "data"
    manifest_csv = tmp_path / "manifest.csv"

    rows = build_manifest(
        data_root=data_root,
        output_csv=manifest_csv,
        small=True,
        seed=SEED,
    )

    # Tamper with snr_db label
    rows[0]["snr_db"] = float(rows[0]["snr_db"]) + 2.0

    with pytest.raises(ValueError, match="measured SNR .* differs by > 0.01 dB"):
        validate_manifest(rows, data_root)


def test_reference_peak_normalisation_regression(rng: np.random.Generator) -> None:
    """Regression test: reference signal whose unscaled peak exceeds PEAK_NORM.

    Verifies the simulated reference signal peak is guarded to <= PEAK_NORM (0.99).
    """
    # Noise with peak close to 0.99 that can peak > 1.0 after FIR + sensor noise
    noise = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    noise[:100] = 0.98 * np.sign(rng.normal(0, 1.0, 100))

    ref = simulate_reference(noise, rng)

    assert np.max(np.abs(ref)) <= PEAK_NORM + 1e-6
    assert np.all(np.isfinite(ref))

