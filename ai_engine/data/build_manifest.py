"""Manifest generation CLI and validation for DRDO-ANC synthetic datasets.

Conforms to DRDO-ANC AI Coding Contract v1 (Section D & Section E).
Outputs manifest CSV with columns:
id, speaker_id, clean_path, noise_path, reference_path, noise_type, snr_db, split, mixed_path
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ai_engine.data.mixer import mix_at_snr
from ai_engine.data.reference_sim import simulate_reference
from common.constants import (
    CLIP_SAMPLES,
    MANIFEST_COLUMNS,
    NOISE_TYPES,
    PEAK_NORM,
    SAMPLE_RATE,
    SEED,
    SNR_LEVELS_DB,
    SPLITS,
)

logger = logging.getLogger(__name__)

# Default held-out noise classes for test_unseen_noise
DEFAULT_HELD_OUT_NOISES = ("gunshot", "explosion")


def validate_manifest(
    rows: list[dict[str, Any]],
    data_root: Path | str,
    eps: float = 1e-10,
) -> None:
    """Validate that a manifest conforms strictly to DRDO-ANC contract rules.

    Fails loudly (raises ValueError) on:
      1. Missing audio files.
      2. Audio length != CLIP_SAMPLES (64000).
      3. Measured SNR of (mixed - clean) vs clean off by > 0.01 dB.
      4. Speaker overlap across splits.
      5. Unseen-noise leakage (held-out noises appearing in train/val/test,
         or seen noises appearing in test_unseen_noise).
      6. Bad noise_type or split values.

    Args:
        rows: List of row dictionaries representing the manifest.
        data_root: Root directory against which relative paths are resolved.
        eps: Epsilon for numerical stability in SNR calculation.

    Raises:
        ValueError: If any contract condition is violated.
    """
    root = Path(data_root)
    speakers_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    unseen_noises: set[str] = set()

    for idx, r in enumerate(rows):
        # 6. Validate split and noise_type
        split = r["split"]
        if split not in SPLITS:
            raise ValueError(f"Row {idx}: invalid split '{split}'. Must be one of {SPLITS}")

        noise_type = r["noise_type"]
        if noise_type not in NOISE_TYPES:
            raise ValueError(
                f"Row {idx}: invalid noise_type '{noise_type}'. Must be one of {NOISE_TYPES}"
            )

        speaker_id = str(r["speaker_id"])
        speakers_by_split[split].add(speaker_id)

        if split == "test_unseen_noise":
            unseen_noises.add(noise_type)

        # 1. Validate file existence
        for key in ["clean_path", "noise_path", "reference_path", "mixed_path"]:
            rel_path = r[key]
            full_path = root / rel_path
            if not full_path.is_file():
                raise ValueError(f"Row {idx}: missing audio file at {full_path}")

        # 2. Validate lengths, sample rate, and peak amplitudes (allowing 16-bit PCM quantization step)
        clean_audio, sr_c = sf.read(root / r["clean_path"], dtype="float32")
        noise_audio, sr_n = sf.read(root / r["noise_path"], dtype="float32")
        mixed_audio, sr_m = sf.read(root / r["mixed_path"], dtype="float32")
        ref_audio, sr_r = sf.read(root / r["reference_path"], dtype="float32")

        if sr_c != SAMPLE_RATE or sr_n != SAMPLE_RATE or sr_m != SAMPLE_RATE or sr_r != SAMPLE_RATE:
            raise ValueError(f"Row {idx}: sample rate mismatch with SAMPLE_RATE {SAMPLE_RATE}")

        for sig_name, sig in [("clean", clean_audio), ("noise", noise_audio), ("mixed", mixed_audio), ("reference", ref_audio)]:
            if len(sig) != CLIP_SAMPLES:
                raise ValueError(
                    f"Row {idx}: {sig_name} length {len(sig)} != CLIP_SAMPLES {CLIP_SAMPLES}"
                )
            # 16-bit quantization tolerance is 1.0 / 32768.0 ~ 3.05e-5
            peak_val = float(np.max(np.abs(sig)))
            if peak_val > PEAK_NORM + 1e-4:
                raise ValueError(
                    f"Row {idx}: {sig_name} peak {peak_val:.6f} exceeds PEAK_NORM {PEAK_NORM}"
                )

        # 3. Validate SNR accuracy within 0.01 dB
        target_snr = float(r["snr_db"])
        c_64 = clean_audio.astype(np.float64)
        m_64 = mixed_audio.astype(np.float64)
        diff_64 = m_64 - c_64

        clean_pow = np.mean(c_64**2)
        diff_pow = np.mean(diff_64**2)

        if clean_pow > eps and diff_pow > eps:
            meas_snr = 10.0 * np.log10(clean_pow / diff_pow)
            if abs(meas_snr - target_snr) > 0.01:
                raise ValueError(
                    f"Row {idx}: measured SNR {meas_snr:.4f} dB differs by > 0.01 dB "
                    f"from target {target_snr:.4f} dB"
                )

    # 4. Validate speaker-disjoint splits:
    # train vs val, and (train | val) vs (test | test_unseen_noise)
    train_speakers = speakers_by_split["train"]
    val_speakers = speakers_by_split["val"]
    test_speakers = speakers_by_split["test"]
    unseen_speakers = speakers_by_split["test_unseen_noise"]

    if train_speakers & val_speakers:
        raise ValueError(
            f"Speaker overlap between train and val: {train_speakers & val_speakers}"
        )
    if train_speakers & test_speakers:
        raise ValueError(
            f"Speaker overlap between train and test: {train_speakers & test_speakers}"
        )
    if train_speakers & unseen_speakers:
        raise ValueError(
            f"Speaker overlap between train and test_unseen_noise: {train_speakers & unseen_speakers}"
        )
    if val_speakers & test_speakers:
        raise ValueError(
            f"Speaker overlap between val and test: {val_speakers & test_speakers}"
        )
    if val_speakers & unseen_speakers:
        raise ValueError(
            f"Speaker overlap between val and test_unseen_noise: {val_speakers & unseen_speakers}"
        )

    # 5. Validate unseen-noise leakage:
    # Classes in test_unseen_noise must NOT appear in train, val, or test.
    for split_name in ["train", "val", "test"]:
        split_noises = {r["noise_type"] for r in rows if r["split"] == split_name}
        leakage = split_noises & unseen_noises
        if leakage:
            raise ValueError(
                f"Unseen-noise leakage: noise classes {leakage} from test_unseen_noise "
                f"found in '{split_name}' split"
            )


def build_manifest(
    data_root: Path | str,
    output_csv: Path | str,
    *,
    held_out_noises: tuple[str, ...] = DEFAULT_HELD_OUT_NOISES,
    small: bool = False,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Build dataset manifest and write synthetic audio mixtures.

    Args:
        data_root: Root dataset folder.
        output_csv: Destination manifest CSV file path.
        held_out_noises: Noise types reserved solely for test_unseen_noise.
        small: If True, generates a small set (e.g. 200 items).
        seed: Random seed for reproducibility.

    Returns:
        list[dict[str, Any]]: Generated manifest rows.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Set up directory layout
    for subdir in ["clean", "noise", "reference", "mixed"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    num_samples = 200 if small else 2000

    # Assign speakers: LibriSpeech structure
    train_speakers = [f"spk_tr_{i:03d}" for i in range(1, 11)]
    val_speakers = [f"spk_val_{i:03d}" for i in range(1, 4)]
    test_speakers = [f"spk_te_{i:03d}" for i in range(1, 4)]
    unseen_speakers = [f"spk_uns_{i:03d}" for i in range(1, 4)]

    available_seen_noises = [n for n in NOISE_TYPES if n not in held_out_noises]

    for i in range(num_samples):
        sample_id = f"anc_{i:06d}"

        # Assign split
        split_roll = rng.random()
        if split_roll < 0.60:
            split = "train"
            speaker = rng.choice(train_speakers)
            noise_type = rng.choice(available_seen_noises)
        elif split_roll < 0.80:
            split = "val"
            speaker = rng.choice(val_speakers)
            noise_type = rng.choice(available_seen_noises)
        elif split_roll < 0.90:
            split = "test"
            speaker = rng.choice(test_speakers)
            noise_type = rng.choice(available_seen_noises)
        else:
            split = "test_unseen_noise"
            speaker = rng.choice(unseen_speakers)
            noise_type = rng.choice(list(held_out_noises))

        target_snr = float(rng.choice(SNR_LEVELS_DB))

        # Synthetic source waveforms
        t = np.linspace(0, CLIP_SAMPLES / SAMPLE_RATE, CLIP_SAMPLES, endpoint=False)
        clean_raw = (0.4 * np.sin(2 * np.pi * rng.uniform(200, 800) * t)).astype(np.float32)
        noise_raw = rng.normal(0, 0.3, CLIP_SAMPLES).astype(np.float32)

        # Mix at exact target SNR
        mixed, clean_proc, noise_proc = mix_at_snr(clean_raw, noise_raw, target_snr, rng)

        # Simulate reference on the SAME cropped noise
        ref_proc = simulate_reference(noise_proc, rng)

        # File paths
        clean_rel = f"clean/{sample_id}_clean.wav"
        noise_rel = f"noise/{sample_id}_noise.wav"
        ref_rel = f"reference/{sample_id}_ref.wav"
        mixed_rel = f"mixed/{sample_id}_mixed.wav"

        # Write 16-bit PCM WAVs
        sf.write(root / clean_rel, clean_proc, SAMPLE_RATE, subtype="PCM_16")
        sf.write(root / noise_rel, noise_proc, SAMPLE_RATE, subtype="PCM_16")
        sf.write(root / ref_rel, ref_proc, SAMPLE_RATE, subtype="PCM_16")
        sf.write(root / mixed_rel, mixed, SAMPLE_RATE, subtype="PCM_16")

        row = {
            "id": sample_id,
            "speaker_id": speaker,
            "clean_path": clean_rel,
            "noise_path": noise_rel,
            "reference_path": ref_rel,
            "noise_type": noise_type,
            "snr_db": target_snr,
            "split": split,
            "mixed_path": mixed_rel,
        }
        rows.append(row)

    # Write manifest CSV with columns in exact order
    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Manifest with %d rows written to %s", len(rows), out_p)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest generation."""
    parser = argparse.ArgumentParser(description="Build and validate DRDO-ANC dataset manifest.")
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="Root directory for dataset files (default: DATA_ROOT env or ./data).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./data/manifest.csv",
        help="Path for output manifest CSV (default: ./data/manifest.csv).",
    )
    parser.add_argument(
        "--held-out-noises",
        nargs="+",
        default=list(DEFAULT_HELD_OUT_NOISES),
        help="Noise classes held out solely for test_unseen_noise (default: gunshot explosion).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate small manifest (200 rows) for smoke delivery.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random generator seed (default: {SEED}).",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Path to existing manifest CSV to validate without generating.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    data_root = Path(args.data_root)

    if args.validate_only:
        csv_path = Path(args.validate_only)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        validate_manifest(rows, data_root)
        print(f"Validation succeeded for {len(rows)} manifest rows at {csv_path}")
        return

    rows = build_manifest(
        data_root=data_root,
        output_csv=args.output_csv,
        held_out_noises=tuple(args.held_out_noises),
        small=args.small,
        seed=args.seed,
    )
    validate_manifest(rows, data_root)
    print(f"Manifest generation and validation successful: {len(rows)} samples created.")


if __name__ == "__main__":
    main()
