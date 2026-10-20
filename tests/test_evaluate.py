"""Tests for evaluation CLI in ai_engine/evaluation/evaluate.py.

Uses temporary directories and synthetic audio (no real files or committed audio).
Imports constants directly from common.constants.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ai_engine.data.build_manifest import build_manifest
from ai_engine.evaluation.evaluate import run_evaluation
from common.constants import RESULT_KEYS, SEED


def test_evaluate_noisy_identity_method(tmp_path: Path) -> None:
    """Test evaluation on synthetic manifest with identity 'noisy' method.

    Asserts output SNR matches input SNR, JSON keys strictly match RESULT_KEYS,
    and the breakdown CSV is created.
    """
    data_root = tmp_path / "data"
    manifest_csv = tmp_path / "manifest.csv"
    out_json = tmp_path / "results" / "eval_noisy.json"

    # Generate small manifest with synthetic audio
    rows = build_manifest(
        data_root=data_root,
        output_csv=manifest_csv,
        small=True,
        seed=SEED,
    )
    test_rows = [r for r in rows if r["split"] == "test"]
    assert len(test_rows) > 0

    # Run evaluation
    results = run_evaluation(
        method="noisy",
        manifest_csv=manifest_csv,
        split="test",
        out_json=out_json,
        data_root=data_root,
        name="test_noisy_run",
    )

    # 1. Assert exact JSON keys and contract order
    assert tuple(results.keys()) == RESULT_KEYS

    # Verify JSON written to disk matches
    assert out_json.is_file()
    with open(out_json, encoding="utf-8") as f:
        disk_results = json.load(f)
    assert tuple(disk_results.keys()) == RESULT_KEYS

    # 2. For identity method 'noisy', output enhanced is noisy itself.
    # Therefore, measured SNR on the test split should approximately match target SNR.
    mean_target_snr = float(np.mean([float(r["snr_db"]) for r in test_rows]))
    assert results["snr"] == pytest.approx(mean_target_snr, abs=0.05)

    # 3. Verify breakdown CSV was created next to the JSON
    breakdown_csv = tmp_path / "results" / "eval_noisy_breakdown.csv"
    assert breakdown_csv.is_file()
    lines = breakdown_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 1  # Header + rows
    assert lines[0] == "category,group,count,snr,si_snr,ssnr,stoi,pesq"


def test_evaluate_invalid_method_or_split_raises(tmp_path: Path) -> None:
    """Invalid method or split should raise ValueError."""
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text("dummy", encoding="utf-8")
    out_json = tmp_path / "out.json"

    with pytest.raises(ValueError, match="Method 'invalid_method' must be one of"):
        run_evaluation(
            method="invalid_method",
            manifest_csv=manifest_csv,
            split="test",
            out_json=out_json,
        )

    with pytest.raises(ValueError, match="Split 'invalid_split' must be one of"):
        run_evaluation(
            method="noisy",
            manifest_csv=manifest_csv,
            split="invalid_split",
            out_json=out_json,
        )
