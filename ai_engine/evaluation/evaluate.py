"""Evaluation CLI for DRDO-ANC speech enhancement algorithms.

Conforms to DRDO-ANC AI Coding Contract v1 (Section D).
Computes SNR, SI-SNR, SSNR, STOI, and PESQ metrics, offline latency estimates, and RTF.
Outputs results JSON with keys strictly matching common.constants.RESULT_KEYS,
and writes per-noise_type and per-snr_db breakdowns to an adjacent CSV.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from ai_engine.evaluation.metrics import (
    pesq_score,
    si_snr,
    snr,
    ssnr,
    stoi_score,
)
from ai_engine.evaluation.metrics import (
    rtf as calc_rtf,
)
from common.constants import (
    FRAME_SAMPLES,
    METHODS,
    RESULT_KEYS,
    SAMPLE_RATE,
    SPLITS,
)
from common.interface import check_enhanced

logger = logging.getLogger(__name__)


def _load_audio(path: Path | str) -> np.ndarray:
    """Load audio as float32 1D array using soundfile with fallback for audio_io."""
    try:
        from ai_engine.models.audio_io import load_wav  # type: ignore

        wave, _ = load_wav(path)
        return np.asarray(wave, dtype=np.float32)
    except ImportError:
        data, _sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)


def _get_git_commit() -> str:
    """Get current git commit hash (HEAD)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_enhancer_fn(method: str) -> Callable[..., np.ndarray]:
    """Dynamically resolve enhancer function for the given method."""
    if method == "noisy":
        # Identity enhancer: leaves noisy signal untouched
        return lambda noisy, reference=None, sr=SAMPLE_RATE, **kwargs: noisy.copy()
    if method == "spectral":
        from signal_processing.spectral import enhance

        return enhance
    if method == "wiener":
        from signal_processing.wiener import enhance

        return enhance
    if method == "lms":
        from signal_processing.lms import enhance

        return enhance
    if method == "nlms":
        from signal_processing.nlms import enhance

        return enhance
    if method == "ai":
        try:
            from ai_engine.inference.enhance import enhance

            return enhance
        except ImportError as err:
            raise NotImplementedError(
                f"AI inference module is not yet implemented: {err}"
            ) from err
    if method == "hybrid":
        from signal_processing.hybrid import enhance

        return enhance

    raise ValueError(f"Unknown method '{method}'. Must be one of {METHODS}")


def run_evaluation(
    method: str,
    manifest_csv: Path | str,
    split: str,
    out_json: Path | str,
    *,
    config_yaml: Path | str | None = None,
    data_root: Path | str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Run evaluation benchmark over a manifest split and save results.

    Note on Latency:
        latency_ms is computed as the mean processing time per 16 ms frame equivalent:
            latency_ms = (processing_seconds / (audio_seconds / 0.016)) * 1000.0
        This is an OFFLINE processing speed estimate. Real-time streaming latency
        is formally benchmarked by P4's measure_latency module.

    Args:
        method: Method name from common.constants.METHODS.
        manifest_csv: Path to dataset manifest CSV.
        split: Dataset split to evaluate ('val', 'test', 'test_unseen_noise').
        out_json: Destination path for results JSON.
        config_yaml: Optional configuration YAML file containing 'params'.
        data_root: Directory against which manifest audio paths are resolved.
        name: Optional run name (defaults to output JSON stem).

    Returns:
        dict[str, Any]: Results dictionary matching RESULT_KEYS.
    """
    if method not in METHODS:
        raise ValueError(f"Method '{method}' must be one of {METHODS}")
    if split not in SPLITS:
        raise ValueError(f"Split '{split}' must be one of {SPLITS}")

    manifest_p = Path(manifest_csv)
    if data_root is None:
        data_root_p = Path(os.environ.get("DATA_ROOT", manifest_p.parent))
    else:
        data_root_p = Path(data_root)

    params: dict[str, Any] = {}
    if config_yaml:
        with open(config_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            params = cfg.get("params", {})

    with open(manifest_p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    target_rows = [r for r in all_rows if r["split"] == split]
    if not target_rows:
        raise ValueError(f"No samples found for split '{split}' in {manifest_p}")

    enhancer_fn = _get_enhancer_fn(method)

    per_sample_results: list[dict[str, Any]] = []
    pesq_failed_count = 0
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    logger.info("Evaluating %d samples on '%s' split with method '%s'...", len(target_rows), split, method)

    for row in target_rows:
        clean = _load_audio(data_root_p / row["clean_path"])
        noisy = _load_audio(data_root_p / row["mixed_path"])
        ref_path = row.get("reference_path")
        ref = _load_audio(data_root_p / ref_path) if ref_path else None

        t0 = time.perf_counter()
        enhanced = enhancer_fn(noisy, reference=ref, sr=SAMPLE_RATE, **params)
        proc_time = time.perf_counter() - t0

        check_enhanced(noisy, enhanced)

        audio_len_sec = len(noisy) / SAMPLE_RATE
        total_audio_sec += audio_len_sec
        total_proc_sec += proc_time

        # Compute metrics
        s_snr = snr(clean, enhanced)
        s_si_snr = si_snr(clean, enhanced)
        s_ssnr = ssnr(clean, enhanced)
        s_stoi = stoi_score(clean, enhanced)
        s_pesq = pesq_score(clean, enhanced)

        if np.isnan(s_pesq):
            pesq_failed_count += 1

        per_sample_results.append({
            "id": row["id"],
            "noise_type": row["noise_type"],
            "snr_db": float(row["snr_db"]),
            "snr": s_snr,
            "si_snr": s_si_snr,
            "ssnr": s_ssnr,
            "stoi": s_stoi,
            "pesq": s_pesq,
            "proc_time": proc_time,
            "audio_len": audio_len_sec,
        })

    if pesq_failed_count > 0:
        logger.warning(
            "PESQ computation failed for %d out of %d clips (logged as NaN and excluded from average).",
            pesq_failed_count,
            len(target_rows),
        )

    # Compute overall aggregates
    mean_snr = float(np.mean([r["snr"] for r in per_sample_results]))
    mean_si_snr = float(np.mean([r["si_snr"] for r in per_sample_results]))
    mean_ssnr = float(np.mean([r["ssnr"] for r in per_sample_results]))
    mean_stoi = float(np.mean([r["stoi"] for r in per_sample_results]))

    valid_pesq = [r["pesq"] for r in per_sample_results if not np.isnan(r["pesq"])]
    mean_pesq = float(np.mean(valid_pesq)) if valid_pesq else float("nan")

    overall_rtf = calc_rtf(total_proc_sec, total_audio_sec)
    # Estimate latency in ms per 16 ms frame (FRAME_SAMPLES = 256 samples = 16 ms)
    frame_duration_s = FRAME_SAMPLES / SAMPLE_RATE
    total_frames = total_audio_sec / frame_duration_s
    latency_ms = float((total_proc_sec / total_frames) * 1000.0) if total_frames > 0 else 0.0

    run_name = name or Path(out_json).stem

    # Form final JSON results strictly conforming to RESULT_KEYS
    results: dict[str, Any] = {
        "name": run_name,
        "method": method,
        "snr": mean_snr,
        "si_snr": mean_si_snr,
        "ssnr": mean_ssnr,
        "stoi": mean_stoi,
        "pesq": mean_pesq,
        "latency_ms": latency_ms,
        "rtf": overall_rtf,
        "commit": _get_git_commit(),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    # Verify keys match RESULT_KEYS exactly
    assert tuple(results.keys()) == RESULT_KEYS, f"Key mismatch with contract: {results.keys()}"

    # Write results JSON
    out_json_p = Path(out_json)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results written to %s", out_json_p)

    # Write breakdowns to separate CSV next to the JSON
    breakdown_csv_p = out_json_p.parent / f"{out_json_p.stem}_breakdown.csv"
    _write_breakdowns(per_sample_results, breakdown_csv_p)

    return results


def _write_breakdowns(samples: list[dict[str, Any]], out_path: Path) -> None:
    """Write per-noise_type and per-snr_db breakdown statistics to CSV."""
    rows: list[dict[str, Any]] = []

    # Group by noise_type
    noise_groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        noise_groups.setdefault(s["noise_type"], []).append(s)

    for n_type, group in sorted(noise_groups.items()):
        v_pesq = [x["pesq"] for x in group if not np.isnan(x["pesq"])]
        rows.append({
            "category": "noise_type",
            "group": n_type,
            "count": len(group),
            "snr": float(np.mean([x["snr"] for x in group])),
            "si_snr": float(np.mean([x["si_snr"] for x in group])),
            "ssnr": float(np.mean([x["ssnr"] for x in group])),
            "stoi": float(np.mean([x["stoi"] for x in group])),
            "pesq": float(np.mean(v_pesq)) if v_pesq else float("nan"),
        })

    # Group by snr_db
    snr_groups: dict[float, list[dict[str, Any]]] = {}
    for s in samples:
        snr_groups.setdefault(s["snr_db"], []).append(s)

    for snr_val, group in sorted(snr_groups.items()):
        v_pesq = [x["pesq"] for x in group if not np.isnan(x["pesq"])]
        rows.append({
            "category": "snr_db",
            "group": str(snr_val),
            "count": len(group),
            "snr": float(np.mean([x["snr"] for x in group])),
            "si_snr": float(np.mean([x["si_snr"] for x in group])),
            "ssnr": float(np.mean([x["ssnr"] for x in group])),
            "stoi": float(np.mean([x["stoi"] for x in group])),
            "pesq": float(np.mean(v_pesq)) if v_pesq else float("nan"),
        })

    fieldnames = ["category", "group", "count", "snr", "si_snr", "ssnr", "stoi", "pesq"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Breakdown report written to %s", out_path)


def main() -> None:
    """CLI entrypoint for speech enhancement evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement methods on manifest.")
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS))
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV.")
    parser.add_argument("--split", type=str, default="test", choices=list(SPLITS))
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON.")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML.")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder for audio files.")
    parser.add_argument("--name", type=str, default=None, help="Optional benchmark name.")
    args = parser.parse_args()

    run_evaluation(
        method=args.method,
        manifest_csv=args.manifest,
        split=args.split,
        out_json=args.out,
        config_yaml=args.config,
        data_root=args.data_root,
        name=args.name,
    )


if __name__ == "__main__":
    main()
