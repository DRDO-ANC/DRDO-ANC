"""Grid search hyperparameter tuning for classical speech enhancement baselines.

Evaluates candidates exclusively on the VAL split of the dataset manifest.
Scores models using ai_engine.evaluation.metrics.si_snr.
Saves best configuration parameters to experiments/configs/<name>.yaml.
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from ai_engine.evaluation.metrics import si_snr
from common.constants import SEED
from signal_processing.lms import enhance as lms_enhance
from signal_processing.nlms import enhance as nlms_enhance
from signal_processing.spectral import enhance as spectral_enhance
from signal_processing.wiener import enhance as wiener_enhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_val_split(data_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Load and filter manifest rows strictly for split == 'val'."""
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    val_rows = [r for r in all_rows if r["split"] == "val"]
    if not val_rows:
        raise ValueError(f"No samples found with split=='val' in {manifest_path}")

    # Preload audio into memory to speed up multi-iteration grid search
    data: list[dict[str, Any]] = []
    for r in val_rows:
        clean, _ = sf.read(data_root / r["clean_path"], dtype="float32")
        mixed, _ = sf.read(data_root / r["mixed_path"], dtype="float32")
        ref, _ = sf.read(data_root / r["reference_path"], dtype="float32")
        data.append({
            "id": r["id"],
            "clean": clean,
            "mixed": mixed,
            "ref": ref,
        })
    logger.info("Loaded %d validation items from %s", len(data), manifest_path)
    return data


def tune_spectral(val_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Grid search for spectral subtraction over (alpha, beta)."""
    alpha_grid = [1.0, 1.5, 2.0, 2.5]
    beta_grid = [0.01, 0.02, 0.05]

    best_score = -float("inf")
    best_params: dict[str, Any] = {}

    for alpha in alpha_grid:
        for beta in beta_grid:
            scores = []
            for item in val_data:
                enhanced = spectral_enhance(item["mixed"], alpha=alpha, beta=beta)
                score = si_snr(item["clean"], enhanced)
                scores.append(score)
            mean_score = float(np.mean(scores))
            logger.info("Spectral alpha=%.2f beta=%.3f -> SI-SNR: %.4f dB", alpha, beta, mean_score)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"alpha": alpha, "beta": beta}

    logger.info("Best Spectral: %s (SI-SNR: %.4f dB)", best_params, best_score)
    return best_params


def tune_wiener(val_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Grid search for Wiener filter over smoothing parameter a."""
    a_grid = [0.90, 0.95, 0.98, 0.99]

    best_score = -float("inf")
    best_params: dict[str, Any] = {}

    for a in a_grid:
        scores = []
        for item in val_data:
            enhanced = wiener_enhance(item["mixed"], a=a)
            score = si_snr(item["clean"], enhanced)
            scores.append(score)
        mean_score = float(np.mean(scores))
        logger.info("Wiener a=%.2f -> SI-SNR: %.4f dB", a, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"a": a}

    logger.info("Best Wiener: %s (SI-SNR: %.4f dB)", best_params, best_score)
    return best_params


def tune_nlms(val_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Grid search for NLMS filter over (taps, mu)."""
    taps_grid = [2, 4, 8]
    mu_grid = [0.1, 0.25, 0.5, 0.75]

    best_score = -float("inf")
    best_params: dict[str, Any] = {}

    for taps in taps_grid:
        for mu in mu_grid:
            scores = []
            for item in val_data:
                enhanced = nlms_enhance(item["mixed"], reference=item["ref"], taps=taps, mu=mu)
                score = si_snr(item["clean"], enhanced)
                scores.append(score)
            mean_score = float(np.mean(scores))
            logger.info("NLMS taps=%d mu=%.2f -> SI-SNR: %.4f dB", taps, mu, mean_score)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"taps": taps, "mu": mu}

    logger.info("Best NLMS: %s (SI-SNR: %.4f dB)", best_params, best_score)
    return best_params


def tune_lms(val_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Grid search for LMS filter over mu."""
    mu_grid = [5e-5, 1e-4, 2e-4]

    best_score = -float("inf")
    best_params: dict[str, Any] = {}

    for mu in mu_grid:
        scores = []
        for item in val_data:
            enhanced = lms_enhance(item["mixed"], reference=item["ref"], mu=mu)
            score = si_snr(item["clean"], enhanced)
            scores.append(score)
        mean_score = float(np.mean(scores))
        logger.info("LMS mu=%.1e -> SI-SNR: %.4f dB", mu, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"mu": mu}

    logger.info("Best LMS: %s (SI-SNR: %.4f dB)", best_params, best_score)
    return best_params


def save_config(name: str, method: str, params: dict[str, Any], output_dir: Path) -> Path:
    """Save configuration YAML to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{name}.yaml"
    config = {
        "name": name,
        "method": method,
        "seed": SEED,
        "dataset_version": "v1.0",
        "params": params,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Configuration saved to %s", out_file)
    return out_file


def main() -> None:
    """Main execution function for tuning classical baselines."""
    parser = argparse.ArgumentParser(description="Tune classical DSP baselines on val split.")
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./experiments/configs")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["spectral", "wiener", "nlms", "lms"],
        default=["spectral", "wiener", "nlms", "lms"],
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else data_root / "manifest.csv"
    output_dir = Path(args.output_dir)

    val_data = load_val_split(data_root, manifest_path)

    if "spectral" in args.methods:
        best_spectral = tune_spectral(val_data)
        save_config("tune_spectral", "spectral", best_spectral, output_dir)

    if "wiener" in args.methods:
        best_wiener = tune_wiener(val_data)
        save_config("tune_wiener", "wiener", best_wiener, output_dir)

    if "nlms" in args.methods:
        best_nlms = tune_nlms(val_data)
        save_config("tune_nlms", "nlms", best_nlms, output_dir)

    if "lms" in args.methods:
        best_lms = tune_lms(val_data)
        save_config("tune_lms", "lms", best_lms, output_dir)


if __name__ == "__main__":
    main()
