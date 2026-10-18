"""Runner script for baseline evaluation across all classical methods.

Runs ai_engine.evaluation.evaluate on a given manifest (default: small manifest)
for all classical baseline configurations:
- noisy
- spectral
- wiener
- lms
- nlms

Outputs are written to experiments/results/<name>.json and adjacent breakdown CSVs.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_engine.evaluation.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_NAMES = [
    "noisy_baseline_v1",
    "spectral_baseline_v1",
    "wiener_baseline_v1",
    "lms_baseline_v1",
    "nlms_baseline_v1",
]


def run_all_baselines(manifest: str, data_root: str | None = None, split: str = "test") -> list[Path]:
    """Execute evaluation for all baseline configs and return generated JSON paths."""
    manifest_p = Path(manifest).resolve()
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    results_dir = REPO_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = REPO_ROOT / "experiments" / "configs"

    output_jsons: list[Path] = []

    for name in CONFIG_NAMES:
        config_path = configs_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config YAML not found: {config_path}")

        method = name.replace("_baseline_v1", "")
        out_json = results_dir / f"{name}.json"

        logger.info("Running baseline for %s (method=%s)...", name, method)
        run_evaluation(
            method=method,
            manifest_csv=manifest_p,
            split=split,
            out_json=out_json,
            config_yaml=config_path,
            data_root=data_root,
            name=name,
        )
        output_jsons.append(out_json)

    return output_jsons


def main() -> None:
    """CLI entrypoint for running all baseline evaluations."""
    parser = argparse.ArgumentParser(description="Run classical baseline evaluations.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.csv",
        help="Path to manifest CSV (defaults to data/manifest.csv).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("DATA_ROOT", None),
        help="Root folder for audio files (defaults to DATA_ROOT env var).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Manifest split to evaluate on (default: test).",
    )
    args = parser.parse_args()

    results = run_all_baselines(manifest=args.manifest, data_root=args.data_root, split=args.split)
    logger.info("Successfully generated %d baseline results:", len(results))
    for res in results:
        logger.info("  %s", res)


if __name__ == "__main__":
    main()
