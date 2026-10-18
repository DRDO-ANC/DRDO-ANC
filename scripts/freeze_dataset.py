"""Dataset freeze script for DRDO-ANC.

Validates the manifest, calculates the SHA-256 checksum, counts rows per split,
appends the official record to docs/datasets.md, and outputs instructions for tagging.
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path

from ai_engine.data.build_manifest import validate_manifest
from common.constants import SPLITS


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def freeze_dataset(
    manifest_path: Path | str,
    data_root: Path | str,
    docs_path: Path | str = "docs/datasets.md",
    tag_name: str = "dataset-v1",
) -> None:
    """Validate manifest, compute SHA-256, and append freeze record to documentation."""
    m_path = Path(manifest_path)
    d_root = Path(data_root)
    doc_p = Path(docs_path)

    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found at {m_path}")

    # 1. Load and validate manifest
    print(f"Validating manifest at {m_path} against {d_root}...")
    with open(m_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    validate_manifest(rows, d_root)
    print("Manifest validation passed successfully!")

    # 2. Count rows per split
    split_counts = {s: 0 for s in SPLITS}
    for r in rows:
        split_counts[r["split"]] += 1

    total_rows = len(rows)
    print(f"Total manifest samples: {total_rows}")
    for s, count in split_counts.items():
        print(f"  - {s}: {count} rows")

    # 3. Compute SHA-256 hash
    sha_hash = compute_sha256(m_path)
    freeze_date = datetime.now().strftime("%Y-%m-%d")
    print(f"\nManifest SHA-256: {sha_hash}")

    # 4. Append record to docs/datasets.md
    if doc_p.is_file():
        freeze_record = f"""
### Freeze Record: `{tag_name}` ({freeze_date})

- **Git Tag**: `{tag_name}`
- **Manifest File**: `{m_path.as_posix()}`
- **SHA-256 Hash**: `{sha_hash}`
- **Total Samples**: {total_rows}
  - `train`: {split_counts.get("train", 0)}
  - `val`: {split_counts.get("val", 0)}
  - `test`: {split_counts.get("test", 0)}
  - `test_unseen_noise`: {split_counts.get("test_unseen_noise", 0)}
- **Status**: FROZEN (immutable).
"""
        with open(doc_p, "a", encoding="utf-8") as f:
            f.write(freeze_record)
        print(f"Freeze record appended to {doc_p}")

    # 5. Print git instructions
    print("\n" + "=" * 60)
    print("DATASET FREEZE READY")
    print("=" * 60)
    print("Please run the git tag commands manually:")
    print(f"  git add {m_path.as_posix()} {doc_p.as_posix()}")
    print(f'  git commit -m "chore(data): freeze dataset manifest {tag_name}"')
    print(f'  git tag -a {tag_name} -m "Freeze dataset manifest {tag_name} ({sha_hash[:8]})"')
    print(f"  git push origin {tag_name}")
    print("=" * 60)


def main() -> None:
    """CLI entrypoint for dataset freezing."""
    parser = argparse.ArgumentParser(description="Freeze dataset manifest and record SHA-256.")
    default_data_root = os.environ.get("DATA_ROOT", "./data")
    parser.add_argument("--manifest", type=str, default="./data/manifest.csv")
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--docs", type=str, default="docs/datasets.md")
    parser.add_argument("--tag", type=str, default="dataset-v1")
    args = parser.parse_args()

    freeze_dataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        docs_path=args.docs,
        tag_name=args.tag,
    )


if __name__ == "__main__":
    main()
