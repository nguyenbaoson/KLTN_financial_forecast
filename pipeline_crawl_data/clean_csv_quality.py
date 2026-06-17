"""
Clean low-risk CSV quality issues.

For each CSV under the selected roots:
  - drop columns that are entirely null
  - drop exact duplicate rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import resolve_repo_path


DEFAULT_ROOTS = [
    "data/raw",
    "data/features",
    "data/raw/analytics",
    "data/raw/insights",
]


def clean_csv(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    original_rows = len(df)
    original_cols = len(df.columns)

    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()

    dropped_rows = original_rows - len(df)
    dropped_cols = original_cols - len(df.columns)
    if dropped_rows or dropped_cols:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return dropped_rows, dropped_cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop exact duplicate rows and all-null columns from CSV data files.")
    parser.add_argument("--roots", default=",".join(DEFAULT_ROOTS), help="Comma-separated roots to scan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [resolve_repo_path(item.strip()) for item in args.roots.split(",") if item.strip()]
    changed = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            dropped_rows, dropped_cols = clean_csv(path)
            if dropped_rows or dropped_cols:
                changed.append((str(path), dropped_rows, dropped_cols))

    for path, dropped_rows, dropped_cols in changed:
        print(f"{path}: dropped_rows={dropped_rows}, dropped_all_null_cols={dropped_cols}")
    print(f"changed_files={len(changed)}")


if __name__ == "__main__":
    main()

