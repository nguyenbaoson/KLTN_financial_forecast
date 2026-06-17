"""
Audit the compact ML data tree under data/raw.

Read-only. Writes a report to outputs/data_quality/ml_tree_audit.csv.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import resolve_repo_path


REQUIRED_FUNDAMENTAL_FILES = [
    "income_statement.csv",
    "balance_sheet.csv",
    "cash_flow.csv",
    "ratio.csv",
]

PRICE_FIELDS = {
    "date": {"date", "time", "trading_date"},
    "close": {"close", "close_price"},
    "volume": {"volume", "match_volume"},
}


def clean_colname(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, str(exc)


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    category: str,
    path: Path,
    issue: str,
    suggestion: str,
    rows: int | None = None,
) -> None:
    issues.append({
        "severity": severity,
        "category": category,
        "path": str(path),
        "issue": issue,
        "suggestion": suggestion,
        "rows": rows,
    })


def audit_all_csv(data_root: Path, issues: list[dict[str, Any]]) -> None:
    for path in (data_root / "raw").rglob("*.csv"):
        df, error = read_csv(path)
        if error:
            add_issue(issues, "ERROR", "read_csv", path, error, "Re-crawl or restore this file.")
            continue
        if df is None or df.empty:
            add_issue(issues, "ERROR", "empty_file", path, "CSV has zero rows.", "Re-crawl or exclude this file.", 0)
            continue
        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            add_issue(issues, "WARN", "duplicates", path, f"{duplicate_rows} duplicated rows.", "Drop duplicates in processed layer.", len(df))
        null_cols = [col for col in df.columns if df[col].isna().all()]
        if null_cols:
            add_issue(
                issues,
                "WARN",
                "all_null_columns",
                path,
                f"All-null columns: {', '.join(null_cols[:20])}",
                "Drop all-null columns in processed layer.",
                len(df),
            )


def audit_fundamental(data_root: Path, issues: list[dict[str, Any]]) -> None:
    root = data_root / "raw" / "fundamental"
    if not root.exists():
        add_issue(issues, "ERROR", "missing_folder", root, "Missing fundamental folder.", "Run vnstock collector.")
        return

    for symbol_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for filename in REQUIRED_FUNDAMENTAL_FILES:
            path = symbol_dir / filename
            if not path.exists():
                add_issue(issues, "ERROR", "missing_fundamental", path, "Required financial file is missing.", "Re-crawl this symbol.")
                continue
            df, error = read_csv(path)
            if error or df is None or df.empty:
                continue
            cols = {clean_colname(col) for col in df.columns}
            if not cols.intersection({"year", "yearreport", "meta_yearreport"}):
                add_issue(issues, "WARN", "fundamental_schema", path, "No year/yearreport column found.", "Map time columns before feature engineering.", len(df))
            if filename != "ratio.csv" and len(df) < 4:
                add_issue(issues, "WARN", "short_history", path, "Less than 4 quarterly rows.", "Use longer history or exclude 4Q targets.", len(df))


def audit_market(data_root: Path, issues: list[dict[str, Any]]) -> None:
    root = data_root / "raw" / "market" / "equity"
    if not root.exists():
        add_issue(issues, "ERROR", "missing_folder", root, "Missing market/equity folder.", "Run vnstock collector.")
        return

    for symbol_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = symbol_dir / "ohlcv.csv"
        if not path.exists():
            add_issue(issues, "ERROR", "missing_ohlcv", path, "Missing OHLCV file.", "Re-crawl OHLCV or use backup source.")
            continue
        df, error = read_csv(path)
        if error or df is None or df.empty:
            continue
        cols = {clean_colname(col) for col in df.columns}
        for field, aliases in PRICE_FIELDS.items():
            if not cols.intersection(aliases):
                add_issue(issues, "ERROR", "market_schema", path, f"Missing {field} field.", "Normalize OHLCV columns.", len(df))
        close_col = next((col for col in df.columns if clean_colname(col) in PRICE_FIELDS["close"]), None)
        if close_col:
            close = pd.to_numeric(df[close_col], errors="coerce")
            bad_close = int(close.isna().sum() + (close <= 0).sum())
            if bad_close:
                add_issue(issues, "WARN", "bad_close", path, f"{bad_close} missing/non-positive close values.", "Forward-fill small gaps or use backup source.", len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit compact ML data tree.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="outputs/data_quality/ml_tree_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, Any]] = []
    audit_all_csv(data_root, issues)
    audit_fundamental(data_root, issues)
    audit_market(data_root, issues)

    report = pd.DataFrame(issues)
    if report.empty:
        report = pd.DataFrame(columns=["severity", "category", "path", "issue", "suggestion", "rows"])
    report.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Audit issues: {len(report)}")
    if not report.empty:
        print(report["severity"].value_counts().to_string())
        print(report["category"].value_counts().head(20).to_string())
    print(f"Report saved to: {output.resolve()}")


if __name__ == "__main__":
    main()

