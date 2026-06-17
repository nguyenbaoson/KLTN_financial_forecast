"""
Materialize ML-tree derived outputs.

Writes:
  data/features/growth_features.csv
  data/raw/analytics/feature_coverage.csv
  data/raw/analytics/symbol_quarter_summary.csv
  data/raw/insights/latest_company_signals.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.growth_feature_engineering import (
    GrowthFeatureEngineer,
    TARGET_COLS,
    profit_target_source,
    select_feature_columns,
)
from pipeline_crawl_data.pipeline_common import resolve_repo_path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def build_feature_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in dataset.columns:
        non_null = int(dataset[col].notna().sum())
        rows.append({
            "column": col,
            "dtype": str(dataset[col].dtype),
            "non_null_rows": non_null,
            "total_rows": int(len(dataset)),
            "non_null_ratio": non_null / len(dataset) if len(dataset) else 0.0,
            "unique_values": int(dataset[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["non_null_ratio", "column"], ascending=[False, True])


def build_symbol_quarter_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in [
            "revenue",
            "net_profit",
            "net_profit_parent",
            "revenue_g1q",
            "net_profit_g1q",
            "net_profit_parent_g1q",
            "q_return",
            "price_volatility",
            "roe",
            "roa",
            "pe",
            "pb",
        ]
        if col in dataset.columns
    ]
    cols = ["symbol", "year", "quarter", "yq_index", *numeric_cols]
    summary = dataset[cols].copy()
    summary["collected_at"] = datetime.now().isoformat(timespec="seconds")
    return summary.sort_values(["symbol", "yq_index"])


def build_latest_company_signals(dataset: pd.DataFrame) -> pd.DataFrame:
    latest = dataset.sort_values(["symbol", "yq_index"]).groupby("symbol", as_index=False).tail(1).copy()

    profit_source = profit_target_source(latest)
    profit_growth_col = f"{profit_source}_g1q" if profit_source else ""
    profit_growth = latest[profit_growth_col] if profit_growth_col in latest.columns else pd.Series(np.nan, index=latest.index)
    revenue_growth = latest["revenue_g1q"] if "revenue_g1q" in latest.columns else pd.Series(np.nan, index=latest.index)
    market_return = latest["q_return"] if "q_return" in latest.columns else pd.Series(np.nan, index=latest.index)
    leverage = latest["debt_to_asset_calc"] if "debt_to_asset_calc" in latest.columns else pd.Series(np.nan, index=latest.index)

    score = (
        profit_growth.fillna(0).clip(-50, 50) * 0.45
        + revenue_growth.fillna(0).clip(-50, 50) * 0.30
        + market_return.fillna(0).clip(-50, 50) * 0.20
        - leverage.fillna(0).clip(0, 2) * 5.0
    )
    latest["insight_score"] = score.round(3)
    latest["insight_signal"] = pd.cut(
        latest["insight_score"],
        bins=[-np.inf, 0, 10, np.inf],
        labels=["weak", "neutral", "strong"],
    )
    latest["early_warning"] = (
        (profit_growth < 0).fillna(False)
        | (revenue_growth < 0).fillna(False)
        | (market_return < -10).fillna(False)
    )
    latest["collected_at"] = datetime.now().isoformat(timespec="seconds")
    latest["profit_target_source"] = profit_source or ""

    cols = [
        "symbol",
        "year",
        "quarter",
        "yq_index",
        "insight_score",
        "insight_signal",
        "early_warning",
        "profit_target_source",
    ]
    for col in ["revenue_g1q", "net_profit_g1q", "net_profit_parent_g1q", "q_return", "roe", "pe", "pb"]:
        if col in latest.columns:
            cols.append(col)
    cols.append("collected_at")
    return latest[cols].sort_values("insight_score", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize feature, analytics, and insight CSVs from ML-tree raw data.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Empty means all available symbols.")
    parser.add_argument("--target", default="target_profit_growth_1q")
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()] or None

    engineer = GrowthFeatureEngineer()
    dataset = engineer.build_dataset_from_ml_tree(data_root=data_root, tickers=tickers)
    if dataset.empty:
        raise ValueError("Feature dataset is empty. Check data/raw/fundamental and data/raw/market/equity.")

    dataset["materialized_at"] = datetime.now().isoformat(timespec="seconds")
    dataset = drop_all_null_columns(dataset)
    save_df(dataset, data_root / "features" / "growth_features.csv")

    coverage = build_feature_coverage(dataset)
    save_df(coverage, data_root / "raw" / "analytics" / "feature_coverage.csv")

    summary = build_symbol_quarter_summary(dataset)
    save_df(summary, data_root / "raw" / "analytics" / "symbol_quarter_summary.csv")

    insights = build_latest_company_signals(dataset)
    save_df(insights, data_root / "raw" / "insights" / "latest_company_signals.csv")

    trainable = dataset.dropna(subset=[args.target]).copy() if args.target in dataset.columns else pd.DataFrame()
    selected_features = (
        select_feature_columns(trainable, args.target, args.min_feature_non_null)
        if not trainable.empty
        else []
    )
    run_summary = pd.DataFrame([{
        "rows": int(dataset.shape[0]),
        "symbols": int(dataset["symbol"].nunique()),
        "columns": int(dataset.shape[1]),
        "trainable_rows": int(trainable.shape[0]),
        "selected_feature_count": len(selected_features),
        "target": args.target,
        "materialized_at": datetime.now().isoformat(timespec="seconds"),
    }])
    save_df(run_summary, data_root / "raw" / "analytics" / "materialize_summary.csv")

    print(run_summary.to_string(index=False))


if __name__ == "__main__":
    main()

