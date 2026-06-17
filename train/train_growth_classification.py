"""
Train classification models for listed-company growth direction or buckets.

Examples:
  python -m train.train_growth_classification --target target_profit_up_4q
  python -m train.train_growth_classification --target target_growth_class_4q --dataset-path data/features/growth_features.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, RobustScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer, profit_target_source, select_feature_columns
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = Path("outputs/growth_classification")
ADAPTIVE_TARGET_HORIZONS = {1, 2, 3, 4, 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train growth direction/bucket classification models.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-layout", choices=["legacy", "ml_tree"], default="ml_tree")
    parser.add_argument("--dataset-path", default="", help="Optional prebuilt feature CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default="target_profit_up_4q")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Empty means all available rows.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=0,
        help="Keep only the top K train-split features ranked by mutual information. 0 disables this step.",
    )
    parser.add_argument(
        "--add-panel-cycle-features",
        action="store_true",
        help="Add same-quarter market/sector aggregate cycle features built from current and historical indicators.",
    )
    parser.add_argument(
        "--add-market-index-features",
        action="store_true",
        help="Add quarterly VNINDEX/VN30 return, volatility, and rolling regime features from raw market index OHLCV.",
    )
    parser.add_argument(
        "--add-industry-cycle-features",
        action="store_true",
        help="Add true ICB industry metadata/dummies and industry-level cycle aggregates when reference data exists.",
    )
    parser.add_argument(
        "--add-macro-regime-features",
        action="store_true",
        help="Add lagged and rolling macro regime features by quarter.",
    )
    parser.add_argument(
        "--add-target-cycle-lag-features",
        action="store_true",
        help="Add lagged historical target-rate cycle features. For 4Q targets this uses rates known four quarters later.",
    )
    parser.add_argument("--no-symbol-dummies", action="store_true")
    parser.add_argument(
        "--feature-set",
        choices=["all", "no_news", "news_only", "financial_only", "market_only"],
        default="all",
        help="Feature ablation for thesis experiments.",
    )
    parser.add_argument(
        "--calibrate-best",
        action="store_true",
        help="Calibrate the selected best model probabilities on the validation split before test predictions.",
    )
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="For binary targets, tune the positive-class probability threshold on validation.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="For binary targets, use this positive-class probability threshold instead of the default 0.5.",
    )
    parser.add_argument(
        "--quarterly-top-rate",
        type=float,
        default=None,
        help="For binary targets, predict the positive class for the top rate of probabilities within each quarter.",
    )
    parser.add_argument(
        "--quarterly-top-rate-from-validation",
        action="store_true",
        help="Set --quarterly-top-rate to the validation positive-class rate.",
    )
    parser.add_argument(
        "--quarterly-min-rate",
        type=float,
        default=None,
        help="For binary targets, ensure at least this positive-class rate within each quarter.",
    )
    parser.add_argument(
        "--quarterly-min-rate-from-validation",
        action="store_true",
        help="Set --quarterly-min-rate to the validation positive-class rate.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "recall", "precision"],
        default="balanced_accuracy",
        help="Validation metric used when tuning the binary decision threshold.",
    )
    parser.add_argument(
        "--include-weighted-voting",
        action="store_true",
        help="Add a weighted soft-voting ensemble built from validation scores of selected base models.",
    )
    parser.add_argument(
        "--voting-models",
        default="RandomForest,ExtraTrees,GradientBoosting,HistGradientBoosting,XGBoost,LightGBM,LogisticRegression",
        help="Comma-separated base models for weighted soft voting.",
    )
    parser.add_argument(
        "--voting-weight-metric",
        choices=["val_BalancedAccuracy", "val_F1", "val_AUC", "val_Accuracy"],
        default="val_BalancedAccuracy",
        help="Validation metric used as soft-voting model weights.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_BalancedAccuracy", "val_F1", "val_AUC", "val_Accuracy"],
        default="val_F1",
        help="Validation metric used to select the best model for predictions/calibration.",
    )
    parser.add_argument(
        "--enabled-models",
        default="",
        help="Comma-separated model names to train. Empty means all available models.",
    )
    parser.add_argument(
        "--adaptive-group",
        choices=["industry_l1", "industry_l2", "bank_nonbank"],
        default="industry_l1",
        help="Group used by target_adaptive_strong_profit_up_* targets.",
    )
    parser.add_argument(
        "--adaptive-strong-quantile",
        type=float,
        default=0.70,
        help="Train-split growth quantile used as the adaptive strong-growth threshold.",
    )
    parser.add_argument(
        "--adaptive-quality-quantile",
        type=float,
        default=0.20,
        help="Train-split group quantile used as the minimum ROA/ROE quality floor.",
    )
    parser.add_argument(
        "--adaptive-profit-ttm-quantile",
        type=float,
        default=0.20,
        help="Train-split group quantile used as the minimum net-profit TTM floor.",
    )
    return parser.parse_args()


def is_news_feature(col: str) -> bool:
    base_news_names = {
        "avg_sentiment",
        "min_sentiment",
        "max_sentiment",
        "sentiment_std",
        "avg_news_text_length",
        "avg_news_source_weight",
        "positive_news_count",
        "negative_news_count",
        "neutral_news_count",
        "risk_news_count",
        "growth_news_count",
        "earnings_news_count",
        "policy_news_count",
        "market_news_count",
        "positive_news_ratio",
        "negative_news_ratio",
        "risk_news_ratio",
        "growth_news_ratio",
        "earnings_news_ratio",
    }
    return (
        col.startswith("news_")
        or col.startswith("event_")
        or col.startswith("transformer_")
        or "_news_" in col
        or col.endswith("_news")
        or col.endswith("_news_count")
        or col.endswith("_news_ratio")
        or col in base_news_names
        or any(col.startswith(f"{name}_") for name in base_news_names)
    )


def apply_feature_set(feature_cols: List[str], feature_set: str) -> List[str]:
    if feature_set == "news_only":
        return [col for col in feature_cols if is_news_feature(col)]
    if feature_set == "no_news":
        return [col for col in feature_cols if not is_news_feature(col)]
    if feature_set == "market_only":
        return [col for col in feature_cols if is_market_feature(col)]
    if feature_set == "financial_only":
        return [col for col in feature_cols if not is_news_feature(col) and not is_market_feature(col)]
    return feature_cols


def is_market_feature(col: str) -> bool:
    market_names = {
        "close_eoq",
        "avg_volume",
        "q_return",
        "price_volatility",
        "momentum_3m",
        "avg_rel_volume",
    }
    return col in market_names or col.startswith("technical_")


def infer_sector_proxy(df: pd.DataFrame) -> pd.Series:
    bank_cols = [
        "bank_net_interest_income",
        "bank_customer_loans_gross",
        "bank_customer_deposits",
        "bank_credit_loss_provision",
    ]
    available = [col for col in bank_cols if col in df.columns]
    if not available:
        return pd.Series("all", index=df.index)
    bank_signal = df[available].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    return pd.Series(np.where(bank_signal, "bank", "nonbank"), index=df.index)


def add_industry_metadata(dataset: pd.DataFrame, data_root: str = "data") -> pd.DataFrame:
    industry_path = Path(data_root) / "raw" / "reference" / "equity" / "list_by_industry.csv"
    if not industry_path.exists() or "symbol" not in dataset.columns:
        return dataset

    industry = pd.read_csv(industry_path)
    required = {"symbol", "icb_level", "icb_code", "icb_name"}
    if not required.issubset(industry.columns):
        return dataset

    industry = industry.copy()
    industry["symbol"] = industry["symbol"].astype(str).str.upper()
    industry["icb_level"] = pd.to_numeric(industry["icb_level"], errors="coerce")
    level1 = (
        industry[industry["icb_level"].eq(1)]
        .sort_values(["symbol", "icb_code"])
        .drop_duplicates("symbol")[["symbol", "icb_code", "icb_name"]]
        .rename(columns={"icb_code": "industry_l1_code", "icb_name": "industry_l1_name"})
    )
    level2 = (
        industry[industry["icb_level"].eq(2)]
        .sort_values(["symbol", "icb_code"])
        .drop_duplicates("symbol")[["symbol", "icb_code", "icb_name"]]
        .rename(columns={"icb_code": "industry_l2_code", "icb_name": "industry_l2_name"})
    )

    result = dataset.copy()
    result["_symbol_upper_for_industry"] = result["symbol"].astype(str).str.upper()
    result = result.merge(level1, left_on="_symbol_upper_for_industry", right_on="symbol", how="left", suffixes=("", "_industry"))
    if "symbol_industry" in result.columns:
        result = result.drop(columns=["symbol_industry"])
    result = result.merge(level2, left_on="_symbol_upper_for_industry", right_on="symbol", how="left", suffixes=("", "_industry"))
    if "symbol_industry" in result.columns:
        result = result.drop(columns=["symbol_industry"])
    result = result.drop(columns=["_symbol_upper_for_industry"])

    for level_col in ("industry_l1_code", "industry_l2_code"):
        result[level_col] = result[level_col].astype(str).replace({"nan": "unknown", "<NA>": "unknown"})
        dummies = pd.get_dummies(result[level_col], prefix=level_col, dtype=int)
        result = pd.concat([result, dummies], axis=1)
    return result


def quarterly_index_features(index_path: Path, prefix: str) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(index_path)
    if not {"time", "close"}.issubset(raw.columns):
        return pd.DataFrame()

    raw = raw.copy()
    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    raw = raw.dropna(subset=["time"]).sort_values("time")
    raw["year"] = raw["time"].dt.year
    raw["quarter"] = raw["time"].dt.quarter
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw.get("volume"), errors="coerce") if "volume" in raw.columns else np.nan
    raw["_daily_return"] = raw["close"].pct_change()

    rows = []
    for (year, quarter), group in raw.groupby(["year", "quarter"], sort=True):
        close = group["close"].dropna()
        if close.empty:
            continue
        quarter_return = (close.iloc[-1] / close.iloc[0] - 1.0) * 100 if close.iloc[0] else np.nan
        running_max = close.cummax()
        drawdown = ((close / running_max) - 1.0).min() * 100 if not running_max.empty else np.nan
        rows.append({
            "year": int(year),
            "quarter": int(quarter),
            f"{prefix}_close_eoq": float(close.iloc[-1]),
            f"{prefix}_q_return": float(quarter_return),
            f"{prefix}_volatility": float(group["_daily_return"].std() * 100),
            f"{prefix}_max_drawdown": float(drawdown),
            f"{prefix}_avg_volume": float(group["volume"].mean()) if "volume" in group else np.nan,
        })

    features = pd.DataFrame(rows).sort_values(["year", "quarter"]).reset_index(drop=True)
    if features.empty:
        return features
    return_col = f"{prefix}_q_return"
    vol_col = f"{prefix}_volatility"
    for lag in (1, 2, 4):
        features[f"{return_col}_lag{lag}q"] = features[return_col].shift(lag)
        features[f"{vol_col}_lag{lag}q"] = features[vol_col].shift(lag)
    features[f"{return_col}_rolling4q_mean"] = features[return_col].rolling(4, min_periods=1).mean()
    features[f"{return_col}_rolling4q_std"] = features[return_col].rolling(4, min_periods=2).std()
    features[f"{prefix}_risk_on_regime"] = (features[f"{return_col}_rolling4q_mean"] > 0).astype(int)
    return features


def add_market_index_features(dataset: pd.DataFrame, data_root: str = "data") -> pd.DataFrame:
    result = dataset.copy()
    index_root = Path(data_root) / "raw" / "market" / "index"
    for index_name, prefix in [("VNINDEX", "vnindex"), ("VN30", "vn30")]:
        features = quarterly_index_features(index_root / index_name / "ohlcv.csv", prefix)
        if features.empty:
            continue
        result = result.merge(features, on=["year", "quarter"], how="left")
    return result


def add_macro_regime_features(dataset: pd.DataFrame) -> pd.DataFrame:
    if "yq_index" not in dataset.columns:
        return dataset
    result = dataset.copy()
    macro_cols = [
        "gdp_growth",
        "inflation",
        "interest_rate",
        "exchange_rate",
        "industrial_growth",
    ]
    macro_cols = [col for col in macro_cols if col in result.columns]
    if not macro_cols:
        return result

    macro = result[["yq_index", *macro_cols]].copy()
    for col in macro_cols:
        macro[col] = pd.to_numeric(macro[col], errors="coerce")
    macro = macro.groupby("yq_index", as_index=False)[macro_cols].mean().sort_values("yq_index")
    for col in macro_cols:
        for lag in (1, 2, 4):
            macro[f"{col}_macro_lag{lag}q"] = macro[col].shift(lag)
        macro[f"{col}_macro_chg1q"] = macro[col].diff(1)
        macro[f"{col}_macro_chg4q"] = macro[col].diff(4)
        macro[f"{col}_macro_rolling4q_mean"] = macro[col].rolling(4, min_periods=1).mean()
        macro[f"{col}_macro_rolling4q_std"] = macro[col].rolling(4, min_periods=2).std()
        expanding_mean = macro[col].expanding(min_periods=4).mean()
        expanding_std = macro[col].expanding(min_periods=4).std()
        macro[f"{col}_macro_expanding_z"] = (macro[col] - expanding_mean) / (expanding_std + 1e-9)
    return result.merge(macro.drop(columns=macro_cols), on="yq_index", how="left")


def add_panel_cycle_features(
    dataset: pd.DataFrame,
    target_col: str,
    add_target_lags: bool = False,
) -> pd.DataFrame:
    result = dataset.copy()
    if "yq_index" not in result.columns:
        return result

    if "industry_l1_code" in result.columns:
        result["_sector_proxy_for_cycle"] = result["industry_l1_code"].astype(str).fillna("unknown")
    else:
        result["_sector_proxy_for_cycle"] = infer_sector_proxy(result)
    cycle_cols = [
        "revenue_g4q",
        "gross_profit_g4q",
        "operating_profit_g4q",
        "profit_before_tax_g4q",
        "net_profit_g4q",
        "earnings_quality_calc",
        "q_return",
        "momentum_3m",
        "price_volatility",
        "avg_rel_volume",
        "gdp_growth",
        "inflation",
        "interest_rate",
        "exchange_rate",
    ]
    cycle_cols = [col for col in cycle_cols if col in result.columns]

    for col in cycle_cols:
        values = pd.to_numeric(result[col], errors="coerce")
        result[f"market_cycle_{col}_mean"] = values.groupby(result["yq_index"]).transform("mean")
        result[f"market_cycle_{col}_median"] = values.groupby(result["yq_index"]).transform("median")
        result[f"sector_cycle_{col}_mean"] = values.groupby([result["yq_index"], result["_sector_proxy_for_cycle"]]).transform("mean")
        if col.endswith("_g4q") or col in {"q_return", "momentum_3m"}:
            positive = (values > 0).astype(float).where(values.notna())
            strong = (values > 10).astype(float).where(values.notna())
            result[f"market_cycle_{col}_positive_rate"] = positive.groupby(result["yq_index"]).transform("mean")
            result[f"market_cycle_{col}_strong_rate"] = strong.groupby(result["yq_index"]).transform("mean")
            result[f"sector_cycle_{col}_positive_rate"] = positive.groupby([result["yq_index"], result["_sector_proxy_for_cycle"]]).transform("mean")
            result[f"sector_cycle_{col}_strong_rate"] = strong.groupby([result["yq_index"], result["_sector_proxy_for_cycle"]]).transform("mean")

    if add_target_lags and target_col in result.columns:
        target = pd.to_numeric(result[target_col], errors="coerce")
        periods = sorted(result["yq_index"].dropna().unique().tolist())
        market_rate = target.groupby(result["yq_index"]).mean().reindex(periods)
        for lag in (4, 5, 6):
            result[f"market_cycle_{target_col}_rate_lag{lag}q"] = result["yq_index"].map(market_rate.shift(lag))

        sector_rates = (
            pd.DataFrame({
                "yq_index": result["yq_index"],
                "_sector_proxy_for_cycle": result["_sector_proxy_for_cycle"],
                "_target": target,
            })
            .groupby(["_sector_proxy_for_cycle", "yq_index"], as_index=False)["_target"]
            .mean()
            .sort_values(["_sector_proxy_for_cycle", "yq_index"])
        )
        for lag in (4, 5, 6):
            col_name = f"sector_cycle_{target_col}_rate_lag{lag}q"
            sector_rates[col_name] = sector_rates.groupby("_sector_proxy_for_cycle")["_target"].shift(lag)
        keep_cols = [
            "_sector_proxy_for_cycle",
            "yq_index",
            *[f"sector_cycle_{target_col}_rate_lag{lag}q" for lag in (4, 5, 6)],
        ]
        result = result.merge(sector_rates[keep_cols], on=["_sector_proxy_for_cycle", "yq_index"], how="left")

    return result.drop(columns=["_sector_proxy_for_cycle"])


def derive_classification_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    for horizon in sorted(ADAPTIVE_TARGET_HORIZONS):
        growth_col = f"target_profit_growth_{horizon}q"
        up_col = f"target_profit_up_{horizon}q"
        strong_up_col = f"target_strong_profit_up_{horizon}q"
        class_col = f"target_growth_class_{horizon}q"
        if growth_col not in result.columns:
            continue
        target = pd.to_numeric(result[growth_col], errors="coerce")
        if up_col not in result.columns:
            result[up_col] = np.where(target.notna(), (target > 0).astype(int), np.nan)
        if strong_up_col not in result.columns:
            result[strong_up_col] = np.select(
                [target > 10, target <= 0],
                [1, 0],
                default=np.nan,
            )
        if class_col not in result.columns:
            result[class_col] = pd.cut(
                target,
                bins=[-np.inf, 0, 10, np.inf],
                labels=["low", "medium", "high"],
            )
    return result


def is_adaptive_strong_target(target_col: str) -> bool:
    return target_col.startswith("target_adaptive_strong_profit_up_") and target_col.endswith("q")


def adaptive_target_horizon(target_col: str) -> int:
    suffix = target_col.removeprefix("target_adaptive_strong_profit_up_").removesuffix("q")
    try:
        horizon = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Cannot parse adaptive target horizon from {target_col!r}.") from exc
    if horizon not in ADAPTIVE_TARGET_HORIZONS:
        supported = ", ".join(f"{item}q" for item in sorted(ADAPTIVE_TARGET_HORIZONS))
        raise ValueError(f"Adaptive strong-growth targets currently support only {supported} horizons.")
    return horizon


def add_net_profit_ttm_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    profit_col = profit_target_source(dataset)
    if profit_col is None or "symbol" not in dataset.columns or "yq_index" not in dataset.columns:
        return dataset
    result = dataset.sort_values(["symbol", "yq_index"]).copy()
    result["net_profit_ttm_calc"] = (
        result.groupby("symbol")[profit_col]
        .transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(4, min_periods=1).sum())
    )
    result["net_profit_ttm_source"] = profit_col
    return result.sort_index()


def adaptive_group_series(dataset: pd.DataFrame, group_mode: str) -> pd.Series:
    if group_mode == "industry_l1" and "industry_l1_code" in dataset.columns:
        group = dataset["industry_l1_code"].astype(str).replace({"nan": "unknown", "<NA>": "unknown"})
        if group.ne("unknown").mean() >= 0.5:
            return group
    if group_mode == "industry_l2" and "industry_l2_code" in dataset.columns:
        group = dataset["industry_l2_code"].astype(str).replace({"nan": "unknown", "<NA>": "unknown"})
        if group.ne("unknown").mean() >= 0.5:
            return group
    return infer_sector_proxy(dataset)


def q_or_nan(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(quantile))


def add_adaptive_strong_profit_target(
    dataset: pd.DataFrame,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    group_mode: str,
    strong_quantile: float,
    quality_quantile: float,
    profit_ttm_quantile: float,
    output_dir: Path,
    min_group_train_rows: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    horizon = adaptive_target_horizon(target_col)
    growth_col = f"target_profit_growth_{horizon}q"
    if growth_col not in dataset.columns:
        raise ValueError(f"Adaptive target {target_col!r} requires {growth_col!r}.")

    result = add_net_profit_ttm_feature(dataset)
    result["_adaptive_group"] = adaptive_group_series(result, group_mode)

    base = result.dropna(subset=[growth_col]).copy()
    train_base, _, _, split_info = split_by_time(base, train_ratio, val_ratio)

    global_growth_threshold = max(0.0, q_or_nan(train_base[growth_col], strong_quantile))
    quality_cols = [col for col in ["roa_ttm_calc", "roa_calc", "roa", "roe_ttm_calc", "roe_calc", "roe"] if col in train_base.columns]
    roa_cols = [col for col in ["roa_ttm_calc", "roa_calc", "roa"] if col in train_base.columns]
    roe_cols = [col for col in ["roe_ttm_calc", "roe_calc", "roe"] if col in train_base.columns]

    if roa_cols:
        train_base["_adaptive_roa_quality"] = train_base[roa_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    else:
        train_base["_adaptive_roa_quality"] = np.nan
    if roe_cols:
        train_base["_adaptive_roe_quality"] = train_base[roe_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    else:
        train_base["_adaptive_roe_quality"] = np.nan

    global_roa_floor = q_or_nan(train_base["_adaptive_roa_quality"], quality_quantile)
    global_roe_floor = q_or_nan(train_base["_adaptive_roe_quality"], quality_quantile)
    global_profit_ttm_floor = max(0.0, q_or_nan(train_base["net_profit_ttm_calc"], profit_ttm_quantile))

    threshold_rows: List[Dict[str, Any]] = []
    thresholds: Dict[str, Dict[str, float]] = {}
    for group, group_df in train_base.groupby("_adaptive_group", dropna=False):
        group_key = str(group)
        use_group = len(group_df) >= min_group_train_rows
        growth_threshold = q_or_nan(group_df[growth_col], strong_quantile) if use_group else float("nan")
        roa_floor = q_or_nan(group_df["_adaptive_roa_quality"], quality_quantile) if use_group else float("nan")
        roe_floor = q_or_nan(group_df["_adaptive_roe_quality"], quality_quantile) if use_group else float("nan")
        profit_ttm_floor = q_or_nan(group_df["net_profit_ttm_calc"], profit_ttm_quantile) if use_group else float("nan")

        if pd.isna(growth_threshold):
            growth_threshold = global_growth_threshold
        if pd.isna(roa_floor):
            roa_floor = global_roa_floor
        if pd.isna(roe_floor):
            roe_floor = global_roe_floor
        if pd.isna(profit_ttm_floor):
            profit_ttm_floor = global_profit_ttm_floor

        growth_threshold = max(0.0, float(growth_threshold))
        profit_ttm_floor = max(0.0, float(profit_ttm_floor))
        thresholds[group_key] = {
            "growth_threshold": growth_threshold,
            "roa_floor": float(roa_floor) if pd.notna(roa_floor) else float("-inf"),
            "roe_floor": float(roe_floor) if pd.notna(roe_floor) else float("-inf"),
            "profit_ttm_floor": profit_ttm_floor,
            "train_rows": int(len(group_df)),
            "used_group_threshold": bool(use_group),
        }
        threshold_rows.append({"group": group_key, **thresholds[group_key]})

    fallback_thresholds = {
        "growth_threshold": global_growth_threshold,
        "roa_floor": float(global_roa_floor) if pd.notna(global_roa_floor) else float("-inf"),
        "roe_floor": float(global_roe_floor) if pd.notna(global_roe_floor) else float("-inf"),
        "profit_ttm_floor": global_profit_ttm_floor,
        "train_rows": int(len(train_base)),
        "used_group_threshold": False,
    }

    group_thresholds = result["_adaptive_group"].astype(str).map(thresholds)
    for threshold_name in ("growth_threshold", "roa_floor", "roe_floor", "profit_ttm_floor"):
        result[f"_adaptive_{threshold_name}"] = group_thresholds.apply(
            lambda item: item.get(threshold_name, fallback_thresholds[threshold_name]) if isinstance(item, dict) else fallback_thresholds[threshold_name]
        )

    if roa_cols:
        result["_adaptive_roa_quality"] = result[roa_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    else:
        result["_adaptive_roa_quality"] = np.nan
    if roe_cols:
        result["_adaptive_roe_quality"] = result[roe_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    else:
        result["_adaptive_roe_quality"] = np.nan

    growth = pd.to_numeric(result[growth_col], errors="coerce")
    profit_ttm = pd.to_numeric(result["net_profit_ttm_calc"], errors="coerce")
    roa_ok = result["_adaptive_roa_quality"] >= result["_adaptive_roa_floor"]
    roe_ok = result["_adaptive_roe_quality"] >= result["_adaptive_roe_floor"]
    quality_ok = roa_ok | roe_ok
    profit_base_ok = profit_ttm >= result["_adaptive_profit_ttm_floor"]
    strong_growth = growth > result["_adaptive_growth_threshold"]
    down_or_flat = growth <= 0

    result[target_col] = np.select(
        [strong_growth & quality_ok & profit_base_ok, down_or_flat],
        [1, 0],
        default=np.nan,
    )

    helper_cols = [
        "_adaptive_growth_threshold",
        "_adaptive_roa_floor",
        "_adaptive_roe_floor",
        "_adaptive_profit_ttm_floor",
        "_adaptive_roa_quality",
        "_adaptive_roe_quality",
    ]
    result = result.drop(columns=helper_cols)
    pd.DataFrame(threshold_rows).sort_values("group").to_csv(output_dir / "adaptive_target_thresholds.csv", index=False)

    target_counts = result[target_col].value_counts(dropna=False).to_dict()
    info = {
        "target_type": "adaptive_strong_profit_up",
        "target_col": target_col,
        "growth_col": growth_col,
        "profit_target_source": profit_target_source(dataset),
        "profit_ttm_source": str(result.get("net_profit_ttm_source", pd.Series([""])).dropna().iloc[0])
        if "net_profit_ttm_source" in result.columns and result["net_profit_ttm_source"].notna().any()
        else "",
        "group_mode_requested": group_mode,
        "group_col": "_adaptive_group",
        "strong_quantile": strong_quantile,
        "quality_quantile": quality_quantile,
        "profit_ttm_quantile": profit_ttm_quantile,
        "min_group_train_rows": min_group_train_rows,
        "split_used_for_thresholds": split_info,
        "fallback_thresholds": fallback_thresholds,
        "quality_features": quality_cols,
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
    }
    return result, info


def load_dataset(args: argparse.Namespace, tickers: List[str] | None) -> pd.DataFrame:
    if args.dataset_path:
        dataset = pd.read_csv(args.dataset_path)
        if tickers:
            dataset = dataset[dataset["symbol"].astype(str).str.upper().isin(tickers)].copy()
        return derive_classification_targets(dataset)

    engineer = GrowthFeatureEngineer()
    if args.data_layout == "ml_tree":
        dataset = engineer.build_dataset_from_ml_tree(data_root=args.data_root, tickers=tickers)
    else:
        dataset = engineer.build_dataset(raw_root=args.raw_root, tickers=tickers)
    return derive_classification_targets(dataset)


def get_models(n_classes: int) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ]),
        "LDA": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "QDA": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("pca", PCA(n_components=10, random_state=42)),
            ("model", QuadraticDiscriminantAnalysis(reg_param=0.2)),
        ]),
        "KNN10": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", KNeighborsClassifier(n_neighbors=10, weights="distance")),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=900,
                max_depth=8,
                min_samples_leaf=2,
                max_features=0.75,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=900,
                max_depth=10,
                min_samples_leaf=2,
                max_features=0.75,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=4,
                subsample=0.85,
                random_state=42,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.5,
                random_state=42,
            )),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ]),
    }
    if XGBOOST_AVAILABLE:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_alpha=0.1,
                reg_lambda=2.0,
                objective=objective,
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                random_state=42,
                n_jobs=-1,
            )),
        ])
    if LIGHTGBM_AVAILABLE:
        objective = "binary" if n_classes == 2 else "multiclass"
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=450,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=5,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=2.0,
                objective=objective,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])
    return models


class WeightedSoftVotingClassifier(ClassifierMixin, BaseEstimator):
    """Prefit weighted soft-voting classifier for models with predict_proba."""

    def __init__(self, estimators: Dict[str, Pipeline], weights: Dict[str, float], classes: np.ndarray):
        self.estimators = estimators
        self.weights = weights
        self.classes = classes
        self.classes_ = np.asarray(classes)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "WeightedSoftVotingClassifier":
        self.n_features_in_ = X.shape[1]
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        total_weight = float(sum(self.weights.values()))
        if total_weight <= 0:
            raise ValueError("WeightedSoftVotingClassifier has no positive weights.")

        weighted_proba: np.ndarray | None = None
        for name, estimator in self.estimators.items():
            proba = predict_proba_or_none(estimator, X)
            if proba is None:
                continue
            contribution = proba * self.weights[name]
            weighted_proba = contribution if weighted_proba is None else weighted_proba + contribution

        if weighted_proba is None:
            raise ValueError("No voting estimator can produce probabilities.")
        return weighted_proba / total_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def encode_target(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    encoder = LabelEncoder()
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    train["_target_encoded"] = encoder.fit_transform(train[target_col].astype(str))
    known = set(encoder.classes_)
    for split_name, frame in [("validation", val), ("test", test)]:
        unknown = sorted(set(frame[target_col].astype(str)) - known)
        if unknown:
            raise ValueError(f"{split_name} contains target classes unseen in train: {unknown}")
        frame["_target_encoded"] = encoder.transform(frame[target_col].astype(str))
    return train, val, test, encoder


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int) -> Dict[str, float]:
    average = "binary" if n_classes == 2 else "weighted"
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["AUC"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["Brier"] = float(brier_score_loss(y_true, y_proba[:, 1]))
            else:
                metrics["AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except ValueError:
            metrics["AUC"] = float("nan")
    return metrics


def predict_proba_or_none(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def build_voting_weights(results: pd.DataFrame, model_names: List[str], metric: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    result_by_model = results.set_index("model")
    for model_name in model_names:
        if model_name not in result_by_model.index or metric not in result_by_model.columns:
            continue
        value = float(result_by_model.loc[model_name, metric])
        if np.isfinite(value) and value > 0:
            scores[model_name] = value
    if not scores:
        return {}
    score_sum = sum(scores.values())
    return {name: value / score_sum for name, value in scores.items()}


def select_top_k_features_by_mi(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int,
    output_dir: Path,
) -> Tuple[List[str], pd.DataFrame]:
    if top_k <= 0 or top_k >= len(feature_cols):
        return feature_cols, pd.DataFrame()

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = imputer.fit_transform(train_df[feature_cols])
    y = train_df["_target_encoded"].values
    scores = mutual_info_classif(X, y, random_state=42)
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "mutual_info": np.nan_to_num(scores, nan=0.0),
    }).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected = ranking.head(top_k)["feature"].tolist()
    ranking.to_csv(output_dir / "feature_selection_mutual_info.csv", index=False)
    return selected, ranking


def positive_class_index(encoder: LabelEncoder) -> int:
    labels = [str(label) for label in encoder.classes_]
    for candidate in ("1.0", "1", "up", "high"):
        if candidate in labels:
            return labels.index(candidate)
    return len(labels) - 1


def predict_with_threshold(
    model: Pipeline,
    X: pd.DataFrame,
    threshold: float | None,
    positive_idx: int,
) -> np.ndarray:
    if threshold is None:
        return model.predict(X)
    proba = predict_proba_or_none(model, X)
    if proba is None or proba.shape[1] != 2:
        return model.predict(X)
    negative_idx = 1 - positive_idx
    return np.where(proba[:, positive_idx] >= threshold, positive_idx, negative_idx)


def predict_with_quarterly_top_rate(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_cols: List[str],
    positive_idx: int,
    top_rate: float,
) -> np.ndarray:
    if not 0 <= top_rate <= 1:
        raise ValueError("quarterly top rate must be between 0 and 1.")
    proba = predict_proba_or_none(model, frame[feature_cols])
    if proba is None or proba.shape[1] != 2:
        return model.predict(frame[feature_cols])

    negative_idx = 1 - positive_idx
    pred = np.full(frame.shape[0], negative_idx, dtype=int)
    work = frame[["year", "quarter"]].copy() if {"year", "quarter"}.issubset(frame.columns) else pd.DataFrame(index=frame.index)
    work["_row_position"] = np.arange(frame.shape[0])
    work["_positive_proba"] = proba[:, positive_idx]
    group_cols = ["year", "quarter"] if {"year", "quarter"}.issubset(work.columns) else [lambda values: 0]

    for _, group in work.groupby(group_cols, sort=False):
        n_positive = int(round(top_rate * len(group)))
        if n_positive <= 0:
            continue
        selected = group.sort_values("_positive_proba", ascending=False).head(n_positive)["_row_position"].to_numpy()
        pred[selected] = positive_idx
    return pred


def predict_with_policy(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_cols: List[str],
    threshold: float | None,
    positive_idx: int,
    quarterly_top_rate: float | None = None,
    quarterly_min_rate: float | None = None,
) -> np.ndarray:
    if quarterly_top_rate is not None:
        return predict_with_quarterly_top_rate(model, frame, feature_cols, positive_idx, quarterly_top_rate)
    pred = predict_with_threshold(model, frame[feature_cols], threshold, positive_idx)
    if quarterly_min_rate is None:
        return pred
    if not 0 <= quarterly_min_rate <= 1:
        raise ValueError("quarterly min rate must be between 0 and 1.")

    proba = predict_proba_or_none(model, frame[feature_cols])
    if proba is None or proba.shape[1] != 2:
        return pred
    work = frame[["year", "quarter"]].copy() if {"year", "quarter"}.issubset(frame.columns) else pd.DataFrame(index=frame.index)
    work["_row_position"] = np.arange(frame.shape[0])
    work["_positive_proba"] = proba[:, positive_idx]
    group_cols = ["year", "quarter"] if {"year", "quarter"}.issubset(work.columns) else [lambda values: 0]
    for _, group in work.groupby(group_cols, sort=False):
        n_positive_min = int(round(quarterly_min_rate * len(group)))
        current_positive = int((pred[group["_row_position"].to_numpy()] == positive_idx).sum())
        needed = n_positive_min - current_positive
        if needed <= 0:
            continue
        selected = (
            group[pred[group["_row_position"].to_numpy()] != positive_idx]
            .sort_values("_positive_proba", ascending=False)
            .head(needed)["_row_position"]
            .to_numpy()
        )
        pred[selected] = positive_idx
    return pred


def tune_binary_threshold(
    model: Pipeline,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    positive_idx: int,
    metric: str = "balanced_accuracy",
) -> Tuple[float, float]:
    proba = predict_proba_or_none(model, val_df[feature_cols])
    if proba is None or proba.shape[1] != 2:
        return 0.5, float("nan")
    y_true = val_df["_target_encoded"].values
    negative_idx = 1 - positive_idx
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = np.where(proba[:, positive_idx] >= threshold, positive_idx, negative_idx)
        if metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=positive_idx, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, pos_label=positive_idx, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, pos_label=positive_idx, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def fit_and_evaluate(
    models: Dict[str, Pipeline],
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_classes: int,
    include_weighted_voting: bool = False,
    voting_model_names: List[str] | None = None,
    voting_weight_metric: str = "val_BalancedAccuracy",
    selection_metric: str = "val_F1",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Pipeline] = {}
    X_train, y_train = train_df[feature_cols], train_df["_target_encoded"].values
    X_val, y_val = val_df[feature_cols], val_df["_target_encoded"].values
    X_test, y_test = test_df[feature_cols], test_df["_target_encoded"].values

    for model_name, pipeline in models.items():
        logger.info("Training %s", model_name)
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)
        val_metrics = classification_metrics(y_val, val_pred, predict_proba_or_none(pipeline, X_val), n_classes)
        test_metrics = classification_metrics(y_test, test_pred, predict_proba_or_none(pipeline, X_test), n_classes)
        rows.append({
            "model": model_name,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        })
        joblib.dump(pipeline, output_dir / f"{model_name.lower()}_classifier.joblib")
        fitted_models[model_name] = pipeline

    if include_weighted_voting:
        base_results = pd.DataFrame(rows)
        requested = voting_model_names or []
        available = [
            name
            for name in requested
            if name in fitted_models and predict_proba_or_none(fitted_models[name], X_val) is not None
        ]
        weights = build_voting_weights(base_results, available, voting_weight_metric)
        if weights:
            estimators = {name: fitted_models[name] for name in weights}
            voting_model = WeightedSoftVotingClassifier(estimators, weights, classes=np.arange(n_classes))
            val_proba = voting_model.predict_proba(X_val)
            test_proba = voting_model.predict_proba(X_test)
            val_pred = voting_model.predict(X_val)
            test_pred = voting_model.predict(X_test)
            val_metrics = classification_metrics(y_val, val_pred, val_proba, n_classes)
            test_metrics = classification_metrics(y_test, test_pred, test_proba, n_classes)
            rows.append({
                "model": "WeightedSoftVoting",
                "voting_weight_metric": voting_weight_metric,
                "voting_weights": json.dumps(weights, sort_keys=True),
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            })
            joblib.dump(voting_model, output_dir / "weightedsoftvoting_classifier.joblib")
        else:
            logger.warning("Skipping WeightedSoftVoting: no usable positive validation weights.")

    sort_cols = [selection_metric]
    for fallback in ["val_BalancedAccuracy", "val_F1", "val_Accuracy"]:
        if fallback not in sort_cols:
            sort_cols.append(fallback)
    return pd.DataFrame(rows).sort_values(sort_cols, ascending=False).reset_index(drop=True)


def save_predictions(
    model: Pipeline,
    encoder: LabelEncoder,
    feature_cols: List[str],
    target_col: str,
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    decision_threshold: float | None = None,
    quarterly_top_rate: float | None = None,
    quarterly_min_rate: float | None = None,
) -> None:
    frames: List[pd.DataFrame] = []
    positive_idx = positive_class_index(encoder)
    for split_name, split_df in splits.items():
        pred_encoded = predict_with_policy(
            model,
            split_df,
            feature_cols,
            decision_threshold,
            positive_idx,
            quarterly_top_rate=quarterly_top_rate,
            quarterly_min_rate=quarterly_min_rate,
        )
        proba = predict_proba_or_none(model, split_df[feature_cols])
        pred_df = split_df[["symbol", "year", "quarter", "yq_index", target_col]].copy()
        pred_df["split"] = split_name
        pred_df["actual_label"] = split_df[target_col].astype(str)
        pred_df["predicted_label"] = encoder.inverse_transform(pred_encoded.astype(int))
        if decision_threshold is not None:
            pred_df["decision_threshold"] = decision_threshold
        if quarterly_top_rate is not None:
            pred_df["quarterly_top_rate"] = quarterly_top_rate
        if quarterly_min_rate is not None:
            pred_df["quarterly_min_rate"] = quarterly_min_rate
        if proba is not None:
            for idx, label in enumerate(encoder.classes_):
                pred_df[f"prob_{label}"] = proba[:, idx]
            pred_df["prediction_confidence"] = proba.max(axis=1)
        frames.append(pred_df)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


def save_latest_forecast(
    model: Pipeline,
    encoder: LabelEncoder,
    dataset: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    decision_threshold: float | None = None,
    quarterly_top_rate: float | None = None,
    quarterly_min_rate: float | None = None,
) -> None:
    latest_idx = dataset.sort_values(["symbol", "yq_index"]).groupby("symbol").tail(1).index
    latest = dataset.loc[latest_idx].copy()
    pred_encoded = predict_with_policy(
        model,
        latest,
        feature_cols,
        decision_threshold,
        positive_class_index(encoder),
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    proba = predict_proba_or_none(model, latest[feature_cols])
    latest["predicted_label"] = encoder.inverse_transform(pred_encoded.astype(int))
    if decision_threshold is not None:
        latest["decision_threshold"] = decision_threshold
    if quarterly_top_rate is not None:
        latest["quarterly_top_rate"] = quarterly_top_rate
    if quarterly_min_rate is not None:
        latest["quarterly_min_rate"] = quarterly_min_rate
    if proba is not None:
        for idx, label in enumerate(encoder.classes_):
            latest[f"prob_{label}"] = proba[:, idx]
        latest["prediction_confidence"] = proba.max(axis=1)
    cols = ["symbol", "year", "quarter", "yq_index", "predicted_label"]
    if "decision_threshold" in latest.columns:
        cols.append("decision_threshold")
    if "quarterly_top_rate" in latest.columns:
        cols.append("quarterly_top_rate")
    if "quarterly_min_rate" in latest.columns:
        cols.append("quarterly_min_rate")
    cols += [col for col in latest.columns if col.startswith("prob_")]
    if "prediction_confidence" in latest.columns:
        cols.append("prediction_confidence")
    latest[cols].sort_values("symbol").to_csv(output_dir / "latest_forecast.csv", index=False)


def save_feature_importance(model: Pipeline, feature_cols: List[str], output_dir: Path) -> None:
    if not hasattr(model, "named_steps"):
        return
    estimator = model.named_steps.get("model")
    if not hasattr(estimator, "feature_importances_"):
        return
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": estimator.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def save_confusion_matrices(
    model: Pipeline,
    encoder: LabelEncoder,
    feature_cols: List[str],
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    decision_threshold: float | None = None,
    quarterly_top_rate: float | None = None,
    quarterly_min_rate: float | None = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    positive_idx = positive_class_index(encoder)
    for split_name, split_df in splits.items():
        y_true = split_df["_target_encoded"].values
        y_pred = predict_with_policy(
            model,
            split_df,
            feature_cols,
            decision_threshold,
            positive_idx,
            quarterly_top_rate=quarterly_top_rate,
            quarterly_min_rate=quarterly_min_rate,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(encoder.classes_))))
        for actual_idx, actual_label in enumerate(encoder.classes_):
            for pred_idx, pred_label in enumerate(encoder.classes_):
                rows.append({
                    "split": split_name,
                    "actual": actual_label,
                    "predicted": pred_label,
                    "count": int(matrix[actual_idx, pred_idx]),
                })
    pd.DataFrame(rows).to_csv(output_dir / "confusion_matrix.csv", index=False)


def calibrate_prefit_model(
    model: Pipeline,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    method: str,
    n_classes: int,
    output_dir: Path,
    decision_threshold: float | None = None,
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
        cv=None,
    )
    calibrated.fit(val_df[feature_cols], val_df["_target_encoded"].values)
    positive_idx = 1 if n_classes == 2 else 0
    val_pred = predict_with_threshold(calibrated, val_df[feature_cols], decision_threshold, positive_idx)
    test_pred = predict_with_threshold(calibrated, test_df[feature_cols], decision_threshold, positive_idx)
    val_metrics = classification_metrics(
        val_df["_target_encoded"].values,
        val_pred,
        calibrated.predict_proba(val_df[feature_cols]),
        n_classes,
    )
    test_metrics = classification_metrics(
        test_df["_target_encoded"].values,
        test_pred,
        calibrated.predict_proba(test_df[feature_cols]),
        n_classes,
    )
    metrics = {
        "model": "CalibratedBest",
        "calibration_method": method,
        "decision_threshold": decision_threshold if decision_threshold is not None else 0.5,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "calibrated_model_results.csv", index=False)
    joblib.dump(calibrated, output_dir / "calibrated_best_classifier.joblib")
    return calibrated, metrics


def save_selected_model_metrics(
    model: Pipeline,
    model_name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_classes: int,
    output_dir: Path,
    decision_threshold: float | None = None,
    threshold_metric: str | None = None,
    quarterly_top_rate: float | None = None,
    quarterly_min_rate: float | None = None,
) -> Dict[str, Any]:
    positive_idx = 1 if n_classes == 2 else 0
    val_pred = predict_with_policy(
        model,
        val_df,
        feature_cols,
        decision_threshold,
        positive_idx,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    test_pred = predict_with_policy(
        model,
        test_df,
        feature_cols,
        decision_threshold,
        positive_idx,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    val_proba = predict_proba_or_none(model, val_df[feature_cols])
    test_proba = predict_proba_or_none(model, test_df[feature_cols])
    val_metrics = classification_metrics(val_df["_target_encoded"].values, val_pred, val_proba, n_classes)
    test_metrics = classification_metrics(test_df["_target_encoded"].values, test_pred, test_proba, n_classes)
    metrics = {
        "model": model_name,
        "decision_threshold": decision_threshold if decision_threshold is not None else 0.5,
        "threshold_metric": threshold_metric,
        "quarterly_top_rate": quarterly_top_rate,
        "quarterly_min_rate": quarterly_min_rate,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "selected_model_results.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()] or None
    dataset = load_dataset(args, tickers)
    if dataset.empty:
        raise ValueError("Feature dataset is empty. Check dataset path/raw inputs.")
    adaptive_target_info: Dict[str, Any] | None = None
    adaptive_target = is_adaptive_strong_target(args.target)

    if not adaptive_target and args.target not in dataset.columns:
        raise ValueError(f"Target {args.target!r} does not exist in dataset.")

    needs_industry_metadata = args.add_industry_cycle_features or (
        adaptive_target and args.adaptive_group in {"industry_l1", "industry_l2"}
    )
    if needs_industry_metadata:
        dataset = add_industry_metadata(dataset, args.data_root)
    if args.add_market_index_features:
        dataset = add_market_index_features(dataset, args.data_root)
    if args.add_macro_regime_features:
        dataset = add_macro_regime_features(dataset)
    if adaptive_target:
        dataset, adaptive_target_info = add_adaptive_strong_profit_target(
            dataset=dataset,
            target_col=args.target,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            group_mode=args.adaptive_group,
            strong_quantile=args.adaptive_strong_quantile,
            quality_quantile=args.adaptive_quality_quantile,
            profit_ttm_quantile=args.adaptive_profit_ttm_quantile,
            output_dir=output_dir,
        )
    if args.add_panel_cycle_features or args.add_target_cycle_lag_features:
        dataset = add_panel_cycle_features(
            dataset,
            args.target,
            add_target_lags=args.add_target_cycle_lag_features,
        )
    dataset = dataset.dropna(axis=1, how="all")

    trainable = dataset.dropna(subset=[args.target]).copy()
    trainable[args.target] = trainable[args.target].astype(str)
    if trainable[args.target].nunique() < 2:
        raise ValueError(f"Target {args.target!r} has fewer than 2 classes.")

    symbol_dummy_cols: List[str] = []
    if not args.no_symbol_dummies:
        symbol_values = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique().tolist())
        dataset, symbol_dummy_cols = add_symbol_dummies(dataset, symbol_values)
        trainable, _ = add_symbol_dummies(trainable, symbol_values)

    dataset.to_csv(output_dir / "classification_dataset.csv", index=False)

    feature_cols = select_feature_columns(trainable, args.target, min_non_null_ratio=args.min_feature_non_null)
    feature_cols = apply_feature_set(feature_cols, args.feature_set)
    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable features: {len(feature_cols)}")

    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, feature_pruning = prune_feature_columns(train_df, feature_cols, max_corr=args.max_feature_corr)
    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable train features after pruning: {len(feature_cols)}")

    train_df, val_df, test_df, encoder = encode_target(train_df, val_df, test_df, args.target)
    feature_selection_info: Dict[str, Any] = {
        "method": None,
        "top_k_requested": int(args.top_k_features),
        "n_features_before_top_k": len(feature_cols),
        "n_features_after_top_k": len(feature_cols),
    }
    if args.top_k_features > 0:
        feature_cols, mi_ranking = select_top_k_features_by_mi(train_df, feature_cols, args.top_k_features, output_dir)
        if len(feature_cols) < 5:
            raise ValueError(f"Too few usable train features after top-k selection: {len(feature_cols)}")
        feature_selection_info = {
            "method": "mutual_info_classif",
            "top_k_requested": int(args.top_k_features),
            "n_features_before_top_k": int(mi_ranking.shape[0]),
            "n_features_after_top_k": len(feature_cols),
        }
    models = get_models(n_classes=len(encoder.classes_))
    enabled_model_names = [item.strip() for item in args.enabled_models.split(",") if item.strip()]
    if enabled_model_names:
        unknown_models = sorted(set(enabled_model_names) - set(models))
        if unknown_models:
            raise ValueError(f"Unknown enabled models: {unknown_models}. Available: {sorted(models)}")
        models = {name: models[name] for name in enabled_model_names}
    voting_model_names = [item.strip() for item in args.voting_models.split(",") if item.strip()]
    results = fit_and_evaluate(
        models,
        feature_cols,
        train_df,
        val_df,
        test_df,
        output_dir,
        len(encoder.classes_),
        include_weighted_voting=args.include_weighted_voting,
        voting_model_names=voting_model_names,
        voting_weight_metric=args.voting_weight_metric,
        selection_metric=args.selection_metric,
    )
    results.to_csv(output_dir / "model_results.csv", index=False)

    best_model_name = str(results.iloc[0]["model"])
    best_model = joblib.load(output_dir / f"{best_model_name.lower()}_classifier.joblib")
    prediction_model = best_model
    calibrated_metrics: Dict[str, Any] | None = None
    decision_threshold: float | None = None
    threshold_tuning_score: float | None = None
    quarterly_top_rate: float | None = None
    quarterly_min_rate: float | None = None
    if args.calibrate_best:
        prediction_model, calibrated_metrics = calibrate_prefit_model(
            best_model,
            val_df,
            test_df,
            feature_cols,
            args.calibration_method,
            len(encoder.classes_),
            output_dir,
        )
    if args.tune_threshold and len(encoder.classes_) == 2:
        decision_threshold, threshold_tuning_score = tune_binary_threshold(
            prediction_model,
            val_df,
            feature_cols,
            positive_class_index(encoder),
            metric=args.threshold_metric,
        )
        if args.calibrate_best:
            prediction_model, calibrated_metrics = calibrate_prefit_model(
                best_model,
                val_df,
                test_df,
                feature_cols,
                args.calibration_method,
                len(encoder.classes_),
                output_dir,
                decision_threshold=decision_threshold,
            )
    elif args.fixed_threshold is not None and len(encoder.classes_) == 2:
        if not 0 <= args.fixed_threshold <= 1:
            raise ValueError("--fixed-threshold must be between 0 and 1.")
        decision_threshold = float(args.fixed_threshold)
    if len(encoder.classes_) == 2:
        if args.quarterly_top_rate_from_validation:
            positive_idx = positive_class_index(encoder)
            quarterly_top_rate = float((val_df["_target_encoded"].values == positive_idx).mean())
        elif args.quarterly_top_rate is not None:
            if not 0 <= args.quarterly_top_rate <= 1:
                raise ValueError("--quarterly-top-rate must be between 0 and 1.")
            quarterly_top_rate = float(args.quarterly_top_rate)
        if args.quarterly_min_rate_from_validation:
            positive_idx = positive_class_index(encoder)
            quarterly_min_rate = float((val_df["_target_encoded"].values == positive_idx).mean())
        elif args.quarterly_min_rate is not None:
            if not 0 <= args.quarterly_min_rate <= 1:
                raise ValueError("--quarterly-min-rate must be between 0 and 1.")
            quarterly_min_rate = float(args.quarterly_min_rate)
    selected_model_name = "CalibratedBest" if args.calibrate_best else best_model_name
    selected_metrics = save_selected_model_metrics(
        prediction_model,
        selected_model_name,
        val_df,
        test_df,
        feature_cols,
        len(encoder.classes_),
        output_dir,
        decision_threshold=decision_threshold,
        threshold_metric=args.threshold_metric if args.tune_threshold else None,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    splits = {"train": train_df, "validation": val_df, "test": test_df}
    save_predictions(
        prediction_model,
        encoder,
        feature_cols,
        args.target,
        splits,
        output_dir,
        decision_threshold,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    save_latest_forecast(
        prediction_model,
        encoder,
        dataset,
        feature_cols,
        output_dir,
        decision_threshold,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )
    save_feature_importance(best_model, feature_cols, output_dir)
    save_confusion_matrices(
        prediction_model,
        encoder,
        feature_cols,
        splits,
        output_dir,
        decision_threshold,
        quarterly_top_rate=quarterly_top_rate,
        quarterly_min_rate=quarterly_min_rate,
    )

    summary = {
        "best_model": best_model_name,
        "task_type": "classification",
        "target_col": args.target,
        "classes": encoder.classes_.tolist(),
        "n_rows": int(dataset.shape[0]),
        "n_trainable_rows": int(trainable.shape[0]),
        "n_tickers": int(dataset["symbol"].nunique()),
        "n_features": len(feature_cols),
        "feature_set": args.feature_set,
        "feature_cols": feature_cols,
        "feature_pruning": feature_pruning,
        "symbol_dummies_enabled": not args.no_symbol_dummies,
        "symbol_dummy_count": len(symbol_dummy_cols),
        "calibrated_best": bool(args.calibrate_best),
        "calibration_method": args.calibration_method if args.calibrate_best else None,
        "calibrated_metrics": calibrated_metrics,
        "selected_metrics": selected_metrics,
        "decision_threshold": decision_threshold,
        "threshold_tuning_metric": args.threshold_metric if args.tune_threshold else None,
        "threshold_tuning_score": threshold_tuning_score,
        "fixed_threshold": args.fixed_threshold,
        "quarterly_top_rate": quarterly_top_rate,
        "quarterly_top_rate_from_validation": bool(args.quarterly_top_rate_from_validation),
        "quarterly_min_rate": quarterly_min_rate,
        "quarterly_min_rate_from_validation": bool(args.quarterly_min_rate_from_validation),
        "feature_selection": feature_selection_info,
        "panel_cycle_features_enabled": bool(args.add_panel_cycle_features or args.add_target_cycle_lag_features),
        "target_cycle_lag_features_enabled": bool(args.add_target_cycle_lag_features),
        "market_index_features_enabled": bool(args.add_market_index_features),
        "industry_cycle_features_enabled": bool(args.add_industry_cycle_features),
        "macro_regime_features_enabled": bool(args.add_macro_regime_features),
        "adaptive_target_info": adaptive_target_info,
        "weighted_voting_enabled": bool(args.include_weighted_voting),
        "voting_models": voting_model_names if args.include_weighted_voting else [],
        "voting_weight_metric": args.voting_weight_metric if args.include_weighted_voting else None,
        "selection_metric": args.selection_metric,
        "enabled_models": enabled_model_names,
        "split_info": split_info,
        "xgboost_available": XGBOOST_AVAILABLE,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(results)
    print(f"Best model: {best_model_name}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
