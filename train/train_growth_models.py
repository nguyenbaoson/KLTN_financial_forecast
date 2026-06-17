"""
Train growth forecasting models for Vietnamese listed companies.

Examples:
  python -m train.train_growth_models --target target_profit_growth_1q
  python -m train.train_growth_models --target target_revenue_growth_4q --tickers FPT,VNM,HPG
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer, select_feature_columns


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = Path("outputs/growth_forecast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train listed-company growth forecasting models.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-layout",
        choices=["legacy", "ml_tree"],
        default="ml_tree",
        help="legacy reads a symbol-folder raw root; ml_tree reads data/raw/fundamental + data/raw/market.",
    )
    parser.add_argument("--macro-path", default="data/raw/macro/macro_data.csv")
    parser.add_argument("--news-path", default="data/processed/news_merged.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default="target_profit_growth_1q")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers. Empty means all available folders.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument(
        "--max-feature-corr",
        type=float,
        default=0.98,
        help="Drop later numeric features whose absolute train correlation exceeds this threshold. Use 0 to disable.",
    )
    parser.add_argument(
        "--target-clip",
        type=float,
        default=300.0,
        help="Clip regression target to +/- this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-symbol-dummies",
        action="store_true",
        help="Disable per-symbol dummy features. Enabled by default for known listed companies.",
    )
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    periods = sorted(df["yq_index"].dropna().astype(int).unique().tolist())
    if len(periods) < 6:
        raise ValueError("Need at least 6 quarterly periods to create train/validation/test splits.")

    train_end_idx = max(1, int(len(periods) * train_ratio)) - 1
    val_end_idx = max(train_end_idx + 1, int(len(periods) * (train_ratio + val_ratio))) - 1
    val_end_idx = min(val_end_idx, len(periods) - 2)

    train_end_yq = periods[train_end_idx]
    val_end_yq = periods[val_end_idx]

    train_df = df[df["yq_index"] <= train_end_yq].copy()
    val_df = df[(df["yq_index"] > train_end_yq) & (df["yq_index"] <= val_end_yq)].copy()
    test_df = df[df["yq_index"] > val_end_yq].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train, validation, or test set.")

    split_info = {
        "periods": periods,
        "train_end_yq": int(train_end_yq),
        "val_end_yq": int(val_end_yq),
        "test_start_yq": int(test_df["yq_index"].min()),
    }
    return train_df, val_df, test_df, split_info


def prune_feature_columns(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    max_corr: float = 0.98,
) -> Tuple[List[str], Dict[str, Any]]:
    usable = []
    dropped_constant = []
    for col in feature_cols:
        if col not in train_df.columns or train_df[col].isna().all():
            dropped_constant.append(col)
            continue
        if train_df[col].nunique(dropna=True) <= 1:
            dropped_constant.append(col)
            continue
        usable.append(col)

    dropped_corr: List[str] = []
    if max_corr and max_corr > 0 and len(usable) > 1:
        numeric = train_df[usable].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            high_corr = upper[col][upper[col] > max_corr]
            if high_corr.empty:
                continue
            dropped_corr.append(col)
        usable = [col for col in usable if col not in set(dropped_corr)]

    info = {
        "dropped_constant_features": dropped_constant,
        "dropped_correlated_features": dropped_corr,
        "max_feature_corr": max_corr,
    }
    return usable, info


def add_symbol_dummies(df: pd.DataFrame, symbols: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    if "symbol" not in df.columns:
        return df, []
    out = df.copy()
    symbols = symbols or sorted(out["symbol"].dropna().astype(str).str.upper().unique().tolist())
    dummy_cols = []
    symbol_values = out["symbol"].astype(str).str.upper()
    for symbol in symbols:
        col = f"symbol_dummy_{symbol}"
        out[col] = symbol_values.eq(symbol).astype(float)
        dummy_cols.append(col)
    return out, dummy_cols


def get_models() -> Dict[str, Pipeline]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    models: Dict[str, Pipeline] = {
        "Ridge": Pipeline([
            *base_steps,
            ("model", Ridge(alpha=1_000_000.0, solver="lsqr")),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=900,
                max_depth=8,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=220,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.75,
                colsample_bytree=0.70,
                min_child_weight=5,
                reg_alpha=0.5,
                reg_lambda=5.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            )),
        ])
        models["XGBoostFlexible"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=550,
                max_depth=3,
                learning_rate=0.025,
                subsample=0.90,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_alpha=0.05,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            )),
        ])
    return models


def fit_and_evaluate(
    models: Dict[str, Pipeline],
    feature_cols: List[str],
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_val, y_val = val_df[feature_cols], val_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    for model_name, pipeline in models.items():
        logger.info("Training %s", model_name)
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)
        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        rows.append({
            "model": model_name,
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_R2": val_metrics["R2"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_R2": test_metrics["R2"],
        })
        joblib.dump(pipeline, output_dir / f"{model_name.lower()}_growth_model.joblib")

    return pd.DataFrame(rows).sort_values(["val_RMSE", "val_MAE"]).reset_index(drop=True)


def save_predictions(
    model: Pipeline,
    feature_cols: List[str],
    target_col: str,
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    frames: List[pd.DataFrame] = []
    for split_name, split_df in splits.items():
        pred_df = split_df[["symbol", "year", "quarter", "yq_index", target_col]].copy()
        pred_df["split"] = split_name
        pred_df["predicted_growth_pct"] = model.predict(split_df[feature_cols])
        pred_df = pred_df.rename(columns={target_col: "actual_growth_pct"})
        pred_df["growth_signal"] = pd.cut(
            pred_df["predicted_growth_pct"],
            bins=[-np.inf, 0, 10, np.inf],
            labels=["low", "medium", "high"],
        )
        pred_df["early_warning"] = pred_df["predicted_growth_pct"] < 0
        frames.append(pred_df)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


def save_latest_forecast(
    model: Pipeline,
    dataset: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    output_dir: Path,
) -> None:
    latest_idx = dataset.sort_values(["symbol", "yq_index"]).groupby("symbol").tail(1).index
    latest = dataset.loc[latest_idx].copy()
    latest["predicted_growth_pct"] = model.predict(latest[feature_cols])
    latest["growth_signal"] = pd.cut(
        latest["predicted_growth_pct"],
        bins=[-np.inf, 0, 10, np.inf],
        labels=["low", "medium", "high"],
    )
    latest["early_warning"] = latest["predicted_growth_pct"] < 0
    cols = ["symbol", "year", "quarter", "yq_index", "predicted_growth_pct", "growth_signal", "early_warning"]
    if target_col in latest.columns:
        cols.append(target_col)
    latest[cols].sort_values("predicted_growth_pct", ascending=False).to_csv(
        output_dir / "latest_forecast.csv",
        index=False,
    )


def save_feature_importance(model: Pipeline, feature_cols: List[str], output_dir: Path) -> None:
    estimator = model.named_steps.get("model")
    if not hasattr(estimator, "feature_importances_"):
        return
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": estimator.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()] or None
    engineer = GrowthFeatureEngineer()
    if args.data_layout == "ml_tree":
        dataset = engineer.build_dataset_from_ml_tree(
            data_root=args.data_root,
            tickers=tickers,
        )
    else:
        dataset = engineer.build_dataset(
            raw_root=args.raw_root,
            macro_path=args.macro_path,
            news_path=args.news_path,
            tickers=tickers,
        )
    if dataset.empty:
        raise ValueError("Feature dataset is empty. Check raw_root and ticker inputs.")
    if args.target not in dataset.columns:
        raise ValueError(f"Target {args.target!r} does not exist in dataset.")
    dataset = dataset.dropna(axis=1, how="all")

    trainable = dataset.dropna(subset=[args.target]).copy()
    if args.target_clip and args.target_clip > 0:
        trainable[args.target] = trainable[args.target].clip(-args.target_clip, args.target_clip)
    symbol_dummy_cols: List[str] = []
    if not args.no_symbol_dummies:
        symbol_values = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique().tolist())
        dataset, symbol_dummy_cols = add_symbol_dummies(dataset, symbol_values)
        trainable, _ = add_symbol_dummies(trainable, symbol_values)

    dataset.to_csv(output_dir / "growth_dataset.csv", index=False)

    feature_cols = select_feature_columns(trainable, args.target, min_non_null_ratio=args.min_feature_non_null)
    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable features: {len(feature_cols)}")

    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, feature_pruning = prune_feature_columns(
        train_df,
        feature_cols,
        max_corr=args.max_feature_corr,
    )
    if len(feature_cols) < 5:
        raise ValueError(f"Too few usable train features after removing all-null columns: {len(feature_cols)}")

    models = get_models()
    results = fit_and_evaluate(models, feature_cols, args.target, train_df, val_df, test_df, output_dir)
    results.to_csv(output_dir / "model_results.csv", index=False)

    best_model_name = str(results.iloc[0]["model"])
    best_model = joblib.load(output_dir / f"{best_model_name.lower()}_growth_model.joblib")
    save_predictions(
        best_model,
        feature_cols,
        args.target,
        {"train": train_df, "validation": val_df, "test": test_df},
        output_dir,
    )
    save_latest_forecast(best_model, dataset, feature_cols, args.target, output_dir)
    save_feature_importance(best_model, feature_cols, output_dir)

    summary = {
        "best_model": best_model_name,
        "target_col": args.target,
        "n_rows": int(dataset.shape[0]),
        "n_tickers": int(dataset["symbol"].nunique()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "feature_pruning": feature_pruning,
        "target_clip": args.target_clip,
        "symbol_dummies_enabled": not args.no_symbol_dummies,
        "symbol_dummy_count": len(symbol_dummy_cols),
        "split_info": split_info,
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(results)
    print(f"Best model: {best_model_name}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
