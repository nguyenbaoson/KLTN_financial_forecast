"""Two-stage SVR fusion classifier adapted from Patel et al. (2015).

Paper idea:
  Stage 1: SVR predicts future technical indicators.
  Stage 2: another model uses those predicted future indicators to predict
           the final future outcome.

Adaptation for this project:
  Stage 1: SVR predicts future quarterly state features at t+horizon
           (technical, market, news/event, transformer sentiment, and selected
           financial/bank indicators).
  Stage 2: classifiers predict target_profit_up_4q from current features plus
           Stage-1 predicted future-state features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer, select_feature_columns  # noqa: E402
from train.train_growth_classification import (  # noqa: E402
    apply_feature_set,
    classification_metrics,
    derive_classification_targets,
    positive_class_index,
    predict_proba_or_none,
    predict_with_threshold,
)
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-stage SVR fusion growth classifier.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-dir", default="outputs/thesis_two_stage_svr_fusion")
    parser.add_argument("--target", default="target_profit_up_4q")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--feature-set", choices=["all", "no_news", "news_only", "financial_only", "market_only"], default="all")
    parser.add_argument(
        "--stage1-target-set",
        choices=["paper_technical", "market_technical", "state_all"],
        default="state_all",
        help="Feature family that stage-1 SVRs predict. paper_technical is closest to Patel et al. (2015).",
    )
    parser.add_argument(
        "--stage2-input",
        choices=["current_plus_future", "future_only"],
        default="current_plus_future",
        help="Use only SVR-predicted future features, or concatenate current and future features.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--max-stage1-targets", type=int, default=45)
    parser.add_argument("--tune-stage1", action="store_true", help="Tune stage-1 SVR hyperparameters on validation split.")
    parser.add_argument("--stage1-kernels", default="rbf", help="Comma-separated SVR kernels, e.g. rbf,linear.")
    parser.add_argument("--stage1-c-values", default="1,10,100")
    parser.add_argument("--stage1-gamma-values", default="scale,0.01,0.1")
    parser.add_argument("--stage1-epsilon-values", default="0.01,0.1,0.5")
    parser.add_argument("--stage1-oof", action="store_true", help="Use expanding-window out-of-fold predictions for train stage-1 features.")
    parser.add_argument("--stage1-oof-min-periods", type=int, default=8)
    parser.add_argument(
        "--min-stage1-validation-r2",
        type=float,
        default=float("-inf"),
        help="Drop stage-1 SVR targets whose validation R2 is below this value. Use 0 to keep only useful future features.",
    )
    parser.add_argument(
        "--stage1-fallback-top-n",
        type=int,
        default=1,
        help="If filtering drops all stage-1 targets, keep the top N by validation R2.",
    )
    parser.add_argument("--no-symbol-dummies", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "recall"],
        default="balanced_accuracy",
        help="Metric used when tuning the binary decision threshold on validation.",
    )
    parser.add_argument(
        "--stage2-selection-metric",
        choices=["val_F1", "val_BalancedAccuracy", "val_AUC", "val_Accuracy"],
        default="val_F1",
        help="Validation metric used to select the final stage-2 classifier.",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, tickers: list[str] | None) -> pd.DataFrame:
    if args.dataset_path:
        df = pd.read_csv(args.dataset_path)
        if tickers:
            df = df[df["symbol"].astype(str).str.upper().isin(tickers)].copy()
        return derive_classification_targets(df)
    engineer = GrowthFeatureEngineer()
    return derive_classification_targets(engineer.build_dataset_from_ml_tree(data_root=args.data_root, tickers=tickers))


def encode_target(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    encoder = LabelEncoder()
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    train["_target_encoded"] = encoder.fit_transform(train[target_col].astype(str))
    val["_target_encoded"] = encoder.transform(val[target_col].astype(str))
    test["_target_encoded"] = encoder.transform(test[target_col].astype(str))
    return train, val, test, encoder


def is_stage1_target(col: str, target_set: str) -> bool:
    paper_technical = {
        "q_return",
        "price_volatility",
        "momentum_3m",
        "avg_rel_volume",
        "technical_sma_20_gap",
        "technical_sma_60_gap",
        "technical_rsi_14",
        "technical_macd",
        "technical_macd_signal",
        "technical_stochastic_k",
        "technical_williams_r",
        "technical_cci_20",
    }
    if target_set == "paper_technical":
        return col in paper_technical
    if target_set == "market_technical":
        return col in paper_technical or col in {"close_eoq", "avg_volume"}

    explicit = {
        "q_return",
        "price_volatility",
        "momentum_3m",
        "avg_rel_volume",
        "avg_sentiment",
        "event_weighted_impact_score_avg",
        "event_net_positive_ratio",
        "transformer_avg_sentiment",
        "transformer_negative_ratio",
        "transformer_positive_ratio",
        "transformer_avg_confidence",
        "gross_profit_margin_calc",
        "net_profit_margin_calc",
        "debt_to_asset_calc",
        "earnings_quality_calc",
    }
    return (
        col in explicit
        or col.startswith("technical_")
        or col.startswith("news_")
        or col.startswith("event_")
        or col.startswith("transformer_")
        or col.startswith("bank_")
    )


def choose_stage1_targets(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    max_targets: int,
    target_set: str,
) -> list[str]:
    candidates = [col for col in feature_cols if is_stage1_target(col, target_set)]
    rows = []
    sorted_df = df.sort_values(["symbol", "yq_index"]).copy()
    for col in candidates:
        future = sorted_df.groupby("symbol")[col].shift(-horizon)
        valid_ratio = future.notna().mean()
        variance = pd.to_numeric(future, errors="coerce").var()
        if valid_ratio >= 0.20 and pd.notna(variance) and variance > 1e-12:
            rows.append({"feature": col, "valid_ratio": valid_ratio, "variance": variance})
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return []
    ranked["score"] = ranked["valid_ratio"] * np.log1p(ranked["variance"].abs())
    return ranked.sort_values("score", ascending=False)["feature"].head(max_targets).tolist()


def make_stage1_training_frame(df: pd.DataFrame, feature_cols: list[str], target_feature: str, horizon: int) -> pd.DataFrame:
    work = df.sort_values(["symbol", "yq_index"]).copy()
    future_col = f"_future_{target_feature}"
    if future_col not in work.columns:
        work[future_col] = work.groupby("symbol")[target_feature].shift(-horizon)
    return work.dropna(subset=[f"_future_{target_feature}"])


def attach_future_stage1_targets(
    frame: pd.DataFrame,
    full_df: pd.DataFrame,
    stage1_targets: list[str],
    horizon: int,
) -> pd.DataFrame:
    result = frame.copy()
    keys = ["symbol", "yq_index"]
    future_source = full_df.sort_values(["symbol", "yq_index"])[keys + stage1_targets].copy()
    for feature in stage1_targets:
        future_source[f"_future_{feature}"] = future_source.groupby("symbol")[feature].shift(-horizon)
    future_cols = [f"_future_{feature}" for feature in stage1_targets]
    future_source = future_source[keys + future_cols]
    result = result.drop(columns=[col for col in future_cols if col in result.columns])
    return result.merge(future_source, on=keys, how="left")


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_gamma_grid(value: str) -> list[str | float]:
    result: list[str | float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"scale", "auto"}:
            result.append(item)
        else:
            result.append(float(item))
    return result


def stage1_param_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    kernels = [item.strip() for item in args.stage1_kernels.split(",") if item.strip()]
    c_values = parse_float_grid(args.stage1_c_values)
    gamma_values = parse_gamma_grid(args.stage1_gamma_values)
    epsilon_values = parse_float_grid(args.stage1_epsilon_values)
    return [
        {"kernel": kernel, "C": c_value, "gamma": gamma, "epsilon": epsilon}
        for kernel, c_value, gamma, epsilon in product(kernels, c_values, gamma_values, epsilon_values)
    ]


def build_stage1_svr(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("svr", SVR(
            kernel=params["kernel"],
            C=float(params["C"]),
            gamma=params["gamma"],
            epsilon=float(params["epsilon"]),
        )),
    ])


def fit_stage1_svrs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    stage1_targets: list[str],
    horizon: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, dict[str, Any]]]:
    models: dict[str, Pipeline] = {}
    tuning_rows: list[dict[str, Any]] = []
    selected_params: dict[str, dict[str, Any]] = {}
    grid = stage1_param_grid(args)
    for target_feature in stage1_targets:
        frame = make_stage1_training_frame(train_df, feature_cols, target_feature, horizon)
        if len(frame) < 50:
            continue
        logger.info("Stage 1 SVR predicts future %s", target_feature)
        best_params = {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1}
        best_score = np.inf
        if args.tune_stage1:
            val_frame = make_stage1_training_frame(val_df, feature_cols, target_feature, horizon)
            for params in grid:
                model = build_stage1_svr(params)
                model.fit(frame[feature_cols], frame[f"_future_{target_feature}"].values)
                if val_frame.empty:
                    y_true = frame[f"_future_{target_feature}"].values
                    y_pred = model.predict(frame[feature_cols])
                    split_name = "train"
                else:
                    y_true = val_frame[f"_future_{target_feature}"].values
                    y_pred = model.predict(val_frame[feature_cols])
                    split_name = "validation"
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                tuning_rows.append({
                    "feature": target_feature,
                    "split": split_name,
                    "RMSE": rmse,
                    **params,
                })
                if rmse < best_score:
                    best_score = rmse
                    best_params = params
        model = build_stage1_svr(best_params)
        model.fit(frame[feature_cols], frame[f"_future_{target_feature}"].values)
        models[target_feature] = model
        selected_params[target_feature] = dict(best_params)
        joblib.dump(model, output_dir / f"stage1_svr_future_{target_feature}.joblib")
    return models, pd.DataFrame(tuning_rows), selected_params


def append_stage1_predictions(df: pd.DataFrame, feature_cols: list[str], stage1_models: dict[str, Pipeline]) -> pd.DataFrame:
    result = df.copy()
    for feature, model in stage1_models.items():
        result[f"svr_future_{feature}"] = model.predict(result[feature_cols])
    return result


def append_stage1_oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    stage1_models: dict[str, Pipeline],
    stage1_params: dict[str, dict[str, Any]],
    horizon: int,
    min_periods: int,
) -> pd.DataFrame:
    result = train_df.copy()
    periods = sorted(result["yq_index"].dropna().astype(int).unique().tolist())
    for feature, final_model in stage1_models.items():
        col = f"svr_future_{feature}"
        result[col] = np.nan
        params = stage1_params.get(feature, {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1})
        for idx, period in enumerate(periods):
            if idx < min_periods:
                continue
            history = result[result["yq_index"].astype(int) < period].copy()
            fold_frame = make_stage1_training_frame(history, feature_cols, feature, horizon)
            predict_mask = result["yq_index"].astype(int).eq(period)
            if len(fold_frame) < 50 or not predict_mask.any():
                continue
            model = build_stage1_svr(params)
            model.fit(fold_frame[feature_cols], fold_frame[f"_future_{feature}"].values)
            result.loc[predict_mask, col] = model.predict(result.loc[predict_mask, feature_cols])
        missing = result[col].isna()
        if missing.any():
            result.loc[missing, col] = final_model.predict(result.loc[missing, feature_cols])
    return result


def stage1_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def evaluate_stage1_svrs(
    models: dict[str, Pipeline],
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, model in models.items():
        for split_name, frame in splits.items():
            eval_frame = make_stage1_training_frame(frame, feature_cols, feature, horizon)
            if eval_frame.empty:
                continue
            y_true = eval_frame[f"_future_{feature}"].values
            y_pred = model.predict(eval_frame[feature_cols])
            rows.append({
                "feature": feature,
                "split": split_name,
                "n_rows": int(len(eval_frame)),
                **stage1_regression_metrics(y_true, y_pred),
            })
    return pd.DataFrame(rows)


def filter_stage1_models(
    models: dict[str, Pipeline],
    selected_params: dict[str, dict[str, Any]],
    stage1_eval: pd.DataFrame,
    min_validation_r2: float,
    fallback_top_n: int,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]], list[str], pd.DataFrame]:
    if not np.isfinite(min_validation_r2) or stage1_eval.empty:
        return models, selected_params, [], stage1_eval

    validation = stage1_eval[stage1_eval["split"].eq("validation")].copy()
    if validation.empty or "R2" not in validation.columns:
        return models, selected_params, [], stage1_eval

    validation = validation.sort_values("R2", ascending=False)
    keep = validation[validation["R2"].ge(min_validation_r2)]["feature"].astype(str).tolist()
    dropped = sorted(set(models) - set(keep))
    if not keep and fallback_top_n > 0:
        keep = validation["feature"].astype(str).head(fallback_top_n).tolist()
        dropped = sorted(set(models) - set(keep))

    filtered_models = {feature: models[feature] for feature in keep if feature in models}
    filtered_params = {feature: selected_params[feature] for feature in keep if feature in selected_params}
    stage1_eval = stage1_eval.copy()
    stage1_eval["selected_for_stage2"] = stage1_eval["feature"].isin(filtered_models)
    return filtered_models, filtered_params, dropped, stage1_eval


def build_fusion_improvement(
    single_results: pd.DataFrame,
    fusion_results: pd.DataFrame,
    single_best: str,
    fusion_best: str,
) -> pd.DataFrame:
    single_row = single_results[single_results["model"].eq(single_best)].iloc[0]
    fusion_row = fusion_results[fusion_results["model"].eq(fusion_best)].iloc[0]
    metrics = [
        "val_Accuracy",
        "val_BalancedAccuracy",
        "val_F1",
        "val_AUC",
        "test_Accuracy",
        "test_BalancedAccuracy",
        "test_F1",
        "test_AUC",
    ]
    rows = []
    for metric in metrics:
        if metric not in single_row.index or metric not in fusion_row.index:
            continue
        single_value = float(single_row[metric])
        fusion_value = float(fusion_row[metric])
        rows.append({
            "metric": metric,
            "single_stage_best_model": single_best,
            "single_stage": single_value,
            "fusion_best_model": fusion_best,
            "two_stage_fusion": fusion_value,
            "delta": fusion_value - single_value,
        })
    return pd.DataFrame(rows)


def tune_binary_threshold_by_metric(
    model: Pipeline,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    positive_idx: int,
    metric: str,
) -> tuple[float, float]:
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
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def get_stage2_models() -> Dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=800,
                max_depth=8,
                min_samples_leaf=2,
                max_features=0.75,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=260,
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
                alpha=0.01,
                max_iter=700,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }


def evaluate_stage2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    encoder: LabelEncoder,
    output_dir: Path,
    model_prefix: str = "stage2",
    selection_metric: str = "val_F1",
) -> tuple[pd.DataFrame, str, Pipeline]:
    models = get_stage2_models()
    rows: List[Dict[str, Any]] = []
    n_classes = len(encoder.classes_)
    for name, model in models.items():
        logger.info("Stage 2 train %s", name)
        model.fit(train_df[feature_cols], train_df["_target_encoded"].values)
        val_pred = model.predict(val_df[feature_cols])
        test_pred = model.predict(test_df[feature_cols])
        rows.append({
            "model": name,
            **{f"val_{k}": v for k, v in classification_metrics(
                val_df["_target_encoded"].values,
                val_pred,
                predict_proba_or_none(model, val_df[feature_cols]),
                n_classes,
            ).items()},
            **{f"test_{k}": v for k, v in classification_metrics(
                test_df["_target_encoded"].values,
                test_pred,
                predict_proba_or_none(model, test_df[feature_cols]),
                n_classes,
            ).items()},
        })
        joblib.dump(model, output_dir / f"{model_prefix}_{name.lower()}.joblib")
    sort_cols = [selection_metric, "val_F1", "val_BalancedAccuracy", "val_Accuracy"]
    sort_cols = list(dict.fromkeys(col for col in sort_cols if col in rows[0]))
    results = pd.DataFrame(rows).sort_values(sort_cols, ascending=False).reset_index(drop=True)
    best_name = str(results.iloc[0]["model"])
    best_model = joblib.load(output_dir / f"{model_prefix}_{best_name.lower()}.joblib")
    return results, best_name, best_model


def save_predictions(
    model: Pipeline,
    encoder: LabelEncoder,
    feature_cols: list[str],
    target_col: str,
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    decision_threshold: float | None,
) -> None:
    frames = []
    positive_idx = positive_class_index(encoder)
    for split, frame in splits.items():
        pred = predict_with_threshold(model, frame[feature_cols], decision_threshold, positive_idx)
        proba = predict_proba_or_none(model, frame[feature_cols])
        out = frame[["symbol", "year", "quarter", "yq_index", target_col]].copy()
        out["split"] = split
        out["actual_label"] = frame[target_col].astype(str)
        out["predicted_label"] = encoder.inverse_transform(pred.astype(int))
        if decision_threshold is not None:
            out["decision_threshold"] = decision_threshold
        if proba is not None:
            for idx, label in enumerate(encoder.classes_):
                out[f"prob_{label}"] = proba[:, idx]
            out["prediction_confidence"] = proba.max(axis=1)
        frames.append(out)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()] or None

    dataset = load_dataset(args, tickers).dropna(axis=1, how="all")
    if args.target not in dataset.columns:
        raise ValueError(f"Missing target: {args.target}")
    trainable = dataset.dropna(subset=[args.target]).copy()
    trainable[args.target] = trainable[args.target].astype(str)
    if trainable[args.target].nunique() < 2:
        raise ValueError("Target has fewer than 2 classes.")

    symbol_dummy_cols: list[str] = []
    if not args.no_symbol_dummies:
        symbol_values = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique().tolist())
        dataset, symbol_dummy_cols = add_symbol_dummies(dataset, symbol_values)
        trainable, _ = add_symbol_dummies(trainable, symbol_values)

    base_feature_cols = select_feature_columns(trainable, args.target, min_non_null_ratio=args.min_feature_non_null)
    base_feature_cols = apply_feature_set(base_feature_cols, args.feature_set)
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    base_feature_cols, base_pruning = prune_feature_columns(train_df, base_feature_cols, max_corr=args.max_feature_corr)

    stage1_targets = choose_stage1_targets(
        train_df,
        base_feature_cols,
        args.horizon,
        args.max_stage1_targets,
        args.stage1_target_set,
    )
    train_df = attach_future_stage1_targets(train_df, trainable, stage1_targets, args.horizon)
    val_df = attach_future_stage1_targets(val_df, trainable, stage1_targets, args.horizon)
    test_df = attach_future_stage1_targets(test_df, trainable, stage1_targets, args.horizon)
    stage1_models, stage1_tuning, selected_stage1_params = fit_stage1_svrs(
        train_df,
        val_df,
        base_feature_cols,
        stage1_targets,
        args.horizon,
        output_dir,
        args,
    )
    if not stage1_models:
        raise ValueError("No usable stage-1 SVR models were trained.")
    if not stage1_tuning.empty:
        stage1_tuning.to_csv(output_dir / "stage1_svr_tuning.csv", index=False)
    stage1_eval = evaluate_stage1_svrs(
        stage1_models,
        {"train": train_df, "validation": val_df, "test": test_df},
        base_feature_cols,
        args.horizon,
    )
    stage1_models, selected_stage1_params, dropped_stage1_targets, stage1_eval = filter_stage1_models(
        stage1_models,
        selected_stage1_params,
        stage1_eval,
        args.min_stage1_validation_r2,
        args.stage1_fallback_top_n,
    )
    if not stage1_models:
        raise ValueError("No stage-1 SVR targets survived filtering.")
    if not stage1_eval.empty:
        stage1_eval.to_csv(output_dir / "stage1_svr_results.csv", index=False)

    train_aug = (
        append_stage1_oof_predictions(
            train_df,
            base_feature_cols,
            stage1_models,
            selected_stage1_params,
            args.horizon,
            args.stage1_oof_min_periods,
        )
        if args.stage1_oof
        else append_stage1_predictions(train_df, base_feature_cols, stage1_models)
    )
    val_aug = append_stage1_predictions(val_df, base_feature_cols, stage1_models)
    test_aug = append_stage1_predictions(test_df, base_feature_cols, stage1_models)
    fusion_cols = [f"svr_future_{feature}" for feature in stage1_models]
    stage2_feature_cols = fusion_cols if args.stage2_input == "future_only" else base_feature_cols + fusion_cols

    train_aug, val_aug, test_aug, encoder = encode_target(train_aug, val_aug, test_aug, args.target)
    single_results, single_best_name, _ = evaluate_stage2(
        train_aug,
        val_aug,
        test_aug,
        base_feature_cols,
        encoder,
        output_dir,
        model_prefix="single_stage",
        selection_metric=args.stage2_selection_metric,
    )
    single_results.to_csv(output_dir / "single_stage_model_results.csv", index=False)

    results, best_name, best_model = evaluate_stage2(
        train_aug,
        val_aug,
        test_aug,
        stage2_feature_cols,
        encoder,
        output_dir,
        model_prefix="stage2_fusion",
        selection_metric=args.stage2_selection_metric,
    )
    improvement = build_fusion_improvement(single_results, results, single_best_name, best_name)
    improvement.to_csv(output_dir / "fusion_improvement.csv", index=False)
    decision_threshold = None
    threshold_score = None
    if args.tune_threshold and len(encoder.classes_) == 2:
        decision_threshold, threshold_score = tune_binary_threshold_by_metric(
            best_model,
            val_aug,
            stage2_feature_cols,
            positive_class_index(encoder),
            args.threshold_metric,
        )

    results.to_csv(output_dir / "model_results.csv", index=False)
    save_predictions(
        best_model,
        encoder,
        stage2_feature_cols,
        args.target,
        {"train": train_aug, "validation": val_aug, "test": test_aug},
        output_dir,
        decision_threshold,
    )
    train_aug.to_csv(output_dir / "train_augmented.csv", index=False)
    val_aug.to_csv(output_dir / "validation_augmented.csv", index=False)
    test_aug.to_csv(output_dir / "test_augmented.csv", index=False)

    summary = {
        "method": "two_stage_svr_fusion_adapted_from_patel_2015",
        "target_col": args.target,
        "horizon": args.horizon,
        "feature_set": args.feature_set,
        "stage1_target_set": args.stage1_target_set,
        "stage2_input": args.stage2_input,
        "stage1_tuned": bool(args.tune_stage1),
        "stage1_oof": bool(args.stage1_oof),
        "stage1_oof_min_periods": args.stage1_oof_min_periods,
        "min_stage1_validation_r2": args.min_stage1_validation_r2 if np.isfinite(args.min_stage1_validation_r2) else None,
        "dropped_stage1_targets": dropped_stage1_targets,
        "selected_stage1_params": selected_stage1_params,
        "stage2_selection_metric": args.stage2_selection_metric,
        "single_stage_best_model": single_best_name,
        "best_model": best_name,
        "classes": encoder.classes_.tolist(),
        "n_rows": int(dataset.shape[0]),
        "n_trainable_rows": int(trainable.shape[0]),
        "n_tickers": int(trainable["symbol"].nunique()),
        "n_base_features": len(base_feature_cols),
        "n_stage1_targets": len(stage1_models),
        "n_stage2_features": len(stage2_feature_cols),
        "stage1_targets": list(stage1_models.keys()),
        "single_stage_results_path": str((output_dir / "single_stage_model_results.csv").as_posix()),
        "stage1_results_path": str((output_dir / "stage1_svr_results.csv").as_posix()) if not stage1_eval.empty else None,
        "stage1_tuning_path": str((output_dir / "stage1_svr_tuning.csv").as_posix()) if not stage1_tuning.empty else None,
        "fusion_improvement_path": str((output_dir / "fusion_improvement.csv").as_posix()),
        "symbol_dummies_enabled": not args.no_symbol_dummies,
        "symbol_dummy_count": len(symbol_dummy_cols),
        "base_feature_pruning": base_pruning,
        "decision_threshold": decision_threshold,
        "threshold_tuning_metric": args.threshold_metric if args.tune_threshold else None,
        "threshold_tuning_score": threshold_score,
        "split_info": split_info,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(results)
    print(f"Best model: {best_name}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
