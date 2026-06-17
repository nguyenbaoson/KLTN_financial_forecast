from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from train.train_growth_classification import WeightedSoftVotingClassifier


CLASSIFIER_ARTIFACTS = {
    "RandomForest": "randomforest_classifier.joblib",
    "XGBoost": "xgboost_classifier.joblib",
    "LightGBM": "lightgbm_classifier.joblib",
    "HistGradientBoosting": "histgradientboosting_classifier.joblib",
    "GradientBoosting": "gradientboosting_classifier.joblib",
    "WeightedSoftVoting": "weightedsoftvoting_classifier.joblib",
    "ExtraTrees": "extratrees_classifier.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adaptive target configs across selected models.")
    parser.add_argument("--s70-dir", default="outputs/adaptive_compare_models_1q/s70_q20_p20")
    parser.add_argument("--s80-dir", default="outputs/adaptive_compare_models_1q/s80_q10_p10")
    parser.add_argument("--output-dir", default="outputs/adaptive_compare_models_1q/comparison")
    parser.add_argument("--growth-target", default="target_profit_growth_1q")
    return parser.parse_args()


def positive_probability(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    proba = model.predict_proba(frame[feature_cols])
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1]))).astype(str)
    if "1.0" in classes:
        positive_idx = int(np.where(classes == "1.0")[0][0])
    elif "1" in classes:
        positive_idx = int(np.where(classes == "1")[0][0])
    else:
        positive_idx = min(1, proba.shape[1] - 1)
    return np.asarray(proba[:, positive_idx], dtype=float)


def metric_dict(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out = {
        "rows": int(len(y_true)),
        "actual_positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else float("nan"),
        "Accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else float("nan"),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else float("nan"),
        "Precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "Recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "F1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
    }
    try:
        out["AUC"] = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")
    except ValueError:
        out["AUC"] = float("nan")
    return out


def tune_threshold(y_true: np.ndarray, score: np.ndarray, higher_is_positive: bool = True) -> tuple[float, float]:
    best_threshold = 0.5 if higher_is_positive else 0.0
    best_score = -1.0
    lo, hi = np.nanpercentile(score, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        grid = np.array([best_threshold])
    else:
        grid = np.linspace(float(lo), float(hi), 101)
    for threshold in grid:
        pred = (score >= threshold).astype(int) if higher_is_positive else (score <= threshold).astype(int)
        current = f1_score(y_true, pred, zero_division=0)
        if current > best_score:
            best_score = float(current)
            best_threshold = float(threshold)
    return best_threshold, best_score


def split_frames(dataset: pd.DataFrame, summary: dict[str, Any], target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainable = dataset.dropna(subset=[target_col]).copy()
    trainable[target_col] = pd.to_numeric(trainable[target_col], errors="coerce").astype(int)
    train_end = int(summary["split_info"]["train_end_yq"])
    val_end = int(summary["split_info"]["val_end_yq"])
    train = trainable[trainable["yq_index"] <= train_end].copy()
    val = trainable[(trainable["yq_index"] > train_end) & (trainable["yq_index"] <= val_end)].copy()
    test = trainable[trainable["yq_index"] > val_end].copy()
    return train, val, test


def add_common_columns(frame: pd.DataFrame, config: str, model: str, target_col: str, score: np.ndarray, pred: np.ndarray, threshold: float, family: str) -> pd.DataFrame:
    out = frame.copy()
    out["config"] = config
    out["model"] = model
    out["family"] = family
    out["actual"] = out[target_col].astype(int)
    out["score"] = score
    out["predicted"] = pred.astype(int)
    out["correct"] = (out["actual"].to_numpy() == pred.astype(int)).astype(int)
    out["threshold"] = threshold
    out["quarter_label"] = out["year"].astype(int).astype(str) + "Q" + out["quarter"].astype(int).astype(str)
    return out


def classifier_predictions(config: str, input_dir: Path, dataset: pd.DataFrame, summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    target_col = summary["target_col"]
    feature_cols = summary["feature_cols"]
    _train, val, test = split_frames(dataset, summary, target_col)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for model_name, artifact in CLASSIFIER_ARTIFACTS.items():
        artifact_path = input_dir / artifact
        if not artifact_path.exists():
            continue
        model = joblib.load(artifact_path)
        val_score = positive_probability(model, val, feature_cols)
        threshold, val_f1 = tune_threshold(val[target_col].to_numpy(), val_score)
        test_score = positive_probability(model, test, feature_cols)
        test_pred = (test_score >= threshold).astype(int)
        test_metrics = metric_dict(test[target_col].to_numpy(), test_pred, test_score)
        rows.append({
            "config": config,
            "model": model_name,
            "family": "Direct classifier",
            "threshold": threshold,
            "val_tuned_F1": val_f1,
            **test_metrics,
        })
        predictions.append(add_common_columns(test, config, model_name, target_col, test_score, test_pred, threshold, "Direct classifier"))
    return rows, predictions


def adaptive_quality_mask(frame: pd.DataFrame) -> np.ndarray:
    parts = []
    if "_adaptive_roa_quality" in frame and "_adaptive_roa_floor" in frame:
        parts.append(pd.to_numeric(frame["_adaptive_roa_quality"], errors="coerce") >= pd.to_numeric(frame["_adaptive_roa_floor"], errors="coerce"))
    if "_adaptive_roe_quality" in frame and "_adaptive_roe_floor" in frame:
        parts.append(pd.to_numeric(frame["_adaptive_roe_quality"], errors="coerce") >= pd.to_numeric(frame["_adaptive_roe_floor"], errors="coerce"))
    if parts:
        quality_ok = parts[0]
        for part in parts[1:]:
            quality_ok = quality_ok | part
    else:
        quality_ok = pd.Series(True, index=frame.index)
    profit_col = "net_profit_ttm" if "net_profit_ttm" in frame else "net_profit_ttm_calc" if "net_profit_ttm_calc" in frame else ""
    if "_adaptive_profit_ttm_floor" in frame and profit_col:
        profit_ok = pd.to_numeric(frame[profit_col], errors="coerce") >= pd.to_numeric(frame["_adaptive_profit_ttm_floor"], errors="coerce")
    else:
        profit_ok = pd.Series(True, index=frame.index)
    return (quality_ok.fillna(False) & profit_ok.fillna(False)).to_numpy()


def growth_to_score(frame: pd.DataFrame, pred_growth: np.ndarray) -> np.ndarray:
    threshold = pd.to_numeric(frame["_adaptive_growth_threshold"], errors="coerce").to_numpy(dtype=float)
    return np.asarray(pred_growth, dtype=float) - threshold


def regression_pred_from_score(frame: pd.DataFrame, score: np.ndarray, margin_threshold: float) -> np.ndarray:
    return ((score >= margin_threshold) & adaptive_quality_mask(frame)).astype(int)


def ridge_predictions(config: str, dataset: pd.DataFrame, summary: dict[str, Any], growth_target: str) -> tuple[dict[str, Any], pd.DataFrame]:
    target_col = summary["target_col"]
    feature_cols = summary["feature_cols"]
    train, val, test = split_frames(dataset, summary, target_col)
    train_reg = train.dropna(subset=[growth_target]).copy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", Ridge(alpha=100.0)),
    ])
    model.fit(train_reg[feature_cols], pd.to_numeric(train_reg[growth_target], errors="coerce").to_numpy())
    val_growth = model.predict(val[feature_cols])
    val_score = growth_to_score(val, val_growth)
    threshold, val_f1 = tune_threshold(val[target_col].to_numpy(), val_score)
    test_growth = model.predict(test[feature_cols])
    test_score = growth_to_score(test, test_growth)
    test_pred = regression_pred_from_score(test, test_score, threshold)
    row = {
        "config": config,
        "model": "Ridge",
        "family": "Growth regression",
        "threshold": threshold,
        "val_tuned_F1": val_f1,
        **metric_dict(test[target_col].to_numpy(), test_pred, test_score),
    }
    pred_frame = add_common_columns(test, config, "Ridge", target_col, test_score, test_pred, threshold, "Growth regression")
    pred_frame["predicted_growth_1q"] = test_growth
    return row, pred_frame


def _fit_arima_forecast(series: pd.Series, steps: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    fallback = float(np.nanmean(values)) if len(values) else 0.0
    if len(values) < 8 or steps == 0:
        return np.repeat(fallback, steps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                values,
                order=(1, 0, 0),
                seasonal_order=(0, 0, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=80)
        forecast = np.asarray(model.forecast(steps=steps), dtype=float)
        return np.nan_to_num(forecast, nan=fallback, posinf=fallback, neginf=fallback)
    except Exception:
        return np.repeat(fallback, steps)


def arima_predictions(config: str, dataset: pd.DataFrame, summary: dict[str, Any], growth_target: str) -> tuple[dict[str, Any], pd.DataFrame]:
    target_col = summary["target_col"]
    train, val, test = split_frames(dataset, summary, target_col)
    val_preds = []
    test_preds = []
    for symbol, train_symbol in train.groupby("symbol", dropna=False):
        val_symbol = val[val["symbol"] == symbol].sort_values("yq_index")
        test_symbol = test[test["symbol"] == symbol].sort_values("yq_index")
        train_series = train_symbol.sort_values("yq_index")[growth_target]
        val_forecast = _fit_arima_forecast(train_series, len(val_symbol))
        val_preds.append(pd.Series(val_forecast, index=val_symbol.index))
        extended_series = pd.concat([train_series, val_symbol[growth_target]], ignore_index=True)
        test_forecast = _fit_arima_forecast(extended_series, len(test_symbol))
        test_preds.append(pd.Series(test_forecast, index=test_symbol.index))
    val_growth = pd.concat(val_preds).reindex(val.index).to_numpy(dtype=float)
    test_growth = pd.concat(test_preds).reindex(test.index).to_numpy(dtype=float)
    val_score = growth_to_score(val, val_growth)
    threshold, val_f1 = tune_threshold(val[target_col].to_numpy(), val_score)
    test_score = growth_to_score(test, test_growth)
    test_pred = regression_pred_from_score(test, test_score, threshold)
    row = {
        "config": config,
        "model": "ARIMA",
        "family": "Time-series growth regression",
        "threshold": threshold,
        "val_tuned_F1": val_f1,
        **metric_dict(test[target_col].to_numpy(), test_pred, test_score),
    }
    pred_frame = add_common_columns(test, config, "ARIMA", target_col, test_score, test_pred, threshold, "Time-series growth regression")
    pred_frame["predicted_growth_1q"] = test_growth
    return row, pred_frame


def load_config(config: str, input_dir: Path, growth_target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(input_dir / "training_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset = pd.read_csv(input_dir / "classification_dataset.csv")
    thresholds_path = input_dir / "adaptive_target_thresholds.csv"
    if thresholds_path.exists() and "_adaptive_group" in dataset.columns:
        thresholds = pd.read_csv(thresholds_path).rename(columns={
            "group": "_adaptive_group",
            "growth_threshold": "_adaptive_growth_threshold",
            "roa_floor": "_adaptive_roa_floor",
            "roe_floor": "_adaptive_roe_floor",
            "profit_ttm_floor": "_adaptive_profit_ttm_floor",
        })
        thresholds["_adaptive_group"] = thresholds["_adaptive_group"].astype(str)
        dataset["_adaptive_group"] = dataset["_adaptive_group"].astype(str)
        keep_cols = [
            "_adaptive_group",
            "_adaptive_growth_threshold",
            "_adaptive_roa_floor",
            "_adaptive_roe_floor",
            "_adaptive_profit_ttm_floor",
        ]
        dataset = dataset.merge(thresholds[keep_cols], on="_adaptive_group", how="left")
    if "_adaptive_roa_quality" not in dataset.columns:
        roa_cols = [col for col in ["roa_ttm_calc", "roa", "roa_calc"] if col in dataset.columns]
        dataset["_adaptive_roa_quality"] = dataset[roa_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0] if roa_cols else np.nan
    if "_adaptive_roe_quality" not in dataset.columns:
        roe_cols = [col for col in ["roe_ttm_calc", "roe", "roe_calc"] if col in dataset.columns]
        dataset["_adaptive_roe_quality"] = dataset[roe_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0] if roe_cols else np.nan
    metric_rows, prediction_frames = classifier_predictions(config, input_dir, dataset, summary)
    ridge_row, ridge_pred = ridge_predictions(config, dataset, summary, growth_target)
    arima_row, arima_pred = arima_predictions(config, dataset, summary, growth_target)
    metric_rows.extend([ridge_row, arima_row])
    prediction_frames.extend([ridge_pred, arima_pred])
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    order = metrics.groupby("model")["BalancedAccuracy"].max().sort_values(ascending=False).index.tolist()
    for metric in ["BalancedAccuracy", "F1", "AUC", "Precision", "Recall", "Accuracy"]:
        pivot = metrics.pivot_table(index="model", columns="config", values=metric, aggfunc="first").reindex(order)
        ax = pivot.plot(kind="bar", figsize=(13.5, 5.8), width=0.78)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(f"Adaptive target configs - test {metric}")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"compare_configs_test_{metric.lower()}.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_quarterly(predictions: pd.DataFrame, output_dir: Path) -> None:
    selected_models = ["RandomForest", "LightGBM", "XGBoost", "WeightedSoftVoting", "Ridge", "ARIMA"]
    data = predictions[predictions["model"].isin(selected_models)].copy()
    grouped = (
        data.groupby(["config", "model", "yq_index", "quarter_label"], dropna=False)
        .agg(
            actual_positive_rate=("actual", "mean"),
            predicted_positive_rate=("predicted", "mean"),
            Accuracy=("correct", "mean"),
            rows=("actual", "size"),
        )
        .reset_index()
        .sort_values(["config", "model", "yq_index"])
    )
    grouped.to_csv(output_dir / "quarterly_actual_vs_predicted.csv", index=False)
    for config, config_df in grouped.groupby("config"):
        models = [m for m in selected_models if m in set(config_df["model"])]
        fig, axes = plt.subplots(len(models), 1, figsize=(13.5, 3.2 * len(models)), sharex=True, sharey=True)
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            part = config_df[config_df["model"] == model].sort_values("yq_index")
            x = np.arange(len(part))
            ax.bar(x, part["Accuracy"], color="#dfe8f3", label="Accuracy")
            ax.plot(x, part["actual_positive_rate"], marker="o", color="#1f9d3a", linewidth=2.2, label="Actual positive rate")
            ax.plot(x, part["predicted_positive_rate"], marker="s", linestyle="--", color="#f29900", linewidth=2.2, label="Predicted positive rate")
            ax.set_title(model)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels(part["quarter_label"], rotation=30, ha="right")
        axes[0].legend(loc="lower left")
        fig.suptitle(f"Actual vs predicted by quarter - {config}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{config}_quarterly_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_config_winners(metrics: pd.DataFrame, output_dir: Path) -> None:
    top = metrics.sort_values(["config", "BalancedAccuracy"], ascending=[True, False]).groupby("config").head(5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (config, part) in zip(axes, top.groupby("config")):
        part = part.sort_values("BalancedAccuracy", ascending=True)
        y = np.arange(len(part))
        ax.barh(y, part["BalancedAccuracy"], color="#2f80ed", label="BA")
        ax.scatter(part["F1"], y, color="#f29900", label="F1", zorder=3)
        ax.scatter(part["AUC"], y, color="#1f9d3a", label="AUC", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(part["model"])
        ax.set_xlim(0, 1)
        ax.set_title(config)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Top models by Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "top_models_by_config.png", dpi=180, bbox_inches="tight")
    plt.close()


def add_notes(metrics: pd.DataFrame) -> pd.DataFrame:
    notes = []
    for _, row in metrics.iterrows():
        if row["model"] == "ARIMA":
            notes.append("Univariate time-series baseline; weak when each symbol has short history.")
        elif row["model"] == "Ridge":
            notes.append("Linear growth regressor; useful baseline but misses nonlinear sector effects.")
        elif row["model"] == "WeightedSoftVoting":
            notes.append("Combines tree models by validation weights; useful when it beats single models.")
        elif row["model"] in {"RandomForest", "ExtraTrees"}:
            notes.append("Tree ensemble; robust with tabular financial ratios and missing values.")
        elif row["model"] in {"LightGBM", "XGBoost", "HistGradientBoosting", "GradientBoosting"}:
            notes.append("Boosting model; often strong for nonlinear tabular financial data.")
        else:
            notes.append("")
    out = metrics.copy()
    out["note"] = notes
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "s70_q20_p20": Path(args.s70_dir),
        "s80_q10_p10": Path(args.s80_dir),
    }
    metrics = []
    predictions = []
    for config, input_dir in configs.items():
        config_metrics, config_predictions = load_config(config, input_dir, args.growth_target)
        metrics.append(config_metrics)
        predictions.append(config_predictions)
    metrics_df = add_notes(pd.concat(metrics, ignore_index=True))
    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df = metrics_df.sort_values(["BalancedAccuracy", "F1", "AUC"], ascending=False)
    metrics_df.to_csv(output_dir / "adaptive_config_model_comparison.csv", index=False)
    predictions_df.to_csv(output_dir / "adaptive_config_model_predictions.csv", index=False)
    plot_metric_bars(metrics_df, output_dir)
    plot_quarterly(predictions_df, output_dir)
    plot_config_winners(metrics_df, output_dir)
    print(f"Saved comparison outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

