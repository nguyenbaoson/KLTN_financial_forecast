from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


MODEL_ARTIFACTS = {
    "LightGBM": "lightgbm_classifier.joblib",
    "RandomForest": "randomforest_classifier.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot actual vs predicted by quarter for one adaptive target config.")
    parser.add_argument(
        "--input-dir",
        default="outputs/adaptive_compare_models_1q/s80_q10_p10",
        help="Training output directory containing classification_dataset.csv and model artifacts.",
    )
    parser.add_argument(
        "--title",
        default="Actual vs predicted by quarter - Adaptive target s80_q10_p10",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/adaptive_compare_models_1q/s80_q10_p10/plots",
    )
    return parser.parse_args()


def positive_probability(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    proba = model.predict_proba(frame[feature_cols])
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1]))).astype(str)
    positive_idx = int(np.where(classes == "1.0")[0][0]) if "1.0" in classes else 1
    return proba[:, positive_idx]


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def split_dataset(dataset: pd.DataFrame, target_col: str, summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trainable = dataset.dropna(subset=[target_col]).copy()
    trainable[target_col] = pd.to_numeric(trainable[target_col], errors="coerce")
    train_end = int(summary["split_info"]["train_end_yq"])
    val_end = int(summary["split_info"]["val_end_yq"])
    val = trainable[(trainable["yq_index"] > train_end) & (trainable["yq_index"] <= val_end)].copy()
    test = trainable[trainable["yq_index"] > val_end].copy()
    return val, test


def build_quarterly_predictions(input_dir: Path) -> pd.DataFrame:
    with open(input_dir / "training_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset = pd.read_csv(input_dir / "classification_dataset.csv")
    target_col = summary["target_col"]
    feature_cols = summary["feature_cols"]
    val, test = split_dataset(dataset, target_col, summary)

    rows = []
    for model_name, artifact_name in MODEL_ARTIFACTS.items():
        model = joblib.load(input_dir / artifact_name)
        val_y = val[target_col].astype(int).to_numpy()
        val_proba = positive_probability(model, val, feature_cols)
        threshold = tune_threshold(val_y, val_proba)

        frame = test[["symbol", "year", "quarter", "yq_index", target_col]].copy()
        frame = frame.rename(columns={target_col: "actual"})
        frame["model"] = model_name
        frame["prob_1"] = positive_probability(model, test, feature_cols)
        frame["predicted"] = (frame["prob_1"] >= threshold).astype(int)
        frame["correct"] = (frame["actual"].astype(int) == frame["predicted"]).astype(int)
        frame["threshold"] = threshold
        rows.append(frame)

    predictions = pd.concat(rows, ignore_index=True)
    quarterly = (
        predictions.groupby(["model", "yq_index", "year", "quarter"], as_index=False)
        .agg(
            actual_rate=("actual", "mean"),
            predicted_rate=("predicted", "mean"),
            accuracy=("correct", "mean"),
            avg_probability=("prob_1", "mean"),
            rows=("actual", "size"),
            threshold=("threshold", "first"),
        )
        .sort_values(["model", "yq_index"])
    )
    quarterly["quarter_label"] = quarterly["year"].astype(int).astype(str) + "Q" + quarterly["quarter"].astype(int).astype(str)
    return quarterly


def plot_quarterly(quarterly: pd.DataFrame, title: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_order = ["LightGBM", "RandomForest"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.2), sharey=True)

    for ax, model_name in zip(axes, model_order):
        model_df = quarterly[quarterly["model"].eq(model_name)].sort_values("yq_index")
        x = np.arange(len(model_df))
        ax.bar(x, model_df["accuracy"], color="#dfe8f3", label="Accuracy")
        ax.plot(x, model_df["actual_rate"], marker="o", color="#1f9d3a", linewidth=2.5, label="Actual positive rate")
        ax.plot(x, model_df["predicted_rate"], marker="s", linestyle="--", color="#f29900", linewidth=2.5, label="Predicted positive rate")
        ax.set_title(model_name)
        ax.set_xlabel("Quarter")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(model_df["quarter_label"], rotation=35, ha="right")

    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="lower left")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "s80_q10_p10_rf_lgbm_actual_vs_predicted_by_quarter.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarterly = build_quarterly_predictions(input_dir)
    quarterly.to_csv(output_dir / "s80_q10_p10_rf_lgbm_quarterly.csv", index=False)
    plot_quarterly(quarterly, args.title, output_dir)
    print(f"Saved plot to: {(output_dir / 's80_q10_p10_rf_lgbm_actual_vs_predicted_by_quarter.png').resolve()}")


if __name__ == "__main__":
    main()

