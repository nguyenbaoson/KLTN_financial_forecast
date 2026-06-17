from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_ARTIFACTS = {
    "RandomForest": "randomforest_classifier.joblib",
    "LightGBM": "lightgbm_classifier.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive target by full test, quarter, and sector.")
    parser.add_argument("--input-dir", default="outputs/adaptive_compare_models_1q/s80_q10_p10")
    parser.add_argument("--output-dir", default="")
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


def metric_dict(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    out = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else float("nan"),
        "Accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else float("nan"),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else float("nan"),
        "Precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "Recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "F1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "AUC": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    return out


def build_predictions(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(input_dir / "training_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset = pd.read_csv(input_dir / "classification_dataset.csv")
    target_col = summary["target_col"]
    feature_cols = summary["feature_cols"]
    trainable = dataset.dropna(subset=[target_col]).copy()
    trainable[target_col] = pd.to_numeric(trainable[target_col], errors="coerce")
    train_end = int(summary["split_info"]["train_end_yq"])
    val_end = int(summary["split_info"]["val_end_yq"])
    val = trainable[(trainable["yq_index"] > train_end) & (trainable["yq_index"] <= val_end)].copy()
    test = trainable[trainable["yq_index"] > val_end].copy()

    rows = []
    metrics = []
    for model_name, artifact_name in MODEL_ARTIFACTS.items():
        model = joblib.load(input_dir / artifact_name)
        threshold = tune_threshold(
            val[target_col].astype(int).to_numpy(),
            positive_probability(model, val, feature_cols),
        )
        proba = positive_probability(model, test, feature_cols)
        pred = (proba >= threshold).astype(int)

        frame = test.copy()
        frame["actual"] = frame[target_col].astype(int)
        frame["model"] = model_name
        frame["prob_1"] = proba
        frame["predicted"] = pred
        frame["correct"] = (frame["actual"].to_numpy() == pred).astype(int)
        frame["threshold"] = threshold
        frame["quarter_label"] = frame["year"].astype(int).astype(str) + "Q" + frame["quarter"].astype(int).astype(str)
        frame["sector_label"] = (
            frame.get("industry_l1_code", pd.Series(["unknown"] * len(frame), index=frame.index)).astype(str)
            + " - "
            + frame.get("industry_l1_name", pd.Series(["unknown"] * len(frame), index=frame.index)).astype(str)
        )
        rows.append(frame)
        metrics.append({"scope": "full_test", "model": model_name, "threshold": threshold, **metric_dict(frame["actual"].to_numpy(), pred, proba)})

    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def grouped_metrics(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", *group_cols], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        model_name = keys[0]
        values = keys[1:]
        data = {"model": model_name}
        data.update(dict(zip(group_cols, values)))
        rows.append({
            **data,
            **metric_dict(group["actual"].to_numpy(), group["predicted"].to_numpy(), group["prob_1"].to_numpy()),
        })
    return pd.DataFrame(rows)


def plot_quarter_metrics(quarter_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), sharey=True)
    for ax, (model_name, data) in zip(axes, quarter_metrics.groupby("model")):
        data = data.sort_values("yq_index")
        x = np.arange(len(data))
        ax.bar(x, data["Accuracy"], color="#dfe8f3", label="Accuracy")
        ax.plot(x, data["positive_rate"], marker="o", color="#1f9d3a", linewidth=2.5, label="Actual positive rate")
        ax.plot(x, data["predicted_positive_rate"], marker="s", linestyle="--", color="#f29900", linewidth=2.5, label="Predicted positive rate")
        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(data["quarter_label"], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="lower left")
    fig.suptitle("Full test evaluation by quarter - s80_q10_p10", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "full_test_by_quarter_s80_q10_p10.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_sector_metrics(sector_metrics: pd.DataFrame, output_dir: Path) -> None:
    for model_name, data in sector_metrics.groupby("model"):
        data = data.sort_values("rows", ascending=False)
        x = np.arange(len(data))
        fig, ax1 = plt.subplots(figsize=(14, 5.8))
        ax1.bar(x, data["rows"], color="#dfe8f3", label="Rows")
        ax1.set_ylabel("Rows")
        ax1.set_xticks(x)
        ax1.set_xticklabels(data["sector_label"], rotation=30, ha="right")
        ax1.grid(axis="y", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(x, data["Accuracy"], marker="o", linewidth=2.4, color="#2f80ed", label="Accuracy")
        ax2.plot(x, data["F1"], marker="s", linewidth=2.4, color="#f29900", label="F1")
        ax2.plot(x, data["BalancedAccuracy"], marker="^", linewidth=2.4, color="#1f9d3a", label="BA")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Score")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title(f"Full test evaluation by sector - {model_name}")
        plt.tight_layout()
        safe_model = model_name.lower()
        plt.savefig(output_dir / f"full_test_by_sector_{safe_model}_s80_q10_p10.png", dpi=180, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots" / "full_test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, full_metrics = build_predictions(input_dir)
    quarter_metrics = grouped_metrics(predictions, ["yq_index", "quarter_label"])
    sector_metrics = grouped_metrics(predictions, ["sector_label"])

    predictions.to_csv(output_dir / "full_test_predictions_s80_q10_p10.csv", index=False)
    full_metrics.to_csv(output_dir / "full_test_metrics_s80_q10_p10.csv", index=False)
    quarter_metrics.to_csv(output_dir / "full_test_by_quarter_s80_q10_p10.csv", index=False)
    sector_metrics.to_csv(output_dir / "full_test_by_sector_s80_q10_p10.csv", index=False)
    plot_quarter_metrics(quarter_metrics, output_dir)
    plot_sector_metrics(sector_metrics, output_dir)
    print(f"Saved full test evaluation to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

