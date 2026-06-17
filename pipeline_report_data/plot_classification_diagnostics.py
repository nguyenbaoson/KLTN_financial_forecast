"""Plot classification diagnostics for thesis model outputs.

Examples:
  python pipeline_report_data/plot_classification_diagnostics.py --output-dir outputs/adaptive_compare_models_1q/s80_q10_p10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["Accuracy", "BalancedAccuracy", "F1", "AUC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot classification model diagnostics.")
    parser.add_argument("--output-dir", required=True, help="Directory containing model_results.csv and predictions.csv.")
    parser.add_argument("--plots-dir", default="", help="Defaults to <output-dir>/plots.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def label_display(label: object) -> str:
    text = str(label)
    if text in {"0", "0.0"}:
        return "not strong"
    if text in {"1", "1.0"}:
        return "strong up"
    return text


def plot_model_metrics(output_dir: Path, plots_dir: Path) -> None:
    results_path = output_dir / "model_results.csv"
    if not results_path.exists():
        return

    results = pd.read_csv(results_path)
    results = results.sort_values("test_F1", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 7))
    x = range(len(results))
    width = 0.18
    colors = ["#4062bb", "#59a14f", "#f28e2b", "#b07aa1"]
    offsets = [(-1.5 + idx) * width for idx in range(len(METRICS))]

    for metric, color, offset in zip(METRICS, colors, offsets):
        col = f"test_{metric}"
        if col not in results.columns:
            continue
        ax.bar([idx + offset for idx in x], results[col], width=width, label=f"test {metric}", color=color)

    ax.set_title("Model Comparison on Test Set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(results["model"].tolist(), rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(plots_dir / "model_test_metrics_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, METRICS):
        val_col = f"val_{metric}"
        test_col = f"test_{metric}"
        if val_col not in results.columns or test_col not in results.columns:
            ax.axis("off")
            continue
        x = range(len(results))
        ax.plot(x, results[val_col], marker="o", linewidth=2, label="validation", color="#4e79a7")
        ax.plot(x, results[test_col], marker="s", linewidth=2, label="test", color="#e15759")
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xticks(list(x))
        ax.set_xticklabels(results["model"].tolist(), rotation=35, ha="right")
    axes[0].legend(loc="lower right")
    fig.suptitle("Validation vs Test Metrics by Model", y=0.995)
    fig.tight_layout()
    fig.savefig(plots_dir / "model_val_vs_test_metrics.png", dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted_by_quarter(output_dir: Path, plots_dir: Path, split: str) -> None:
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        return

    df = pd.read_csv(predictions_path)
    required = {"year", "quarter", "split", "actual_label", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions.csv: {sorted(missing)}")

    df = df[df["split"].eq(split)].copy()
    if df.empty:
        return

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "quarter"])
    df["period"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    df["actual_num"] = pd.to_numeric(df["actual_label"], errors="coerce")
    df["predicted_num"] = pd.to_numeric(df["predicted_label"], errors="coerce")
    df["correct"] = (df["actual_label"].astype(str) == df["predicted_label"].astype(str)).astype(float)

    grouped = (
        df.groupby(["year", "quarter", "period"], as_index=False)
        .agg(
            actual_strong_rate=("actual_num", "mean"),
            predicted_strong_rate=("predicted_num", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
        .sort_values(["year", "quarter"])
    )
    grouped.to_csv(plots_dir / f"aggregate_actual_vs_predicted_{split}.csv", index=False)

    x = range(len(grouped))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, grouped["actual_strong_rate"], marker="o", linewidth=2.5, label="Actual strong-up rate", color="#1f9d2f")
    ax.plot(
        x,
        grouped["predicted_strong_rate"],
        marker="s",
        linewidth=2.5,
        linestyle="--",
        label="Predicted strong-up rate",
        color="#f5a000",
    )
    ax.bar(x, grouped["accuracy"], alpha=0.16, label="Accuracy", color="#4e79a7")
    ax.set_title(f"Actual vs Predicted Strong-Growth Rate by Quarter ({split})")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["period"].tolist(), rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plots_dir / f"aggregate_actual_vs_predicted_{split}.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(output_dir: Path, plots_dir: Path, split: str) -> None:
    matrix_path = output_dir / "confusion_matrix.csv"
    if not matrix_path.exists():
        return

    df = pd.read_csv(matrix_path)
    df = df[df["split"].eq(split)].copy()
    if df.empty:
        return

    labels = sorted(set(df["actual"].astype(str)) | set(df["predicted"].astype(str)))
    matrix = (
        df.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=labels, columns=labels, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({split})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([label_display(label) for label in labels])
    ax.set_yticklabels([label_display(label) for label in labels])
    for row_idx in range(len(labels)):
        for col_idx in range(len(labels)):
            value = int(matrix.iloc[row_idx, col_idx])
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / f"confusion_matrix_{split}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_model_metrics(output_dir, plots_dir)
    plot_actual_vs_predicted_by_quarter(output_dir, plots_dir, args.split)
    plot_confusion_matrix(output_dir, plots_dir, args.split)
    print(f"Saved diagnostic plots to {plots_dir.resolve()}")


if __name__ == "__main__":
    main()

