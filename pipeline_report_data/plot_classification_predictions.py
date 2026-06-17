"""
Plot actual vs predicted classification labels from growth classification outputs.

Examples:
  python pipeline_report_data/plot_classification_predictions.py --output-dir outputs/adaptive_compare_models_1q/s80_q10_p10 --symbol ACB
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot actual vs predicted classification labels.")
    parser.add_argument("--output-dir", required=True, help="Directory containing predictions.csv.")
    parser.add_argument("--symbol", default="ACB")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--plots-dir", default="", help="Optional plot directory. Defaults to <output-dir>/plots.")
    return parser.parse_args()


def infer_label_order(labels: List[str]) -> List[str]:
    normalized = {str(label).lower() for label in labels}
    if normalized.issubset({"0.0", "1.0", "0", "1"}):
        return ["0.0", "1.0"]
    preferred = ["low", "medium", "high"]
    if normalized.issubset(set(preferred)):
        return preferred
    return sorted({str(label) for label in labels})


def label_display(label: str) -> str:
    text = str(label)
    if text in {"0", "0.0"}:
        return "down"
    if text in {"1", "1.0"}:
        return "up"
    return text


def encode_labels(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    return series.astype(str).map(mapping)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    required = {"symbol", "year", "quarter", "split", "actual_label", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions.csv: {sorted(missing)}")

    symbol = args.symbol.upper()
    plot_df = df[df["symbol"].astype(str).str.upper().eq(symbol)].copy()
    if args.split != "all":
        plot_df = plot_df[plot_df["split"].eq(args.split)].copy()
    if plot_df.empty:
        raise ValueError(f"No rows found for symbol={symbol}, split={args.split}")

    plot_df["year"] = pd.to_numeric(plot_df["year"], errors="coerce").astype("Int64")
    plot_df["quarter"] = pd.to_numeric(plot_df["quarter"], errors="coerce").astype("Int64")
    plot_df = plot_df.dropna(subset=["year", "quarter"]).sort_values(["year", "quarter"])
    plot_df["period"] = plot_df["year"].astype(str) + "Q" + plot_df["quarter"].astype(str)

    labels = infer_label_order(
        plot_df["actual_label"].astype(str).tolist() + plot_df["predicted_label"].astype(str).tolist()
    )
    mapping = {label: idx for idx, label in enumerate(labels)}
    plot_df["actual_encoded"] = encode_labels(plot_df["actual_label"], mapping)
    plot_df["predicted_encoded"] = encode_labels(plot_df["predicted_label"], mapping)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = range(len(plot_df))
    ax.plot(x, plot_df["actual_encoded"], color="#1f9d2f", linewidth=2.2, label="Actual label")
    ax.plot(x, plot_df["predicted_encoded"], color="#f5a000", linewidth=2.2, linestyle="--", label="Model prediction")

    if "prediction_confidence" in plot_df.columns:
        ax2 = ax.twinx()
        ax2.fill_between(
            x,
            0,
            plot_df["prediction_confidence"].astype(float),
            color="#7aa6dc",
            alpha=0.12,
            label="Prediction confidence",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Confidence")

    ax.set_title(f"Actual vs Predicted Growth Class - {symbol} ({args.split} splits)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Class")
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels([label_display(label) for label in labels])
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["period"].tolist(), rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)

    lines, line_labels = ax.get_legend_handles_labels()
    if "prediction_confidence" in plot_df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        line_labels += labels2
    ax.legend(lines, line_labels, loc="upper left")
    fig.tight_layout()

    suffix = f"{symbol}_{args.split}"
    png_path = plots_dir / f"actual_vs_predicted_class_{suffix}.png"
    csv_path = plots_dir / f"actual_vs_predicted_class_{suffix}.csv"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    cols = ["symbol", "year", "quarter", "split", "actual_label", "predicted_label"]
    if "prediction_confidence" in plot_df.columns:
        cols.append("prediction_confidence")
    plot_df[cols].to_csv(csv_path, index=False)
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

