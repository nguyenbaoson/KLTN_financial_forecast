from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot adaptive strong-growth target thresholds.")
    parser.add_argument(
        "--output-dir",
        default="outputs/adaptive_compare_models_1q/s80_q10_p10",
        help="Directory containing adaptive_target_thresholds.csv and classification_dataset.csv.",
    )
    return parser.parse_args()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_group_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    if not {"_adaptive_group", "industry_l1_name"}.issubset(dataset.columns):
        return pd.DataFrame(columns=["group", "group_label"])
    labels = (
        dataset[["_adaptive_group", "industry_l1_name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"_adaptive_group": "group"})
    )
    labels["group"] = labels["group"].astype(str)
    labels["group_label"] = labels["group"] + " - " + labels["industry_l1_name"].astype(str)
    return labels[["group", "group_label"]]


def add_labels(thresholds: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    result = thresholds.copy()
    result["group"] = result["group"].astype(str)
    labels = build_group_labels(dataset)
    result = result.merge(labels, on="group", how="left")
    result["group_label"] = result["group_label"].fillna(result["group"])
    result = result.sort_values("growth_threshold", ascending=False).reset_index(drop=True)
    return result


def plot_growth_threshold(thresholds: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = np.where(thresholds["used_group_threshold"], "#2f80ed", "#9aa6b2")
    ax.bar(thresholds["group_label"], thresholds["growth_threshold"], color=colors)
    ax.set_title("Adaptive strong-growth threshold by industry")
    ax.set_ylabel("Profit growth threshold (%)")
    ax.set_xlabel("Industry")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(thresholds["growth_threshold"]):
        ax.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    savefig(plots_dir / "adaptive_growth_threshold_by_industry.png")


def plot_quality_floors(thresholds: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(thresholds))
    width = 0.38
    ax.bar(x - width / 2, thresholds["roa_floor"], width, label="ROA floor", color="#27ae60")
    ax.bar(x + width / 2, thresholds["roe_floor"], width, label="ROE floor", color="#f2994a")
    ax.set_title("Minimum ROA/ROE floor by industry")
    ax.set_ylabel("Ratio floor (%)")
    ax.set_xlabel("Industry")
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds["group_label"], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    savefig(plots_dir / "adaptive_roa_roe_floors_by_industry.png")


def plot_profit_ttm_floor(thresholds: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    values = thresholds["profit_ttm_floor"] / 1e9
    ax.bar(thresholds["group_label"], values, color="#8e44ad")
    ax.set_title("Minimum net profit TTM floor by industry")
    ax.set_ylabel("Net profit TTM floor (billion VND)")
    ax.set_xlabel("Industry")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    savefig(plots_dir / "adaptive_net_profit_ttm_floor_by_industry.png")


def plot_target_distribution(dataset: pd.DataFrame, thresholds: pd.DataFrame, plots_dir: Path) -> None:
    target_col = "target_adaptive_strong_profit_up_1q"
    if target_col not in dataset.columns or "_adaptive_group" not in dataset.columns:
        return
    rows = dataset.dropna(subset=[target_col]).copy()
    rows["_adaptive_group"] = rows["_adaptive_group"].astype(str)
    counts = (
        rows.groupby(["_adaptive_group", target_col])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0.0: "label_0", 1.0: "label_1"})
    )
    for col in ["label_0", "label_1"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.reset_index().rename(columns={"_adaptive_group": "group"})
    counts = thresholds[["group", "group_label"]].merge(counts, on="group", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(counts))
    ax.bar(x, counts["label_0"], label="Label 0: non-growth", color="#b8c2cc")
    ax.bar(x, counts["label_1"], bottom=counts["label_0"], label="Label 1: adaptive strong growth", color="#2f80ed")
    ax.set_title("Adaptive target distribution by industry")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Industry")
    ax.set_xticks(x)
    ax.set_xticklabels(counts["group_label"], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    savefig(plots_dir / "adaptive_target_distribution_by_industry.png")


def plot_dashboard(thresholds: pd.DataFrame, dataset: pd.DataFrame, plots_dir: Path) -> None:
    target_col = "target_adaptive_strong_profit_up_1q"
    counts = pd.DataFrame()
    if target_col in dataset.columns and "_adaptive_group" in dataset.columns:
        rows = dataset.dropna(subset=[target_col]).copy()
        rows["_adaptive_group"] = rows["_adaptive_group"].astype(str)
        counts = (
            rows.groupby(["_adaptive_group", target_col])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0.0: "label_0", 1.0: "label_1"})
            .reset_index()
            .rename(columns={"_adaptive_group": "group"})
        )
        for col in ["label_0", "label_1"]:
            if col not in counts.columns:
                counts[col] = 0
        counts = thresholds[["group", "group_label"]].merge(counts, on="group", how="left").fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    x = np.arange(len(thresholds))

    axes[0, 0].bar(x, thresholds["growth_threshold"], color="#2f80ed")
    axes[0, 0].set_title("Growth threshold")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].grid(axis="y", alpha=0.25)

    width = 0.38
    axes[0, 1].bar(x - width / 2, thresholds["roa_floor"], width, label="ROA", color="#27ae60")
    axes[0, 1].bar(x + width / 2, thresholds["roe_floor"], width, label="ROE", color="#f2994a")
    axes[0, 1].set_title("ROA/ROE floor")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].bar(x, thresholds["profit_ttm_floor"] / 1e9, color="#8e44ad")
    axes[1, 0].set_title("Net profit TTM floor")
    axes[1, 0].set_ylabel("Billion VND")
    axes[1, 0].grid(axis="y", alpha=0.25)

    if not counts.empty:
        axes[1, 1].bar(x, counts["label_0"], label="0", color="#b8c2cc")
        axes[1, 1].bar(x, counts["label_1"], bottom=counts["label_0"], label="1", color="#2f80ed")
        axes[1, 1].set_title("Target distribution")
        axes[1, 1].set_ylabel("Rows")
        axes[1, 1].grid(axis="y", alpha=0.25)
        axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds["group_label"], rotation=35, ha="right")

    savefig(plots_dir / "adaptive_target_dashboard.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    thresholds_path = output_dir / "adaptive_target_thresholds.csv"
    dataset_path = output_dir / "classification_dataset.csv"
    plots_dir = output_dir / "plots"

    thresholds = pd.read_csv(thresholds_path)
    dataset = pd.read_csv(dataset_path)
    thresholds = add_labels(thresholds, dataset)

    plot_growth_threshold(thresholds, plots_dir)
    plot_quality_floors(thresholds, plots_dir)
    plot_profit_ttm_floor(thresholds, plots_dir)
    plot_target_distribution(dataset, thresholds, plots_dir)
    plot_dashboard(thresholds, dataset, plots_dir)

    print(f"Saved plots to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()

