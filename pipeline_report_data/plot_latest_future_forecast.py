from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot latest future forecasts.")
    parser.add_argument("--input-dir", default="outputs/adaptive_compare_models_1q/s80_q10_p10")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def next_quarter_label(year: int, quarter: int) -> str:
    if quarter == 4:
        return f"{year + 1}Q1"
    return f"{year}Q{quarter + 1}"


def enrich_forecast(input_dir: Path) -> pd.DataFrame:
    forecast = pd.read_csv(input_dir / "latest_forecast.csv")
    dataset = pd.read_csv(input_dir / "classification_dataset.csv")
    meta_cols = [
        "symbol",
        "year",
        "quarter",
        "yq_index",
        "industry_l1_code",
        "industry_l1_name",
        "industry_l2_code",
        "industry_l2_name",
        "net_profit",
        "net_profit_ttm_calc",
        "roa",
        "roe",
        "roa_ttm_calc",
        "roe_ttm_calc",
    ]
    available = [col for col in meta_cols if col in dataset.columns]
    latest_meta = (
        dataset[available]
        .sort_values(["symbol", "yq_index"])
        .drop_duplicates("symbol", keep="last")
    )
    result = forecast.merge(latest_meta, on=["symbol", "year", "quarter", "yq_index"], how="left")
    result["current_quarter"] = result["year"].astype(int).astype(str) + "Q" + result["quarter"].astype(int).astype(str)
    result["forecast_quarter"] = [
        next_quarter_label(int(year), int(quarter))
        for year, quarter in zip(result["year"], result["quarter"])
    ]
    result["sector"] = (
        result.get("industry_l1_code", pd.Series(["unknown"] * len(result))).astype(str)
        + " - "
        + result.get("industry_l1_name", pd.Series(["unknown"] * len(result))).astype(str)
    )
    result["signal"] = np.where(result["predicted_label"].astype(float).eq(1.0), "strong_growth", "not_strong")
    return result.sort_values("prob_1.0", ascending=False).reset_index(drop=True)


def plot_top_forecasts(forecast: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    top = forecast.head(top_n).sort_values("prob_1.0")
    colors = np.where(top["predicted_label"].astype(float).eq(1.0), "#2f80ed", "#b8c2cc")
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.28)))
    ax.barh(top["symbol"], top["prob_1.0"], color=colors)
    ax.axvline(float(top["decision_threshold"].iloc[0]), color="#d35400", linestyle="--", linewidth=2, label="Decision threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of adaptive strong growth")
    ax.set_ylabel("Symbol")
    ax.set_title("Top latest forecasts - adaptive target s80_q10_p10")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "latest_forecast_top_symbols.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_sector_forecasts(forecast: pd.DataFrame, output_dir: Path) -> None:
    sector = (
        forecast.groupby("sector", as_index=False)
        .agg(
            symbols=("symbol", "count"),
            predicted_positive_rate=("predicted_label", lambda values: pd.to_numeric(values, errors="coerce").mean()),
            avg_probability=("prob_1.0", "mean"),
        )
        .sort_values("avg_probability", ascending=False)
    )
    sector.to_csv(output_dir / "latest_forecast_by_sector.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(sector))
    ax1.bar(x, sector["symbols"], color="#dfe8f3", label="Symbols")
    ax1.set_ylabel("Symbols")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sector["sector"], rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, sector["avg_probability"], marker="o", color="#2f80ed", linewidth=2.5, label="Avg probability")
    ax2.plot(x, sector["predicted_positive_rate"], marker="s", color="#1f9d3a", linewidth=2.5, label="Predicted positive rate")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Rate")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Latest forecasts by sector")
    plt.tight_layout()
    plt.savefig(output_dir / "latest_forecast_by_sector.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots" / "future_forecast"
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast = enrich_forecast(input_dir)
    forecast.to_csv(output_dir / "latest_future_forecast_enriched.csv", index=False)
    plot_top_forecasts(forecast, output_dir, args.top_n)
    plot_sector_forecasts(forecast, output_dir)

    top_positive = forecast[forecast["predicted_label"].astype(float).eq(1.0)].head(args.top_n)
    top_positive.to_csv(output_dir / "latest_future_forecast_top_positive.csv", index=False)
    print(f"Saved latest future forecast summary to: {output_dir.resolve()}")
    print(top_positive[["symbol", "current_quarter", "forecast_quarter", "sector", "prob_1.0", "prediction_confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()

