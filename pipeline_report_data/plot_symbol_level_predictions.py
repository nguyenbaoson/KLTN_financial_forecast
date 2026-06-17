from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODEL_ARTIFACTS = {
    "RandomForest": "randomforest_classifier.joblib",
    "LightGBM": "lightgbm_classifier.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot symbol-level actual vs predicted labels.")
    parser.add_argument("--input-dir", default="outputs/adaptive_compare_models_1q/s80_q10_p10")
    parser.add_argument("--tickers", default="ACB,VCB,HPG,MWG,FPT,VHM")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
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


def assign_split(dataset: pd.DataFrame, target_col: str, summary: dict[str, Any]) -> pd.DataFrame:
    trainable = dataset.dropna(subset=[target_col]).copy()
    trainable[target_col] = pd.to_numeric(trainable[target_col], errors="coerce")
    train_end = int(summary["split_info"]["train_end_yq"])
    val_end = int(summary["split_info"]["val_end_yq"])
    trainable["split"] = np.select(
        [
            trainable["yq_index"] <= train_end,
            (trainable["yq_index"] > train_end) & (trainable["yq_index"] <= val_end),
            trainable["yq_index"] > val_end,
        ],
        ["train", "validation", "test"],
        default="unknown",
    )
    return trainable


def build_predictions(input_dir: Path, tickers: list[str], split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(input_dir / "training_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    dataset = pd.read_csv(input_dir / "classification_dataset.csv")
    target_col = summary["target_col"]
    feature_cols = summary["feature_cols"]
    trainable = assign_split(dataset, target_col, summary)

    val = trainable[trainable["split"].eq("validation")].copy()
    if split_name == "all":
        frame = trainable.copy()
    else:
        frame = trainable[trainable["split"].eq(split_name)].copy()

    tickers_upper = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    frame = frame[frame["symbol"].astype(str).str.upper().isin(tickers_upper)].copy()
    if frame.empty:
        raise ValueError("No selected ticker rows found in the requested split.")

    rows = []
    metric_rows = []
    for model_name, artifact_name in MODEL_ARTIFACTS.items():
        model = joblib.load(input_dir / artifact_name)
        val_y = val[target_col].astype(int).to_numpy()
        val_proba = positive_probability(model, val, feature_cols)
        threshold = tune_threshold(val_y, val_proba)

        out = frame[["symbol", "year", "quarter", "yq_index", "split", target_col]].copy()
        out = out.rename(columns={target_col: "actual"})
        out["model"] = model_name
        out["prob_1"] = positive_probability(model, frame, feature_cols)
        out["predicted"] = (out["prob_1"] >= threshold).astype(int)
        out["correct"] = (out["actual"].astype(int) == out["predicted"]).astype(int)
        out["threshold"] = threshold
        rows.append(out)

        for symbol, symbol_df in out.groupby("symbol"):
            metric_rows.append({
                "symbol": symbol,
                "model": model_name,
                "rows": int(len(symbol_df)),
                "accuracy": float(accuracy_score(symbol_df["actual"].astype(int), symbol_df["predicted"])),
                "actual_positive_rate": float(symbol_df["actual"].mean()),
                "predicted_positive_rate": float(symbol_df["predicted"].mean()),
                "avg_probability": float(symbol_df["prob_1"].mean()),
                "threshold": threshold,
            })

    predictions = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions["quarter_label"] = predictions["year"].astype(int).astype(str) + "Q" + predictions["quarter"].astype(int).astype(str)
    return predictions, metrics


def plot_symbols(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = sorted(predictions["symbol"].unique())
    ncols = 2
    nrows = int(np.ceil(len(symbols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, max(4, nrows * 3.8)), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, symbol in zip(axes, symbols):
        symbol_df = predictions[predictions["symbol"].eq(symbol)].sort_values(["model", "yq_index"])
        base = symbol_df.drop_duplicates("yq_index").sort_values("yq_index")
        x = np.arange(len(base))
        ax.plot(x, base["actual"], marker="o", color="#1f9d3a", linewidth=2.5, label="Actual label")
        for model_name, model_df in symbol_df.groupby("model"):
            model_df = model_df.sort_values("yq_index")
            style = "--" if model_name == "RandomForest" else ":"
            color = "#f29900" if model_name == "RandomForest" else "#2f80ed"
            ax.plot(x, model_df["predicted"], marker="s", linestyle=style, color=color, linewidth=2, label=f"{model_name} pred")
            ax.bar(
                x,
                model_df["prob_1"],
                width=0.32,
                alpha=0.12,
                color=color,
                label=f"{model_name} prob" if symbol == symbols[0] else None,
            )
        ax.set_title(symbol)
        ax.set_xticks(x)
        ax.set_xticklabels(base["quarter_label"], rotation=35, ha="right")
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(symbols):]:
        ax.axis("off")

    axes[0].legend(loc="lower left")
    fig.suptitle("Symbol-level actual vs predicted labels - adaptive target s80_q10_p10", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "symbol_level_actual_vs_predicted_s80_q10_p10.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots" / "symbol_level"
    tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]

    predictions, metrics = build_predictions(input_dir, tickers, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "symbol_level_predictions.csv", index=False)
    metrics.to_csv(output_dir / "symbol_level_metrics.csv", index=False)
    plot_symbols(predictions, output_dir)
    print(f"Saved symbol-level plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

