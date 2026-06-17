from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "adaptive_compare_models_multiq" / "s80_q10_p10"
BASE_FEATURES = OUTPUT_ROOT / "multiq_base_features_clean.csv"
LATEX_DIR = ROOT / "latex_kltn_nbs"
MAIN_TEX = LATEX_DIR / "main.tex"
HORIZONS = (2, 3, 4, 8)
MODELS = "RandomForest,ExtraTrees,GradientBoosting,HistGradientBoosting,XGBoost,LightGBM"


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def python_cmd() -> str:
    return sys.executable


def build_base_features(force: bool = False) -> Path:
    if BASE_FEATURES.exists() and not force:
        print(f"Base feature file exists: {BASE_FEATURES}")
        return BASE_FEATURES

    print("Building multi-horizon base feature dataset...")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    from train.train_growth_classification import derive_classification_targets
    from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer

    engineer = GrowthFeatureEngineer()
    dataset = engineer.build_dataset_from_ml_tree(data_root=str(ROOT / "data"), tickers=None)
    dataset = derive_classification_targets(dataset)

    drop_cols = [
        col
        for col in dataset.columns
        if col in {"industry_l1_code", "industry_l1_name", "industry_l2_code", "industry_l2_name"}
        or col.startswith("industry_l1_code_")
        or col.startswith("industry_l2_code_")
        or col.startswith("_adaptive_")
        or col.startswith("target_adaptive_strong_profit_up_")
    ]
    dataset = dataset.drop(columns=drop_cols, errors="ignore")
    dataset.to_csv(BASE_FEATURES, index=False)
    print(f"Wrote {BASE_FEATURES} with shape {dataset.shape}")
    return BASE_FEATURES


def train_multiq(force: bool = False) -> None:
    base = build_base_features()
    for horizon in HORIZONS:
        out_dir = OUTPUT_ROOT / f"{horizon}q"
        summary_path = out_dir / "training_summary.json"
        if summary_path.exists() and not force:
            print(f"Training output exists for {horizon}q: {out_dir}")
            continue
        run(
            [
                python_cmd(),
                "-m",
                "train.train_growth_classification",
                "--dataset-path",
                str(base),
                "--target",
                f"target_adaptive_strong_profit_up_{horizon}q",
                "--output-dir",
                str(out_dir),
                "--adaptive-strong-quantile",
                "0.80",
                "--adaptive-quality-quantile",
                "0.10",
                "--adaptive-profit-ttm-quantile",
                "0.10",
                "--include-weighted-voting",
                "--enabled-models",
                MODELS,
                "--selection-metric",
                "val_F1",
            ]
        )


def collect_multiq_metrics() -> tuple[Path, Path]:
    warnings.filterwarnings("ignore")
    import __main__
    from train.train_growth_classification import WeightedSoftVotingClassifier
    from pipeline_report_data.compare_adaptive_configs_models import classifier_predictions

    __main__.WeightedSoftVotingClassifier = WeightedSoftVotingClassifier

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        out_dir = OUTPUT_ROOT / f"{horizon}q"
        dataset_path = out_dir / "classification_dataset.csv"
        summary_path = out_dir / "training_summary.json"
        if not dataset_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Missing training outputs for {horizon}q. Run mode 'multiq' first.")

        dataset = pd.read_csv(dataset_path)
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        horizon_rows, _ = classifier_predictions("s80_q10_p10", out_dir, dataset, summary)
        for row in horizon_rows:
            row["horizon"] = f"{horizon}q"
        rows.extend(horizon_rows)

    columns = [
        "horizon",
        "config",
        "model",
        "family",
        "threshold",
        "val_tuned_F1",
        "rows",
        "actual_positive_rate",
        "predicted_positive_rate",
        "Accuracy",
        "BalancedAccuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
    ]
    comparison = pd.DataFrame(rows)[columns].sort_values(
        ["horizon", "F1", "BalancedAccuracy", "AUC"],
        ascending=[True, False, False, False],
    )
    comparison_path = OUTPUT_ROOT / "multiq_classifier_comparison.csv"
    best_path = OUTPUT_ROOT / "multiq_best_by_horizon.csv"
    comparison.to_csv(comparison_path, index=False)
    comparison.groupby("horizon", sort=False).head(1).to_csv(best_path, index=False)
    print(f"Wrote {comparison_path}")
    print(f"Wrote {best_path}")
    return comparison_path, best_path


def latex_model_name(name: str) -> str:
    return {
        "WeightedSoftVoting": "Weighted Soft Voting",
        "RandomForest": "Random Forest",
        "GradientBoosting": "GradientBoosting",
        "HistGradientBoosting": "HistGradientBoosting",
        "ExtraTrees": "ExtraTrees",
        "LightGBM": "LightGBM",
        "XGBoost": "XGBoost",
    }.get(name, name)


def update_latex_multiq_table(best_path: Path | None = None) -> None:
    best_path = best_path or (OUTPUT_ROOT / "multiq_best_by_horizon.csv")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Run mode 'metrics' first.")

    best = pd.read_csv(best_path)
    rows = []
    order = {f"{item}q": item for item in HORIZONS}
    best["_order"] = best["horizon"].map(order)
    best = best.sort_values("_order")
    for item in best.itertuples(index=False):
        horizon = str(item.horizon).replace("q", " quý")
        rows.append(
            (
                f"{horizon} & {int(item.rows)} & {item.actual_positive_rate:.3f} "
                f"& {latex_model_name(str(item.model))} & {item.Accuracy:.3f} "
                f"& {item.BalancedAccuracy:.3f} & {item.Precision:.3f} "
                f"& {item.Recall:.3f} & {item.F1:.3f} & {item.AUC:.3f} \\\\\n\\hline"
            )
        )

    text = MAIN_TEX.read_text(encoding="utf-8")
    start = "% AUTO-GENERATED-MULTIQ-ROWS-START"
    end = "% AUTO-GENERATED-MULTIQ-ROWS-END"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.S)
    replacement = start + "\n\\hline\n" + "\n".join(rows) + "\n" + end
    updated, count = pattern.subn(lambda _match: replacement, text)
    if count != 1:
        raise RuntimeError("Could not locate multi-horizon auto-generated block in main.tex")
    MAIN_TEX.write_text(updated, encoding="utf-8")
    print(f"Updated {MAIN_TEX}")


def compile_pdf() -> None:
    run(["xelatex", "-interaction=nonstopmode", "main.tex"], cwd=LATEX_DIR)
    run(["xelatex", "-interaction=nonstopmode", "main.tex"], cwd=LATEX_DIR)
    print(f"Built {LATEX_DIR / 'main.pdf'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KLTN experiment and PDF pipeline.")
    parser.add_argument(
        "mode",
        choices=["base", "multiq", "metrics", "update-latex", "pdf", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument("--force-base", action="store_true", help="Rebuild the base feature CSV.")
    parser.add_argument("--force-train", action="store_true", help="Retrain horizons even if outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"base", "all"}:
        build_base_features(force=args.force_base)
    if args.mode in {"multiq", "all"}:
        if args.force_base:
            build_base_features(force=True)
        train_multiq(force=args.force_train)
    if args.mode in {"metrics", "all"}:
        _comparison_path, best_path = collect_multiq_metrics()
    else:
        best_path = OUTPUT_ROOT / "multiq_best_by_horizon.csv"
    if args.mode in {"update-latex", "all"}:
        update_latex_multiq_table(best_path)
    if args.mode in {"pdf", "all"}:
        compile_pdf()


if __name__ == "__main__":
    main()
