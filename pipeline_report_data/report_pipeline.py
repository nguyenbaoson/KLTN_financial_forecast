"""One entry point for thesis evaluation and plotting scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import resolve_repo_path


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    args: tuple[str, ...]
    required: tuple[Path, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plotting/evaluation scripts from one place. This is not a crawl pipeline."
    )
    parser.add_argument(
        "mode",
        choices=["all", "classification", "adaptive"],
        nargs="?",
        default="all",
        help="Which report group to run.",
    )
    parser.add_argument(
        "--model-output-dir",
        default="outputs/adaptive_compare_models_1q/s80_q10_p10",
        help="A model output directory containing model_results.csv/predictions.csv/latest_forecast.csv.",
    )
    parser.add_argument(
        "--adaptive-input-dir",
        default="outputs/adaptive_compare_models_1q/s80_q10_p10",
        help="Adaptive config output directory containing classification_dataset.csv and model artifacts.",
    )
    parser.add_argument("--s70-dir", default="outputs/adaptive_compare_models_1q/s70_q20_p20")
    parser.add_argument("--s80-dir", default="outputs/adaptive_compare_models_1q/s80_q10_p10")
    parser.add_argument("--comparison-output-dir", default="outputs/adaptive_compare_models_1q/comparison")
    parser.add_argument("--tickers", default="ACB,VCB,HPG,MWG,FPT,VHM")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping steps with missing inputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def path_arg(path: Path) -> str:
    return str(path)


def classification_steps(args: argparse.Namespace) -> list[Step]:
    output_dir = resolve_repo_path(args.model_output_dir)
    split = "test" if args.split == "all" else args.split
    return [
        Step(
            name="classification diagnostics",
            script="pipeline_report_data/plot_classification_diagnostics.py",
            args=("--output-dir", path_arg(output_dir), "--split", split),
            required=(output_dir / "model_results.csv", output_dir / "predictions.csv"),
        ),
        Step(
            name="classification predictions",
            script="pipeline_report_data/plot_classification_predictions.py",
            args=("--output-dir", path_arg(output_dir), "--split", split),
            required=(output_dir / "predictions.csv",),
        ),
        Step(
            name="latest forecast",
            script="pipeline_report_data/plot_latest_future_forecast.py",
            args=("--input-dir", path_arg(output_dir)),
            required=(output_dir / "latest_forecast.csv", output_dir / "classification_dataset.csv"),
        ),
        Step(
            name="symbol-level predictions",
            script="pipeline_report_data/plot_symbol_level_predictions.py",
            args=("--input-dir", path_arg(output_dir), "--tickers", args.tickers, "--split", args.split),
            required=(output_dir / "classification_dataset.csv", output_dir / "training_summary.json"),
        ),
    ]


def adaptive_steps(args: argparse.Namespace) -> list[Step]:
    adaptive_dir = resolve_repo_path(args.adaptive_input_dir)
    model_dir = resolve_repo_path(args.model_output_dir)
    s70_dir = resolve_repo_path(args.s70_dir)
    s80_dir = resolve_repo_path(args.s80_dir)
    comparison_dir = resolve_repo_path(args.comparison_output_dir)
    return [
        Step(
            name="compare adaptive configs",
            script="pipeline_report_data/compare_adaptive_configs_models.py",
            args=("--s70-dir", path_arg(s70_dir), "--s80-dir", path_arg(s80_dir), "--output-dir", path_arg(comparison_dir)),
            required=(s70_dir / "classification_dataset.csv", s80_dir / "classification_dataset.csv"),
        ),
        Step(
            name="adaptive quarter/sector evaluation",
            script="pipeline_report_data/evaluate_adaptive_by_quarter_sector.py",
            args=("--input-dir", path_arg(adaptive_dir)),
            required=(adaptive_dir / "classification_dataset.csv", adaptive_dir / "training_summary.json"),
        ),
        Step(
            name="adaptive target thresholds",
            script="pipeline_report_data/plot_adaptive_target_thresholds.py",
            args=("--output-dir", path_arg(model_dir)),
            required=(model_dir / "adaptive_target_thresholds.csv", model_dir / "classification_dataset.csv"),
        ),
        Step(
            name="single adaptive quarterly plot",
            script="pipeline_report_data/plot_single_adaptive_config_quarterly.py",
            args=("--input-dir", path_arg(adaptive_dir), "--output-dir", path_arg(adaptive_dir / "plots")),
            required=(adaptive_dir / "classification_dataset.csv", adaptive_dir / "training_summary.json"),
        ),
        Step(
            name="adaptive symbol-level predictions",
            script="pipeline_report_data/plot_symbol_level_predictions.py",
            args=("--input-dir", path_arg(adaptive_dir), "--tickers", args.tickers, "--split", args.split),
            required=(adaptive_dir / "classification_dataset.csv", adaptive_dir / "training_summary.json"),
        ),
        Step(
            name="adaptive latest forecast",
            script="pipeline_report_data/plot_latest_future_forecast.py",
            args=("--input-dir", path_arg(adaptive_dir)),
            required=(adaptive_dir / "latest_forecast.csv", adaptive_dir / "classification_dataset.csv"),
        ),
    ]


def build_steps(args: argparse.Namespace) -> list[Step]:
    steps: list[Step] = []
    if args.mode in {"all", "classification"}:
        steps.extend(classification_steps(args))
    if args.mode in {"all", "adaptive"}:
        steps.extend(adaptive_steps(args))
    return steps


def missing_inputs(step: Step) -> list[Path]:
    return [path for path in step.required if not path.exists()]


def run_step(step: Step, *, strict: bool, dry_run: bool) -> bool:
    missing = missing_inputs(step)
    if missing:
        message = f"Skip {step.name}: missing " + ", ".join(str(path) for path in missing)
        if strict:
            raise FileNotFoundError(message)
        print(message)
        return False

    command = [sys.executable, step.script, *step.args]
    print("Run:", " ".join(command))
    if dry_run:
        return True
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False, env=env)
    if completed.returncode != 0:
        if strict:
            raise RuntimeError(f"Step failed ({completed.returncode}): {step.name}")
        print(f"Step failed ({completed.returncode}): {step.name}")
        return False
    return True


def main() -> None:
    args = parse_args()
    steps = build_steps(args)
    ran = 0
    skipped = 0
    for step in steps:
        if run_step(step, strict=args.strict, dry_run=args.dry_run):
            ran += 1
        else:
            skipped += 1
    print(f"report_steps_ran={ran}, report_steps_skipped={skipped}")


if __name__ == "__main__":
    main()

