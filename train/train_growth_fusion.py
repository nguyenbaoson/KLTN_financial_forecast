"""
Train a two-stage fusion model for growth direction classification.

Stage 1 trains separate base models on financial, market and news feature groups.
Stage 2 trains a meta classifier on the base-model probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer, select_feature_columns
from train.train_growth_classification import (
    apply_feature_set,
    classification_metrics,
    derive_classification_targets,
    positive_class_index,
    predict_proba_or_none,
    predict_with_threshold,
    tune_binary_threshold,
)
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time


DEFAULT_OUTPUT_DIR = Path("outputs/thesis_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-stage fusion growth classifier.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--target", default="target_profit_up_4q")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.0)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--no-symbol-dummies", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, tickers: List[str] | None) -> pd.DataFrame:
    if args.dataset_path:
        dataset = pd.read_csv(args.dataset_path)
        if tickers:
            dataset = dataset[dataset["symbol"].astype(str).str.upper().isin(tickers)].copy()
        return derive_classification_targets(dataset)
    engineer = GrowthFeatureEngineer()
    return derive_classification_targets(engineer.build_dataset_from_ml_tree(args.data_root, tickers=tickers))


def encode_target(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    encoder = LabelEncoder()
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    train["_target_encoded"] = encoder.fit_transform(train[target_col].astype(str))
    val["_target_encoded"] = encoder.transform(val[target_col].astype(str))
    test["_target_encoded"] = encoder.transform(test[target_col].astype(str))
    return train, val, test, encoder


def base_model_for_group(group_name: str) -> Pipeline:
    if group_name == "news":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=7,
                l2_regularization=0.5,
                random_state=42,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.75,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_meta_frame(
    models: Dict[str, Pipeline],
    feature_groups: Dict[str, List[str]],
    frame: pd.DataFrame,
    positive_idx: int,
) -> pd.DataFrame:
    meta = pd.DataFrame(index=frame.index)
    for group_name, model in models.items():
        proba = predict_proba_or_none(model, frame[feature_groups[group_name]])
        if proba is None:
            pred = model.predict(frame[feature_groups[group_name]])
            meta[f"prob_up_{group_name}"] = pred.astype(float)
        else:
            meta[f"prob_up_{group_name}"] = proba[:, positive_idx]
    return meta


def save_predictions(
    model: Pipeline,
    encoder: LabelEncoder,
    meta_splits: Dict[str, pd.DataFrame],
    source_splits: Dict[str, pd.DataFrame],
    target_col: str,
    output_dir: Path,
    threshold: float | None,
) -> None:
    frames = []
    positive_idx = positive_class_index(encoder)
    for split_name, meta in meta_splits.items():
        source = source_splits[split_name]
        pred = predict_with_threshold(model, meta, threshold, positive_idx)
        proba = predict_proba_or_none(model, meta)
        out = source[["symbol", "year", "quarter", "yq_index", target_col]].copy()
        out["split"] = split_name
        out["actual_label"] = source[target_col].astype(str)
        out["predicted_label"] = encoder.inverse_transform(pred.astype(int))
        if threshold is not None:
            out["decision_threshold"] = threshold
        if proba is not None:
            for idx, label in enumerate(encoder.classes_):
                out[f"prob_{label}"] = proba[:, idx]
            out["prediction_confidence"] = proba.max(axis=1)
        frames.append(out)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()] or None
    dataset = load_dataset(args, tickers)
    if dataset.empty:
        raise ValueError("Feature dataset is empty.")
    if args.target not in dataset.columns:
        raise ValueError(f"Target {args.target!r} does not exist.")

    trainable = dataset.dropna(subset=[args.target]).copy()
    trainable[args.target] = trainable[args.target].astype(str)
    if not args.no_symbol_dummies:
        symbols = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique())
        dataset, _ = add_symbol_dummies(dataset, symbols)
        trainable, _ = add_symbol_dummies(trainable, symbols)

    feature_cols = select_feature_columns(trainable, args.target, min_non_null_ratio=args.min_feature_non_null)
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, feature_pruning = prune_feature_columns(train_df, feature_cols, args.max_feature_corr)

    feature_groups = {
        "financial": apply_feature_set(feature_cols, "financial_only"),
        "market": apply_feature_set(feature_cols, "market_only"),
        "news": apply_feature_set(feature_cols, "news_only"),
    }
    feature_groups = {name: cols for name, cols in feature_groups.items() if len(cols) >= 2}
    if len(feature_groups) < 2:
        raise ValueError(f"Need at least two usable feature groups, got {list(feature_groups)}")

    train_df, val_df, test_df, encoder = encode_target(train_df, val_df, test_df, args.target)
    positive_idx = positive_class_index(encoder)
    base_models: Dict[str, Pipeline] = {}
    base_rows: List[Dict[str, Any]] = []
    for group_name, cols in feature_groups.items():
        model = base_model_for_group(group_name)
        model.fit(train_df[cols], train_df["_target_encoded"].values)
        base_models[group_name] = model
        for split_name, frame in {"validation": val_df, "test": test_df}.items():
            pred = model.predict(frame[cols])
            proba = predict_proba_or_none(model, frame[cols])
            metrics = classification_metrics(frame["_target_encoded"].values, pred, proba, len(encoder.classes_))
            base_rows.append({
                "stage": "base",
                "group": group_name,
                "split": split_name,
                "n_features": len(cols),
                **metrics,
            })
        joblib.dump(model, output_dir / f"base_{group_name}.joblib")

    meta_train = make_meta_frame(base_models, feature_groups, train_df, positive_idx)
    meta_val = make_meta_frame(base_models, feature_groups, val_df, positive_idx)
    meta_test = make_meta_frame(base_models, feature_groups, test_df, positive_idx)

    meta_model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)),
    ])
    meta_model.fit(meta_train, train_df["_target_encoded"].values)
    threshold = None
    threshold_score = None
    if args.tune_threshold and len(encoder.classes_) == 2:
        threshold, threshold_score = tune_binary_threshold(meta_model, val_df.assign(**meta_val), list(meta_val.columns), positive_idx)

    result_rows = []
    for split_name, meta, frame in [("validation", meta_val, val_df), ("test", meta_test, test_df)]:
        pred = predict_with_threshold(meta_model, meta, threshold, positive_idx)
        proba = predict_proba_or_none(meta_model, meta)
        metrics = classification_metrics(frame["_target_encoded"].values, pred, proba, len(encoder.classes_))
        result_rows.append({
            "stage": "fusion",
            "group": "meta",
            "split": split_name,
            "n_features": meta.shape[1],
            **metrics,
        })

    pd.DataFrame([*base_rows, *result_rows]).to_csv(output_dir / "model_results.csv", index=False)
    save_predictions(
        meta_model,
        encoder,
        {"train": meta_train, "validation": meta_val, "test": meta_test},
        {"train": train_df, "validation": val_df, "test": test_df},
        args.target,
        output_dir,
        threshold,
    )
    joblib.dump(meta_model, output_dir / "fusion_meta_model.joblib")
    dataset.to_csv(output_dir / "fusion_dataset.csv", index=False)

    summary = {
        "task_type": "fusion_classification",
        "target_col": args.target,
        "classes": encoder.classes_.tolist(),
        "n_rows": int(dataset.shape[0]),
        "n_trainable_rows": int(trainable.shape[0]),
        "n_tickers": int(dataset["symbol"].nunique()),
        "feature_group_counts": {name: len(cols) for name, cols in feature_groups.items()},
        "meta_features": list(meta_train.columns),
        "decision_threshold": threshold,
        "threshold_tuning_score": threshold_score,
        "feature_pruning": feature_pruning,
        "split_info": split_info,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(pd.DataFrame([*base_rows, *result_rows]))
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
