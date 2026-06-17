"""Train a lightweight quarterly sequence model for growth classification.

This is intentionally smaller than TFT/TCN because the project has only a few
dozen quarterly timestamps. It uses a GRU/LSTM over per-symbol feature
histories and keeps the same time-based split/evaluation files as the tabular
classifier.

Example:
  python -m train.train_growth_sequence_model --dataset-path data/features/growth_features.csv --target target_strong_profit_up_4q
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.growth_feature_engineering import select_feature_columns  # noqa: E402
from train.train_growth_classification import (  # noqa: E402
    add_industry_metadata,
    add_macro_regime_features,
    add_market_index_features,
    add_panel_cycle_features,
    apply_feature_set,
    derive_classification_targets,
    select_top_k_features_by_mi,
)
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time  # noqa: E402


class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, model_type: str) -> None:
        super().__init__()
        recurrent_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quarterly sequence growth classifier.")
    parser.add_argument("--dataset-path", default="data/features/growth_features.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/strong_4q_sequence_gru")
    parser.add_argument("--target", default="target_strong_profit_up_4q")
    parser.add_argument("--feature-set", choices=["all", "no_news", "news_only", "financial_only", "market_only"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--top-k-features", type=int, default=120)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--no-symbol-dummies", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--add-panel-cycle-features", action="store_true")
    parser.add_argument("--add-market-index-features", action="store_true")
    parser.add_argument("--add-industry-cycle-features", action="store_true")
    parser.add_argument("--add-macro-regime-features", action="store_true")
    parser.add_argument("--add-target-cycle-lag-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, proba)),
    }
    try:
        metrics["AUC"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["AUC"] = float("nan")
    return metrics


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dataset = pd.read_csv(args.dataset_path)
    dataset = derive_classification_targets(dataset)
    if args.target not in dataset.columns:
        raise ValueError(f"Target {args.target!r} does not exist.")

    if args.add_industry_cycle_features:
        dataset = add_industry_metadata(dataset, args.data_root)
    if args.add_market_index_features:
        dataset = add_market_index_features(dataset, args.data_root)
    if args.add_macro_regime_features:
        dataset = add_macro_regime_features(dataset)
    if args.add_panel_cycle_features or args.add_target_cycle_lag_features:
        dataset = add_panel_cycle_features(dataset, args.target, add_target_lags=args.add_target_cycle_lag_features)

    trainable = dataset.dropna(subset=[args.target]).copy()
    trainable[args.target] = pd.to_numeric(trainable[args.target], errors="coerce")
    trainable = trainable.dropna(subset=[args.target]).copy()

    symbol_dummy_cols: list[str] = []
    if not args.no_symbol_dummies:
        symbols = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique().tolist())
        trainable, symbol_dummy_cols = add_symbol_dummies(trainable, symbols)

    feature_cols = select_feature_columns(trainable, args.target, min_non_null_ratio=args.min_feature_non_null)
    feature_cols = apply_feature_set(feature_cols, args.feature_set)
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, pruning = prune_feature_columns(train_df, feature_cols, max_corr=args.max_feature_corr)

    metadata = {
        "n_rows": int(dataset.shape[0]),
        "n_trainable_rows": int(trainable.shape[0]),
        "n_tickers": int(trainable["symbol"].nunique()),
        "symbol_dummy_count": len(symbol_dummy_cols),
        "split_info": split_info,
        "feature_pruning": pruning,
    }
    return trainable, feature_cols, metadata


def build_sequences(
    frame: pd.DataFrame,
    transformed_features: pd.DataFrame,
    rows: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int,
    min_history: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    kept_rows: list[pd.Series] = []

    frame = frame.sort_values(["symbol", "yq_index"]).copy()
    transformed_features = transformed_features.loc[frame.index]
    row_keys = set(rows.index.tolist())

    for _, symbol_frame in frame.groupby("symbol", sort=False):
        indices = symbol_frame.index.tolist()
        for position, row_idx in enumerate(indices):
            if row_idx not in row_keys:
                continue
            start = max(0, position - sequence_length + 1)
            history_indices = indices[start:position + 1]
            if len(history_indices) < min_history:
                continue
            seq = transformed_features.loc[history_indices, feature_cols].to_numpy(dtype=np.float32)
            if len(seq) < sequence_length:
                pad = np.zeros((sequence_length - len(seq), len(feature_cols)), dtype=np.float32)
                seq = np.vstack([pad, seq])
            sequences.append(seq)
            labels.append(int(float(frame.loc[row_idx, target_col])))
            kept_rows.append(frame.loc[row_idx])

    if not sequences:
        return np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32), np.array([], dtype=np.int64), pd.DataFrame()
    return np.stack(sequences), np.asarray(labels, dtype=np.int64), pd.DataFrame(kept_rows)


def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def save_predictions(
    output_dir: Path,
    target_col: str,
    split_rows: dict[str, pd.DataFrame],
    y_true: dict[str, np.ndarray],
    proba: dict[str, np.ndarray],
    threshold: float,
) -> None:
    frames = []
    for split_name, rows in split_rows.items():
        pred = (proba[split_name] >= threshold).astype(int)
        out = rows[["symbol", "year", "quarter", "yq_index"]].copy()
        out[target_col] = y_true[split_name].astype(float)
        out["split"] = split_name
        out["actual_label"] = y_true[split_name].astype(float).astype(str)
        out["predicted_label"] = pred.astype(float).astype(str)
        out["prob_0.0"] = 1 - proba[split_name]
        out["prob_1.0"] = proba[split_name]
        out["prediction_confidence"] = np.maximum(out["prob_0.0"], out["prob_1.0"])
        out["decision_threshold"] = threshold
        frames.append(out)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


def save_confusion(output_dir: Path, y_true: dict[str, np.ndarray], proba: dict[str, np.ndarray], threshold: float) -> None:
    rows = []
    for split_name, labels in y_true.items():
        pred = (proba[split_name] >= threshold).astype(int)
        matrix = confusion_matrix(labels, pred, labels=[0, 1])
        for actual_idx, actual_label in enumerate(["0.0", "1.0"]):
            for pred_idx, pred_label in enumerate(["0.0", "1.0"]):
                rows.append({
                    "split": split_name,
                    "actual": actual_label,
                    "predicted": pred_label,
                    "count": int(matrix[actual_idx, pred_idx]),
                })
    pd.DataFrame(rows).to_csv(output_dir / "confusion_matrix.csv", index=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainable, feature_cols, metadata = prepare_dataset(args)
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    metadata["split_info"] = split_info

    if args.top_k_features > 0:
        temp_train = train_df.copy()
        temp_train["_target_encoded"] = temp_train[args.target].astype(int)
        feature_cols, mi_ranking = select_top_k_features_by_mi(temp_train, feature_cols, args.top_k_features, output_dir)
        metadata["feature_selection"] = {
            "method": "mutual_info_classif",
            "top_k": args.top_k_features,
            "n_features_before_top_k": int(mi_ranking.shape[0]),
            "n_features_after_top_k": len(feature_cols),
        }

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler()
    imputer.fit(train_df[feature_cols])
    scaler.fit(imputer.transform(train_df[feature_cols]))

    transformed = pd.DataFrame(
        scaler.transform(imputer.transform(trainable[feature_cols])),
        index=trainable.index,
        columns=feature_cols,
    )

    split_rows = {"train": train_df, "validation": val_df, "test": test_df}
    X: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    kept_rows: dict[str, pd.DataFrame] = {}
    for split_name, rows in split_rows.items():
        X[split_name], y[split_name], kept_rows[split_name] = build_sequences(
            trainable,
            transformed,
            rows,
            feature_cols,
            args.target,
            args.sequence_length,
            args.min_history,
        )
        if X[split_name].shape[0] == 0:
            raise ValueError(f"No usable sequences for {split_name}. Lower --min-history.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceClassifier(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)

    train_dataset = TensorDataset(torch.tensor(X["train"], dtype=torch.float32), torch.tensor(y["train"], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    positive = max(float(y["train"].sum()), 1.0)
    negative = max(float(len(y["train"]) - y["train"].sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -np.inf
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_proba = predict_proba(model, X["validation"], device, args.batch_size)
        val_pred = (val_proba >= 0.5).astype(int)
        val_f1 = f1_score(y["validation"], val_pred, zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_F1": float(val_f1)})
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    proba = {split_name: predict_proba(model, X[split_name], device, args.batch_size) for split_name in X}
    threshold = 0.5
    threshold_score = None
    if args.tune_threshold:
        threshold, threshold_score = tune_threshold(y["validation"], proba["validation"])

    rows = []
    for split_name in ["validation", "test"]:
        pred = (proba[split_name] >= threshold).astype(int)
        metrics = classification_metrics(y[split_name], pred, proba[split_name])
        rows.append({
            "model": args.model_type.upper(),
            "split": split_name,
            "decision_threshold": threshold,
            **metrics,
        })
    results = pd.DataFrame(rows)
    wide = {
        "model": args.model_type.upper(),
        "decision_threshold": threshold,
    }
    for _, row in results.iterrows():
        split_name = row["split"]
        for key, value in row.items():
            if key in {"model", "split", "decision_threshold"}:
                continue
            wide[f"{split_name}_{key}"] = value
    pd.DataFrame([wide]).to_csv(output_dir / "selected_model_results.csv", index=False)
    pd.DataFrame([wide]).rename(columns=lambda col: col.replace("validation_", "val_")).to_csv(output_dir / "model_results.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_predictions(output_dir, args.target, kept_rows, y, proba, threshold)
    save_confusion(output_dir, y, proba, threshold)

    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
        "args": vars(args),
    }, output_dir / "sequence_model.pt")
    joblib.dump({"feature_cols": feature_cols, "imputer": imputer, "scaler": scaler}, output_dir / "sequence_preprocess.joblib")

    summary = {
        **metadata,
        "model": args.model_type.upper(),
        "target_col": args.target,
        "n_features": len(feature_cols),
        "sequence_length": args.sequence_length,
        "min_history": args.min_history,
        "device": str(device),
        "best_val_F1_at_threshold_0_5": best_val_f1,
        "decision_threshold": threshold,
        "threshold_tuning_score": threshold_score,
        "add_panel_cycle_features": bool(args.add_panel_cycle_features),
        "add_market_index_features": bool(args.add_market_index_features),
        "add_industry_cycle_features": bool(args.add_industry_cycle_features),
        "add_macro_regime_features": bool(args.add_macro_regime_features),
        "add_target_cycle_lag_features": bool(args.add_target_cycle_lag_features),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(pd.DataFrame([wide]))
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
