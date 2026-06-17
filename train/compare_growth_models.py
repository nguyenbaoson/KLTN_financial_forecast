"""
Compare growth forecasting models with regression and class metrics.

Regression target: target_profit_growth_1q.
Class metrics are computed by binning actual and predicted growth into:
  low: < 0%
  medium: 0% to 10%
  high: > 10%
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from feature_engineering.growth_feature_engineering import GrowthFeatureEngineer, select_feature_columns
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time


CLASS_BINS = [-np.inf, 0.0, 10.0, np.inf]
CLASS_LABELS = ["low", "medium", "high"]


def class_labels(values: np.ndarray) -> np.ndarray:
    return pd.cut(values, bins=CLASS_BINS, labels=CLASS_LABELS).astype(str).to_numpy()


def metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, n_test: int) -> dict[str, Any]:
    true_cls = class_labels(y_true)
    pred_cls = class_labels(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_cls,
        pred_cls,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "accuracy_pct": accuracy_score(true_cls, pred_cls) * 100,
        "precision_pct": precision * 100,
        "recall_pct": recall * 100,
        "f1_score_pct": f1 * 100,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "n_test": n_test,
    }


def sklearn_models() -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=900,
                max_depth=8,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=240,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.5,
                min_samples_leaf=12,
                random_state=42,
            )),
        ]),
        "Bagging": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=42),
                n_estimators=250,
                max_samples=0.80,
                max_features=0.80,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=8, random_state=42)),
        ]),
    }


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_lstm_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_df: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: RobustScaler,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    keyed_split = set(zip(split_df["symbol"], split_df["yq_index"]))
    all_features = pd.DataFrame(
        scaler.transform(imputer.transform(df[feature_cols])),
        columns=feature_cols,
        index=df.index,
    )

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _, group in df.sort_values(["symbol", "yq_index"]).groupby("symbol"):
        group = group.reset_index()
        for pos in range(seq_len - 1, len(group)):
            row = group.iloc[pos]
            key = (row["symbol"], row["yq_index"])
            if key not in keyed_split or pd.isna(row[target_col]):
                continue
            idx = group.iloc[pos - seq_len + 1:pos + 1]["index"].to_numpy()
            xs.append(all_features.loc[idx].to_numpy(dtype=np.float32))
            ys.append(float(row[target_col]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_predict_lstm(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    imputer.fit(train_df[feature_cols])
    scaler.fit(imputer.transform(train_df[feature_cols]))

    sequence_base = dataset.dropna(subset=[target_col]).copy()
    x_train, y_train = build_lstm_arrays(sequence_base, feature_cols, target_col, train_df, imputer, scaler, seq_len)
    x_val, y_val = build_lstm_arrays(sequence_base, feature_cols, target_col, val_df, imputer, scaler, seq_len)
    x_test, y_test = build_lstm_arrays(sequence_base, feature_cols, target_col, test_df, imputer, scaler, seq_len)
    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Not enough rows to train/evaluate LSTM sequences.")

    torch.manual_seed(42)
    model = LSTMRegressor(n_features=len(feature_cols), hidden_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32,
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    patience = 12
    stale = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if len(x_val):
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(torch.from_numpy(x_val)), torch.from_numpy(y_val)).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test)).numpy()
    info = {"seq_len": seq_len, "train_sequences": len(x_train), "test_sequences": len(x_test)}
    return y_test, pred, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare growth models on the ML-tree dataset.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/growth_forecast")
    parser.add_argument("--target", default="target_profit_growth_1q")
    parser.add_argument("--target-clip", type=float, default=300.0)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-symbol-dummies", action="store_true")
    parser.add_argument("--lstm-seq-len", type=int, default=4)
    parser.add_argument("--lstm-epochs", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GrowthFeatureEngineer().build_dataset_from_ml_tree(data_root=args.data_root)
    if dataset.empty:
        raise ValueError("Feature dataset is empty.")
    dataset = dataset.dropna(axis=1, how="all")
    trainable = dataset.dropna(subset=[args.target]).copy()
    if args.target_clip and args.target_clip > 0:
        trainable[args.target] = trainable[args.target].clip(-args.target_clip, args.target_clip)
    if not args.no_symbol_dummies:
        symbol_values = sorted(trainable["symbol"].dropna().astype(str).str.upper().unique().tolist())
        dataset, _ = add_symbol_dummies(dataset, symbol_values)
        trainable, _ = add_symbol_dummies(trainable, symbol_values)

    feature_cols = select_feature_columns(trainable, args.target, args.min_feature_non_null)
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, pruning = prune_feature_columns(train_df, feature_cols, args.max_feature_corr)

    rows = []
    for name, model in sklearn_models().items():
        model.fit(train_df[feature_cols], train_df[args.target])
        pred = model.predict(test_df[feature_cols])
        rows.append(metrics_row(name, test_df[args.target].to_numpy(), pred, len(test_df)))

    lstm_info: dict[str, Any] = {"available": TORCH_AVAILABLE}
    if TORCH_AVAILABLE:
        try:
            y_true, y_pred, lstm_info = fit_predict_lstm(
                dataset=trainable,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col=args.target,
                seq_len=args.lstm_seq_len,
                epochs=args.lstm_epochs,
            )
            rows.append(metrics_row("LSTM", y_true, y_pred, len(y_true)))
            lstm_info["available"] = True
        except Exception as exc:
            rows.append({
                "model": "LSTM",
                "accuracy_pct": np.nan,
                "precision_pct": np.nan,
                "recall_pct": np.nan,
                "f1_score_pct": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "n_test": 0,
            })
            lstm_info = {"available": TORCH_AVAILABLE, "error": str(exc)}

    result = pd.DataFrame(rows)
    result = result[["model", "accuracy_pct", "precision_pct", "recall_pct", "f1_score_pct", "mse", "rmse", "n_test"]]
    result.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    result.to_markdown(output_dir / "model_comparison_metrics.md", index=False, floatfmt=".4f")

    summary = {
        "target": args.target,
        "target_clip": args.target_clip,
        "rows": int(dataset.shape[0]),
        "trainable_rows": int(trainable.shape[0]),
        "symbols": int(dataset["symbol"].nunique()),
        "feature_count": len(feature_cols),
        "split_info": split_info,
        "feature_pruning": pruning,
        "lstm": lstm_info,
        "classification_bins": {"low": "<0", "medium": "0..10", "high": ">10"},
    }
    with open(output_dir / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
