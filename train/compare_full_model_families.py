"""Compare classical, ML, and sequence models on profit-growth forecasting.

The experiment uses the full feature dataset, including ROA/ROE fields, and
predicts a continuous profit-growth target. For thesis readability it also
reports a derived strong-growth classification score by thresholding predicted
growth at 10%.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from feature_engineering.growth_feature_engineering import select_feature_columns
from train.train_growth_classification import derive_classification_targets
from train.train_growth_models import add_symbol_dummies, prune_feature_columns, split_by_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full model families with ROA/ROE features.")
    parser.add_argument("--dataset-path", default="data/features/growth_features.csv")
    parser.add_argument("--output-dir", default="outputs/full_model_comparison_roa_roe")
    parser.add_argument("--target", default="target_profit_growth_1q")
    parser.add_argument("--strong-label", default="target_strong_profit_up_1q")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-feature-non-null", type=float, default=0.15)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--target-clip", type=float, default=300.0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--top-k-sequence-features", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=300.0, neginf=-300.0)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def strong_metrics(df: pd.DataFrame, pred_growth: np.ndarray, strong_label: str) -> dict[str, float]:
    if strong_label not in df.columns:
        return {}
    labels = pd.to_numeric(df[strong_label], errors="coerce")
    mask = labels.notna().to_numpy()
    if mask.sum() == 0:
        return {}
    y_true = labels.to_numpy()[mask].astype(int)
    y_score = np.nan_to_num(pred_growth[mask], nan=0.0, posinf=300.0, neginf=-300.0)
    y_pred = (y_score > 10.0).astype(int)
    metrics = {
        "StrongAccuracy": float(accuracy_score(y_true, y_pred)),
        "StrongBalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "StrongPrecision": float(precision_score(y_true, y_pred, zero_division=0)),
        "StrongRecall": float(recall_score(y_true, y_pred, zero_division=0)),
        "StrongF1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["StrongAUC"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["StrongAUC"] = float("nan")
    return metrics


def top_k_features_by_mi(train_df: pd.DataFrame, feature_cols: list[str], target: str, top_k: int) -> list[str]:
    if top_k <= 0 or len(feature_cols) <= top_k:
        return feature_cols
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(train_df[feature_cols])
    y = train_df[target].to_numpy()
    scores = mutual_info_regression(x, y, random_state=42)
    ranking = pd.DataFrame({"feature": feature_cols, "score": scores}).sort_values("score", ascending=False)
    return ranking.head(top_k)["feature"].tolist()


def build_tabular_models() -> dict[str, Pipeline]:
    linear_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    return {
        "LinearRegression": Pipeline([*linear_steps, ("model", LinearRegression())]),
        "Ridge": Pipeline([*linear_steps, ("model", Ridge(alpha=100.0))]),
        "Lasso": Pipeline([*linear_steps, ("model", Lasso(alpha=0.01, max_iter=10000, random_state=42))]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=700,
                max_depth=8,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=450,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=5,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
        "SVR": Pipeline([*linear_steps, ("model", SVR(C=10.0, gamma="scale", epsilon=0.1, kernel="rbf"))]),
    }


def fit_tabular_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    strong_label: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_train, y_train = train_df[feature_cols], train_df[target].to_numpy()
    x_val, y_val = val_df[feature_cols], val_df[target].to_numpy()
    x_test, y_test = test_df[feature_cols], test_df[target].to_numpy()
    for name, model in build_tabular_models().items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        joblib.dump(model, output_dir / f"{name.lower()}_regressor.joblib")
        row = {"model": name, "family": "tabular_ml", **{f"val_{k}": v for k, v in regression_metrics(y_val, val_pred).items()}, **{f"test_{k}": v for k, v in regression_metrics(y_test, test_pred).items()}}
        row.update({f"test_{k}": v for k, v in strong_metrics(test_df, test_pred, strong_label).items()})
        rows.append(row)
    return rows


def _fit_sarimax_forecast(series: pd.Series, steps: int, seasonal: bool) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    fallback = float(np.nanmean(values)) if len(values) else 0.0
    if len(values) < 8:
        return np.repeat(fallback, steps)
    try:
        order = (1, 0, 0)
        seasonal_order = (1, 0, 0, 4) if seasonal and len(values) >= 12 else (0, 0, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = SARIMAX(values, order=order, seasonal_order=seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=80)
        forecast = np.asarray(fitted.forecast(steps=steps), dtype=float)
        return np.nan_to_num(forecast, nan=fallback, posinf=fallback, neginf=fallback)
    except Exception:
        return np.repeat(fallback, steps)


def fit_arima_family(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    strong_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for name, seasonal in [("ARIMA", False), ("SARIMA", True)]:
        val_preds = []
        test_preds = []
        for symbol, train_symbol in train_df.groupby("symbol"):
            val_symbol = val_df[val_df["symbol"] == symbol].sort_values("yq_index")
            test_symbol = test_df[test_df["symbol"] == symbol].sort_values("yq_index")
            train_series = train_symbol.sort_values("yq_index")[target]
            forecast = _fit_sarimax_forecast(train_series, len(val_symbol) + len(test_symbol), seasonal)
            val_preds.append(pd.Series(forecast[:len(val_symbol)], index=val_symbol.index))
            extended = pd.concat([train_series, val_symbol[target]], ignore_index=True)
            test_forecast = _fit_sarimax_forecast(extended, len(test_symbol), seasonal)
            test_preds.append(pd.Series(test_forecast, index=test_symbol.index))
        val_pred = pd.concat(val_preds).reindex(val_df.index).to_numpy()
        test_pred = pd.concat(test_preds).reindex(test_df.index).to_numpy()
        row = {"model": name, "family": "classical_time_series", **{f"val_{k}": v for k, v in regression_metrics(val_df[target].to_numpy(), val_pred).items()}, **{f"test_{k}": v for k, v in regression_metrics(test_df[target].to_numpy(), test_pred).items()}}
        row.update({f"test_{k}": v for k, v in strong_metrics(test_df, test_pred, strong_label).items()})
        rows.append(row)
    return rows


def fit_var_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    strong_label: str,
) -> list[dict[str, Any]]:
    var_cols = [target, "revenue_g1q", "net_profit_g1q", "q_return", "roa", "roe", "debt_to_asset_calc"]
    val_preds = []
    test_preds = []
    for symbol, train_symbol in train_df.groupby("symbol"):
        val_symbol = val_df[val_df["symbol"] == symbol].sort_values("yq_index")
        test_symbol = test_df[test_df["symbol"] == symbol].sort_values("yq_index")
        cols = [col for col in var_cols if col in train_symbol.columns]
        history = train_symbol.sort_values("yq_index")[cols].apply(pd.to_numeric, errors="coerce")
        history = history.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        if len(history) < 10 or target not in cols:
            mean_value = float(train_symbol[target].mean())
            val_preds.append(pd.Series(np.repeat(mean_value, len(val_symbol)), index=val_symbol.index))
            test_preds.append(pd.Series(np.repeat(mean_value, len(test_symbol)), index=test_symbol.index))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VAR(history).fit(maxlags=1, trend="c")
            val_forecast = model.forecast(history.values[-model.k_ar:], steps=len(val_symbol))
            val_preds.append(pd.Series(val_forecast[:, cols.index(target)], index=val_symbol.index))
            extended = pd.concat([history, val_symbol[cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)])
            model = VAR(extended).fit(maxlags=1, trend="c")
            test_forecast = model.forecast(extended.values[-model.k_ar:], steps=len(test_symbol))
            test_preds.append(pd.Series(test_forecast[:, cols.index(target)], index=test_symbol.index))
        except Exception:
            mean_value = float(train_symbol[target].mean())
            val_preds.append(pd.Series(np.repeat(mean_value, len(val_symbol)), index=val_symbol.index))
            test_preds.append(pd.Series(np.repeat(mean_value, len(test_symbol)), index=test_symbol.index))
    val_pred = pd.concat(val_preds).reindex(val_df.index).to_numpy()
    test_pred = pd.concat(test_preds).reindex(test_df.index).to_numpy()
    row = {"model": "VAR", "family": "classical_time_series", **{f"val_{k}": v for k, v in regression_metrics(val_df[target].to_numpy(), val_pred).items()}, **{f"test_{k}": v for k, v in regression_metrics(test_df[target].to_numpy(), test_pred).items()}}
    row.update({f"test_{k}": v for k, v in strong_metrics(test_df, test_pred, strong_label).items()})
    return [row]


class SequenceRegressor(nn.Module):
    def __init__(self, input_size: int, model_type: str, hidden_size: int = 64, dropout: float = 0.2, ratio_size: int = 0) -> None:
        super().__init__()
        self.model_type = model_type
        self.ratio_size = ratio_size
        if model_type == "transformer":
            self.input_projection = nn.Linear(input_size, hidden_size)
            layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            recurrent_cls = nn.LSTM if model_type in {"lstm", "hybrid_lstm_ratios"} else nn.GRU
            self.encoder = recurrent_cls(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        head_in = hidden_size + ratio_size
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout), nn.Linear(head_in, 1))

    def forward(self, x: torch.Tensor, ratios: torch.Tensor | None = None) -> torch.Tensor:
        if self.model_type == "transformer":
            encoded = self.encoder(self.input_projection(x))[:, -1, :]
        else:
            output, _ = self.encoder(x)
            encoded = output[:, -1, :]
        if self.ratio_size and ratios is not None:
            encoded = torch.cat([encoded, ratios], dim=1)
        return self.head(encoded).squeeze(-1)


def make_sequences(df: pd.DataFrame, feature_cols: list[str], target: str, sequence_length: int, min_history: int, ratio_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ratios, ys, indices = [], [], [], []
    for _, group in df.sort_values(["symbol", "yq_index"]).groupby("symbol"):
        values = group[feature_cols].to_numpy(dtype=float)
        ratio_values = group[ratio_cols].to_numpy(dtype=float) if ratio_cols else np.zeros((len(group), 0))
        targets = group[target].to_numpy(dtype=float)
        idx = group.index.to_numpy()
        for pos in range(len(group)):
            start = max(0, pos - sequence_length + 1)
            seq = values[start:pos + 1]
            if len(seq) < min_history:
                continue
            if len(seq) < sequence_length:
                pad = np.repeat(seq[:1], sequence_length - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            xs.append(seq)
            ratios.append(ratio_values[pos])
            ys.append(targets[pos])
            indices.append(idx[pos])
    return np.asarray(xs, dtype=np.float32), np.asarray(ratios, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(indices)


def train_sequence_model(name: str, model_type: str, train_data: tuple[np.ndarray, np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray, np.ndarray], args: argparse.Namespace, ratio_size: int) -> SequenceRegressor:
    torch.manual_seed(args.seed)
    x_train, r_train, y_train = train_data
    x_val, r_val, y_val = val_data
    model = SequenceRegressor(x_train.shape[-1], model_type=model_type, ratio_size=ratio_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=25.0)
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(r_train), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    for _ in range(args.epochs):
        model.train()
        for xb, rb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb, rb if ratio_size else None)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val), torch.tensor(r_val) if ratio_size else None)
            val_loss = float(loss_fn(val_pred, torch.tensor(y_val)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def fit_sequence_family(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    strong_label: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(imp.fit_transform(train_df[feature_cols]))
    val_scaled[feature_cols] = scaler.transform(imp.transform(val_df[feature_cols]))
    test_scaled[feature_cols] = scaler.transform(imp.transform(test_df[feature_cols]))
    ratio_cols = [col for col in ["roa", "roe", "roa_calc", "roe_calc", "debt_to_asset_calc", "net_profit_margin_calc"] if col in feature_cols]
    rows = []
    seq_train = make_sequences(train_scaled, feature_cols, target, args.sequence_length, args.min_history, ratio_cols)
    seq_val = make_sequences(val_scaled, feature_cols, target, args.sequence_length, args.min_history, ratio_cols)
    seq_test = make_sequences(test_scaled, feature_cols, target, args.sequence_length, args.min_history, ratio_cols)
    for model_name, model_type, use_ratios in [
        ("LSTM", "lstm", False),
        ("GRU", "gru", False),
        ("TransformerTimeSeries", "transformer", False),
        ("Hybrid_LSTM_FinancialRatios", "hybrid_lstm_ratios", True),
    ]:
        ratio_size = len(ratio_cols) if use_ratios else 0
        model = train_sequence_model(model_name, model_type, (seq_train[0], seq_train[1] if use_ratios else np.zeros((len(seq_train[0]), 0), dtype=np.float32), seq_train[2]), (seq_val[0], seq_val[1] if use_ratios else np.zeros((len(seq_val[0]), 0), dtype=np.float32), seq_val[2]), args, ratio_size)
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(seq_val[0]), torch.tensor(seq_val[1]) if use_ratios else None).numpy()
            test_pred = model(torch.tensor(seq_test[0]), torch.tensor(seq_test[1]) if use_ratios else None).numpy()
        torch.save(model.state_dict(), output_dir / f"{model_name.lower()}_regressor.pt")
        val_eval = val_df.loc[seq_val[3]].copy()
        test_eval = test_df.loc[seq_test[3]].copy()
        row = {"model": model_name, "family": "neural_sequence", **{f"val_{k}": v for k, v in regression_metrics(seq_val[2], val_pred).items()}, **{f"test_{k}": v for k, v in regression_metrics(seq_test[2], test_pred).items()}}
        row.update({f"test_{k}": v for k, v in strong_metrics(test_eval, test_pred, strong_label).items()})
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset_path)
    dataset = derive_classification_targets(dataset)
    trainable = dataset.dropna(subset=[args.target]).copy()
    trainable[args.target] = pd.to_numeric(trainable[args.target], errors="coerce")
    if args.target_clip > 0:
        trainable[args.target] = trainable[args.target].clip(-args.target_clip, args.target_clip)
    trainable, symbol_dummy_cols = add_symbol_dummies(trainable)
    feature_cols = select_feature_columns(trainable, args.target, args.min_feature_non_null)
    feature_cols = [col for col in feature_cols if col != args.strong_label]
    train_df, val_df, test_df, split_info = split_by_time(trainable, args.train_ratio, args.val_ratio)
    feature_cols, pruning_info = prune_feature_columns(train_df, feature_cols, args.max_feature_corr)
    sequence_features = top_k_features_by_mi(train_df, feature_cols, args.target, args.top_k_sequence_features)

    rows: list[dict[str, Any]] = []
    rows.extend(fit_arima_family(train_df, val_df, test_df, args.target, args.strong_label))
    rows.extend(fit_var_model(train_df, val_df, test_df, args.target, args.strong_label))
    rows.extend(fit_tabular_models(train_df, val_df, test_df, feature_cols, args.target, args.strong_label, output_dir))
    rows.extend(fit_sequence_family(train_df, val_df, test_df, sequence_features, args.target, args.strong_label, args, output_dir))

    results = pd.DataFrame(rows)
    results = results.sort_values(["test_StrongF1", "test_StrongBalancedAccuracy", "test_RMSE"], ascending=[False, False, True])
    results.to_csv(output_dir / "full_model_comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dataset_path": args.dataset_path,
        "target": args.target,
        "strong_label": args.strong_label,
        "rows": int(len(dataset)),
        "trainable_rows": int(len(trainable)),
        "feature_count": int(len(feature_cols)),
        "sequence_feature_count": int(len(sequence_features)),
        "roa_roe_features_present": [col for col in ["roa", "roe", "roa_calc", "roe_calc", "roa_ttm_calc", "roe_ttm_calc"] if col in feature_cols],
        "split_info": split_info,
        "feature_pruning": pruning_info,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
