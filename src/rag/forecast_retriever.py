from __future__ import annotations

import os
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_DIR = ROOT_DIR / "outputs" / "adaptive_compare_models_1q" / "s80_q10_p10"


@dataclass(frozen=True)
class ForecastRecord:
    source_id: str
    symbol: str
    year: int
    quarter: int
    yq_index: int
    target_year: int | None
    target_quarter: int | None
    target_yq_index: int | None
    horizon_quarters: int | None
    target_col: str | None
    predicted_label: int | None
    prob_strong_growth: float | None
    decision_threshold: float | None
    confidence: float | None
    model_name: str | None = None


def _resolve_forecast_dir(forecast_dir: str | Path | None = None) -> Path:
    configured = forecast_dir or os.getenv("FORECAST_OUTPUT_DIR")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT_DIR / path

    if (DEFAULT_FORECAST_DIR / "latest_forecast.csv").exists():
        return DEFAULT_FORECAST_DIR

    candidates = sorted(
        (p.parent for p in (ROOT_DIR / "outputs").rglob("latest_forecast.csv")),
        key=lambda p: (p / "latest_forecast.csv").stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else DEFAULT_FORECAST_DIR


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _add_quarters(year: int, quarter: int, horizon: int) -> tuple[int, int, int]:
    zero_based = (year * 4 + (quarter - 1)) + horizon
    target_year = zero_based // 4
    target_quarter = zero_based % 4 + 1
    return target_year, target_quarter, target_year * 10 + target_quarter


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _detect_periods(query: str) -> set[int]:
    text = _normalize_text(query)
    periods: set[int] = set()
    for year, quarter in re.findall(r"(20\d{2})\s*[-/]?\s*q([1-4])", text):
        periods.add(int(year) * 10 + int(quarter))
    for quarter, year in re.findall(r"q([1-4])\s*[-/]?\s*(20\d{2})", text):
        periods.add(int(year) * 10 + int(quarter))
    for quarter, year in re.findall(r"quy\s*([1-4])\D{0,12}(20\d{2})", text):
        periods.add(int(year) * 10 + int(quarter))
    for yq in re.findall(r"\b(20\d{2}[1-4])\b", text):
        periods.add(int(yq))
    return periods


class ForecastRetriever:
    """Reads the latest trained classifier output for chatbot grounding."""

    def __init__(self, forecast_dir: str | Path | None = None):
        self.forecast_dir = _resolve_forecast_dir(forecast_dir)
        self.forecast_path = self.forecast_dir / "latest_forecast.csv"
        self.results_path = self.forecast_dir / "model_results.csv"
        self.training_summary_path = self.forecast_dir / "training_summary.json"
        self._forecast_df: pd.DataFrame | None = None
        self._results_df: pd.DataFrame | None = None
        self._training_summary: dict | None = None
        self._symbols: set[str] = set()

    @property
    def forecast_df(self) -> pd.DataFrame:
        if self._forecast_df is None:
            if not self.forecast_path.exists():
                self._forecast_df = pd.DataFrame()
                self._symbols = set()
            else:
                df = pd.read_csv(self.forecast_path)
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper()
                    self._symbols = set(df["symbol"].dropna().unique().tolist())
                self._forecast_df = df
        return self._forecast_df

    @property
    def results_df(self) -> pd.DataFrame:
        if self._results_df is None:
            self._results_df = pd.read_csv(self.results_path) if self.results_path.exists() else pd.DataFrame()
        return self._results_df

    @property
    def training_summary(self) -> dict:
        if self._training_summary is None:
            if not self.training_summary_path.exists():
                self._training_summary = {}
            else:
                with self.training_summary_path.open("r", encoding="utf-8") as f:
                    self._training_summary = json.load(f)
        return self._training_summary

    @property
    def symbols(self) -> set[str]:
        _ = self.forecast_df
        return set(self._symbols)

    def detect_symbols(self, query: str, ticker: str | None = None) -> list[str]:
        if ticker:
            return [ticker.upper()]
        text = query.upper()
        hits = [
            symbol
            for symbol in self.symbols
            if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", text)
        ]
        return sorted(hits)

    def best_model_name(self) -> str | None:
        if self.training_summary.get("best_model"):
            return str(self.training_summary["best_model"])
        df = self.results_df
        if df.empty or "model" not in df.columns:
            return None
        metric = "test_AUC" if "test_AUC" in df.columns else "test_BalancedAccuracy"
        if metric not in df.columns:
            return str(df.iloc[0]["model"])
        ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if ranked.empty:
            return str(df.iloc[0]["model"])
        return str(ranked.iloc[0]["model"])

    def target_col(self) -> str | None:
        value = self.training_summary.get("target_col") or self.training_summary.get("target")
        return str(value) if value else None

    def horizon_quarters(self) -> int | None:
        target_col = self.target_col() or ""
        match = re.search(r"_(\d+)q$", target_col)
        if match:
            return int(match.group(1))
        return None

    def retrieve(self, query: str, ticker: str | None = None, top_k: int = 3) -> list[ForecastRecord]:
        df = self.forecast_df
        required = {"symbol", "year", "quarter", "yq_index"}
        if df.empty or not required.issubset(df.columns):
            return []

        symbols = self.detect_symbols(query, ticker)
        if not symbols:
            return []

        work = df[df["symbol"].isin(symbols)].copy()
        if work.empty:
            return []

        model_name = self.best_model_name()
        target_col = self.target_col()
        horizon = self.horizon_quarters()
        records = []
        for _, row in work.sort_values(["symbol", "yq_index"], ascending=[True, False]).groupby("symbol").head(top_k).iterrows():
            label = _safe_int(row.get("predicted_label"))
            year = int(row["year"])
            quarter = int(row["quarter"])
            if horizon is None:
                target_year = target_quarter = target_yq_index = None
            else:
                target_year, target_quarter, target_yq_index = _add_quarters(year, quarter, horizon)
            records.append(
                ForecastRecord(
                    source_id=f"Forecast {len(records) + 1}",
                    symbol=str(row["symbol"]).upper(),
                    year=year,
                    quarter=quarter,
                    yq_index=int(row["yq_index"]),
                    target_year=target_year,
                    target_quarter=target_quarter,
                    target_yq_index=target_yq_index,
                    horizon_quarters=horizon,
                    target_col=target_col,
                    predicted_label=label,
                    prob_strong_growth=_safe_float(row.get("prob_1.0", row.get("prob_1"))),
                    decision_threshold=_safe_float(row.get("decision_threshold")),
                    confidence=_safe_float(row.get("prediction_confidence")),
                    model_name=model_name,
                )
            )
        requested_periods = _detect_periods(query)
        if requested_periods:
            target_matches = [
                record for record in records
                if record.target_yq_index is not None and record.target_yq_index in requested_periods
            ]
            if target_matches:
                return target_matches
        return records

    def format_context(self, records: Iterable[ForecastRecord]) -> str:
        blocks = []
        adaptive_info = self.training_summary.get("adaptive_target_info", {})
        profit_source = adaptive_info.get("profit_target_source") or self.training_summary.get("profit_target_source") or "N/A"
        for record in records:
            if record.target_year is not None and record.target_quarter is not None:
                target_period = f"{record.target_year}Q{record.target_quarter} ({record.target_yq_index})"
            else:
                target_period = "N/A"
            label_text = (
                "Có khả năng thuộc nhóm tăng trưởng lợi nhuận mạnh"
                if record.predicted_label == 1
                else "Chưa thuộc nhóm tăng trưởng lợi nhuận mạnh"
                if record.predicted_label == 0
                else "N/A"
            )
            blocks.append(
                f"[{record.source_id}]\n"
                f"Loại nguồn: Latest model forecast\n"
                f"Mã: {record.symbol}\n"
                f"Kỳ dự báo có sẵn: {target_period}\n"
                f"Kỳ mục tiêu dự báo: {target_period}\n"
                f"Kỳ dữ liệu đầu vào mới nhất dùng để tạo dự báo: {record.year}Q{record.quarter} ({record.yq_index})\n"
                f"Cột target mô hình: {record.target_col or 'N/A'}\n"
                f"Nhãn dự báo cho kỳ mục tiêu: {label_text}\n"
                f"Cơ sở target lợi nhuận: {profit_source}\n"
                f"Xác suất tăng trưởng mạnh: {_format_pct(record.prob_strong_growth)}\n"
                f"Ngưỡng quyết định xác suất: {_format_pct(record.decision_threshold)}\n"
                f"Độ tin cậy dự báo: {_format_pct(record.confidence)}\n"
                f"Mô hình tham chiếu: {record.model_name or 'N/A'}\n"
                f"Lưu ý diễn giải: nguồn này chỉ dự báo nhãn/xác suất tăng trưởng mạnh cho kỳ mục tiêu; "
                f"không dự báo giá trị tuyệt đối như doanh thu hoặc net_profit_parent. "
                f"Ngưỡng quyết định là ngưỡng xác suất, không phải ngưỡng phần trăm tăng trưởng lợi nhuận.\n"
                f"Nguồn dữ liệu: {self.forecast_path.as_posix()}"
            )
        return "\n\n---\n\n".join(blocks)
