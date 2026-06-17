from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_PATH = ROOT_DIR / "data" / "features" / "growth_features.csv"


METRIC_ALIASES: dict[str, list[str]] = {
    "revenue": ["doanh thu", "revenue"],
    "net_profit_parent": [
        "loi nhuan",
        "lợi nhuận",
        "loi nhuan cong ty me",
        "lợi nhuận công ty mẹ",
        "loi nhuan thuoc co dong cong ty me",
        "lợi nhuận thuộc cổ đông công ty mẹ",
        "net profit parent",
        "attributable to parent",
    ],
    "net_profit": ["loi nhuan hop nhat", "lợi nhuận hợp nhất", "net profit", "profit after tax"],
    "gross_profit": ["loi nhuan gop", "lợi nhuận gộp", "gross profit"],
    "operating_profit": ["loi nhuan hoat dong", "lợi nhuận hoạt động", "operating profit"],
    "revenue_g1q": ["tang truong doanh thu quy", "doanh thu g1q", "revenue growth qoq"],
    "revenue_g4q": ["tang truong doanh thu nam", "doanh thu yoy", "revenue growth yoy"],
    "net_profit_g1q": ["tang truong loi nhuan quy", "loi nhuan g1q", "profit growth qoq"],
    "net_profit_g4q": ["tang truong loi nhuan nam", "loi nhuan yoy", "profit growth yoy"],
    "target_profit_growth_1q": [
        "target profit growth 1q",
        "nhan tang truong loi nhuan 1 quy",
        "nhãn tăng trưởng lợi nhuận 1 quý",
    ],
    "target_profit_growth_4q": [
        "target profit growth 4q",
        "nhan tang truong loi nhuan 4 quy",
        "nhãn tăng trưởng lợi nhuận 4 quý",
    ],
    "target_profit_up_4q": ["loi nhuan co tang 4 quy", "profit up 4q"],
    "q_return": ["ty suat sinh loi", "lợi suất", "q return", "return"],
    "price_volatility": ["bien dong gia", "volatility", "biến động"],
    "momentum_3m": ["momentum", "dong luong gia", "động lượng"],
    "roe": ["roe"],
    "roa": ["roa"],
    "pe": ["p/e", "pe"],
    "pb": ["p/b", "pb"],
    "eps": ["eps", "lai co ban", "lãi cơ bản"],
    "debt_to_asset_calc": ["no tren tai san", "nợ trên tài sản", "debt to asset"],
    "net_profit_margin_calc": ["bien loi nhuan rong", "biên lợi nhuận ròng", "net margin"],
    "asset_turnover_calc": ["vong quay tai san", "vòng quay tài sản", "asset turnover"],
    "bank_nim_proxy": ["nim", "bien lai", "biên lãi"],
    "bank_casa_proxy": ["casa"],
    "bank_loan_to_deposit": ["ldr", "loan to deposit", "cho vay tren tien gui"],
    "bank_provision_to_loans": ["du phong tren cho vay", "provision to loans"],
}

DEFAULT_METRICS = [
    "revenue",
    "net_profit_parent",
    "net_profit",
    "revenue_g1q",
    "net_profit_g1q",
    "q_return",
    "roe",
    "pe",
    "pb",
]


@dataclass(frozen=True)
class GraphRecord:
    source_id: str
    symbol: str
    year: int
    quarter: int
    yq_index: int
    metrics: dict[str, float | int | str]
    materialized_at: str | None = None


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "đ": "d",
        "Đ": "D",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    try:
        import unicodedata

        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    except Exception:
        pass
    return re.sub(r"\s+", " ", text.lower()).strip()


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric) >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f} tỷ"
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.2f} triệu"
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    return f"{numeric:.4f}"


def _previous_yq_index(yq_index: int) -> int:
    year = yq_index // 10
    quarter = yq_index % 10
    if quarter <= 1:
        return (year - 1) * 10 + 4
    return year * 10 + (quarter - 1)


class FinancialGraphRetriever:
    """Lightweight graph-style retriever for structured quarterly financial data.

    This applies the FS-Graph RAG idea to the project's existing panel data without
    requiring a graph database. Each symbol-quarter row is treated as a report node,
    and selected numeric columns are metric nodes attached to that report.
    """

    def __init__(self, feature_path: str | Path = DEFAULT_FEATURE_PATH):
        self.feature_path = Path(feature_path)
        self._dataset: pd.DataFrame | None = None
        self._symbols: set[str] = set()

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            if not self.feature_path.exists():
                self._dataset = pd.DataFrame()
                self._symbols = set()
            else:
                df = pd.read_csv(self.feature_path)
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper()
                    self._symbols = set(df["symbol"].dropna().unique().tolist())
                self._dataset = df
        return self._dataset

    def detect_symbols(self, query: str, ticker: str | None = None) -> list[str]:
        if ticker:
            return [ticker.upper()]
        text = query.upper()
        hits = [symbol for symbol in self._symbols if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", text)]
        return sorted(hits)

    def detect_periods(self, query: str) -> set[int]:
        text = _normalize_text(query)
        periods: set[int] = set()
        for year, quarter in re.findall(r"(20\d{2})\s*[-/]?\s*q([1-4])", text):
            periods.add(int(year) * 10 + int(quarter))
        for quarter, year in re.findall(r"q([1-4])\s*[-/]?\s*(20\d{2})", text):
            periods.add(int(year) * 10 + int(quarter))
        for quarter, year in re.findall(r"quy\s*([1-4])\D{0,8}(20\d{2})", text):
            periods.add(int(year) * 10 + int(quarter))
        for yq in re.findall(r"\b(20\d{2}[1-4])\b", text):
            periods.add(int(yq))
        return periods

    def detect_metrics(self, query: str) -> list[str]:
        text = _normalize_text(query)
        metrics = []
        for metric, aliases in METRIC_ALIASES.items():
            normalized_aliases = [_normalize_text(alias) for alias in aliases]
            if any(alias in text for alias in normalized_aliases):
                metrics.append(metric)
        return metrics or DEFAULT_METRICS

    def should_include_previous_period(self, query: str) -> bool:
        text = _normalize_text(query)
        keywords = [
            "quy truoc",
            "quy lien truoc",
            "ky truoc",
            "so voi quy truoc",
            "so voi ky truoc",
            "qoq",
        ]
        return any(keyword in text for keyword in keywords)

    def retrieve(self, query: str, ticker: str | None = None, top_k: int = 4) -> list[GraphRecord]:
        df = self.dataset
        required = {"symbol", "year", "quarter", "yq_index"}
        if df.empty or not required.issubset(df.columns):
            return []

        symbols = self.detect_symbols(query, ticker)
        if not symbols:
            return []

        metrics = [metric for metric in self.detect_metrics(query) if metric in df.columns]
        if not metrics:
            return []

        work = df[df["symbol"].isin(symbols)].copy()
        periods = self.detect_periods(query)
        if periods:
            if self.should_include_previous_period(query):
                periods = set(periods) | {_previous_yq_index(period) for period in periods}
            work = work[work["yq_index"].astype(int).isin(periods)]
        else:
            work = (
                work.sort_values(["symbol", "yq_index"])
                .groupby("symbol", group_keys=False)
                .tail(max(1, top_k))
            )
        if work.empty:
            return []

        records = []
        for i, row in work.sort_values(["symbol", "yq_index"], ascending=[True, False]).head(top_k * max(1, len(symbols))).iterrows():
            metric_values = {
                metric: row[metric]
                for metric in metrics
                if metric in row.index and pd.notna(row[metric])
            }
            if not metric_values:
                continue
            yq_index = int(row["yq_index"])
            records.append(
                GraphRecord(
                    source_id=f"Graph {len(records) + 1}",
                    symbol=str(row["symbol"]).upper(),
                    year=int(row["year"]),
                    quarter=int(row["quarter"]),
                    yq_index=yq_index,
                    metrics=metric_values,
                    materialized_at=str(row["materialized_at"]) if "materialized_at" in row.index and pd.notna(row["materialized_at"]) else None,
                )
            )
        return records

    def format_context(self, records: Iterable[GraphRecord]) -> str:
        blocks = []
        for record in records:
            metric_lines = "\n".join(
                f"- {metric}: {_format_value(value)}"
                for metric, value in record.metrics.items()
            )
            blocks.append(
                f"[{record.source_id}]\n"
                f"Loại nguồn: Structured financial graph\n"
                f"Node: Company({record.symbol}) -> Report({record.year}Q{record.quarter}) -> Metric\n"
                f"Mã: {record.symbol}\n"
                f"Kỳ: {record.year}Q{record.quarter} ({record.yq_index})\n"
                f"Trạng thái dữ liệu: observed/materialized feature data, không phải output dự báo của mô hình\n"
                f"Thời điểm materialize: {record.materialized_at or 'N/A'}\n"
                f"Metrics:\n{metric_lines}\n"
                f"Lưu ý diễn giải: các metric tài chính như revenue, net_profit_parent là số liệu đã có trong feature panel. "
                f"Các cột target_* nếu xuất hiện là nhãn/target huấn luyện được tính từ dữ liệu tương lai đã biết, không phải dự báo mới.\n"
                f"Nguồn dữ liệu: {self.feature_path.as_posix()}"
            )
        return "\n\n---\n\n".join(blocks)
