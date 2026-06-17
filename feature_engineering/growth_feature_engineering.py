"""
Feature engineering for listed-company growth forecasting.

The module builds a quarterly panel from financial statements, ratios,
market prices, optional macro data, and optional news sentiment aggregates.
It is intentionally schema-tolerant because Vietnamese market data vendors
often expose slightly different column names across sectors.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


IDENTIFIER_COLS = {"symbol", "ticker", "year", "quarter", "yq_index", "source", "period"}
TARGET_HORIZONS = (1, 2, 3, 4, 8)
TARGET_COLS = {
    f"target_{kind}_{horizon}q"
    for horizon in TARGET_HORIZONS
    for kind in (
        "revenue_growth",
        "profit_growth",
        "profit_up",
        "strong_profit_up",
        "growth_class",
        "adaptive_strong_profit_up",
    )
}

PROFIT_TARGET_SOURCE_CANDIDATES = ["net_profit_parent", "net_profit"]


def profit_target_source(df: pd.DataFrame) -> str | None:
    """Prefer profit attributable to parent shareholders for model targets."""
    for col in PROFIT_TARGET_SOURCE_CANDIDATES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def clean_colname(name: object) -> str:
    text = strip_accents(str(name)).lower()
    text = text.replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [clean_colname(col) for col in result.columns]
    return result


def coalesce_columns(df: pd.DataFrame, candidates: Iterable[str], new_col: str) -> pd.DataFrame:
    result = df.copy()
    for candidate in candidates:
        col = clean_colname(candidate)
        if col in result.columns:
            result[new_col] = pd.to_numeric(result[col], errors="coerce")
            return result
    if new_col not in result.columns:
        result[new_col] = np.nan
    return result


def normalize_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")


def normalize_quarter(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.upper()
    q = raw.str.extract(r"Q([1-4])", expand=False)
    fallback = raw.str.extract(r"([1-4])", expand=False)
    return pd.to_numeric(q.fillna(fallback), errors="coerce")


def make_yq_index(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return year.astype(int) * 10 + quarter.astype(int)


def parse_news_dates(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str)
    numeric_date = raw.str.extract(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", expand=False)
    parsed = pd.to_datetime(numeric_date.fillna(raw), dayfirst=True, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_convert(None)


def normalize_symbol_value(value: object) -> str:
    if pd.isna(value):
        return ""
    symbol = str(value).strip().upper()
    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
    return "" if symbol in {"", "NAN", "NONE", "NULL", "UNKNOWN"} else symbol


def infer_data_root_from_news_path(path: Path) -> Path | None:
    resolved = path.resolve()
    parts = [part.lower() for part in resolved.parts]
    if len(parts) >= 3 and parts[-3:] == ["raw", "news_external", resolved.name.lower()]:
        return resolved.parents[2]
    if "data" in parts:
        index = len(parts) - 1 - parts[::-1].index("data")
        return Path(*resolved.parts[: index + 1])
    return None


def load_valid_symbols(data_root: Path | None) -> set[str]:
    symbols: set[str] = set()
    if data_root is None:
        return symbols

    reference_path = data_root / "raw" / "reference" / "equity" / "list.csv"
    if reference_path.exists():
        try:
            reference = pd.read_csv(reference_path, usecols=["symbol"])
            symbols.update(normalize_symbol_value(value) for value in reference["symbol"])
        except Exception as exc:
            logger.warning("Cannot load symbol reference %s: %s", reference_path, exc)

    fundamental_root = data_root / "raw" / "fundamental"
    if fundamental_root.exists():
        symbols.update(path.name.upper() for path in fundamental_root.iterdir() if path.is_dir())

    return {symbol for symbol in symbols if symbol}


def map_news_symbols(news: pd.DataFrame, valid_symbols: set[str]) -> pd.Series:
    result = pd.Series("", index=news.index, dtype="object")

    # The crawl keyword is usually the intended ticker. Prefer it over ticker
    # extracted from article text, which can be noisy when articles mention peers.
    for col in ["keyword", "symbol", "ticker"]:
        if col not in news.columns:
            continue
        candidate = news[col].map(normalize_symbol_value)
        if valid_symbols:
            candidate = candidate.where(candidate.isin(valid_symbols), "")
        result = result.where(result.ne(""), candidate)

    if valid_symbols:
        text = (
            news.get("title", pd.Series("", index=news.index)).fillna("").astype(str)
            + " "
            + news.get("content", pd.Series("", index=news.index)).fillna("").astype(str)
        ).str.upper()
        unresolved = result.eq("")
        if unresolved.any():
            # Fall back to ticker tokens in text for rows with no usable keyword.
            symbols_by_length = sorted(valid_symbols, key=len, reverse=True)
            for symbol in symbols_by_length:
                pattern = rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])"
                hit = unresolved & text.str.contains(pattern, regex=True, na=False)
                if hit.any():
                    result.loc[hit] = symbol
                    unresolved = result.eq("")
                if not unresolved.any():
                    break

    return result.replace("", np.nan)


class GrowthFeatureEngineer:
    def __init__(self, winsorize_quantiles: tuple[float, float] = (0.01, 0.99)):
        self.winsorize_quantiles = winsorize_quantiles

    def discover_tickers(self, raw_root: str | Path, tickers: Optional[List[str]] = None) -> List[str]:
        root = Path(raw_root)
        if tickers:
            return [ticker.upper() for ticker in tickers if (root / ticker.upper()).is_dir()]
        return sorted(path.name.upper() for path in root.iterdir() if path.is_dir())

    def discover_tickers_ml_tree(self, data_root: str | Path, tickers: Optional[List[str]] = None) -> List[str]:
        root = Path(data_root) / "raw" / "fundamental"
        if tickers:
            return [ticker.upper() for ticker in tickers if (root / ticker.upper()).is_dir()]
        if not root.exists():
            return []
        return sorted(path.name.upper() for path in root.iterdir() if path.is_dir())

    def load_stock_data(self, stock_dir: str | Path) -> Dict[str, pd.DataFrame]:
        stock_dir = Path(stock_dir)
        files = {
            "income": "income.csv",
            "balance": "balance.csv",
            "cashflow": "cashflow.csv",
            "ratios": "ratios.csv",
            "price": "price.csv",
        }
        data: Dict[str, pd.DataFrame] = {}
        for key, filename in files.items():
            path = stock_dir / filename
            if not path.exists():
                data[key] = pd.DataFrame()
                continue
            if key == "price":
                data[key] = clean_columns(pd.read_csv(path, parse_dates=["date"]))
            else:
                data[key] = clean_columns(pd.read_csv(path))
        return data

    def load_stock_data_ml_tree(self, data_root: str | Path, symbol: str) -> Dict[str, pd.DataFrame]:
        root = Path(data_root)
        symbol = symbol.upper()
        files = {
            "income": root / "raw" / "fundamental" / symbol / "income_statement.csv",
            "balance": root / "raw" / "fundamental" / symbol / "balance_sheet.csv",
            "cashflow": root / "raw" / "fundamental" / symbol / "cash_flow.csv",
            "ratios": root / "raw" / "fundamental" / symbol / "ratio.csv",
            "price": root / "raw" / "market" / "equity" / symbol / "ohlcv.csv",
        }
        data: Dict[str, pd.DataFrame] = {}
        for key, path in files.items():
            if not path.exists():
                data[key] = pd.DataFrame()
                continue
            if key == "price":
                price = clean_columns(pd.read_csv(path))
                if "date" not in price.columns:
                    for candidate in ["time", "trading_date"]:
                        if candidate in price.columns:
                            price = price.rename(columns={candidate: "date"})
                            break
                data[key] = price
            else:
                data[key] = clean_columns(pd.read_csv(path))
        return data

    def load_macro_data(self, macro_path: str | Path | None) -> pd.DataFrame:
        if not macro_path:
            return pd.DataFrame()
        path = Path(macro_path)
        if not path.exists():
            return pd.DataFrame()
        macro = clean_columns(pd.read_csv(path))
        if "year" not in macro.columns:
            return pd.DataFrame()
        macro["year"] = normalize_year(macro["year"])
        macro = macro.dropna(subset=["year"]).copy()
        macro["year"] = macro["year"].astype(int)
        numeric_cols = [c for c in macro.select_dtypes(include=[np.number]).columns if c != "year"]
        return macro[["year", *numeric_cols]].drop_duplicates("year")

    def load_news_sentiment(self, news_path: str | Path | None) -> pd.DataFrame:
        if not news_path:
            return pd.DataFrame()
        path = Path(news_path)
        if not path.exists():
            return pd.DataFrame()

        news = clean_columns(pd.read_csv(path))
        valid_symbols = load_valid_symbols(infer_data_root_from_news_path(path))
        news["symbol"] = map_news_symbols(news, valid_symbols)
        if "date" not in news.columns or "symbol" not in news.columns:
            return pd.DataFrame()
        if "is_relevant_news" in news.columns:
            relevant = news["is_relevant_news"].astype(str).str.lower().isin({"true", "1", "yes"})
            news = news[relevant].copy()
            if news.empty:
                return pd.DataFrame()

        news["date"] = parse_news_dates(news["date"])
        news = news.dropna(subset=["date", "symbol"]).copy()
        news["symbol"] = news["symbol"].astype(str).str.upper()
        news["year"] = news["date"].dt.year
        news["quarter"] = news["date"].dt.quarter
        news["yq_index"] = make_yq_index(news["year"], news["quarter"])

        if "sentiment" in news.columns:
            sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            news["sentiment_score"] = news["sentiment"].astype(str).str.lower().map(sentiment_map)
        elif "sentiment_score" not in news.columns:
            news["sentiment_score"] = np.nan

        label = news.get("sentiment_label", pd.Series("", index=news.index)).astype(str).str.lower()
        news["positive_news"] = label.eq("positive").astype(float)
        news["negative_news"] = label.eq("negative").astype(float)
        news["neutral_news"] = label.eq("neutral").astype(float)
        text = (
            news.get("title", pd.Series("", index=news.index)).fillna("").astype(str)
            + " "
            + news.get("content", pd.Series("", index=news.index)).fillna("").astype(str)
        ).str.lower()
        news["news_text_length"] = text.str.len()
        topic_keywords = {
            "risk_news": r"rủi ro|rui ro|áp lực|ap luc|khó khăn|kho khan|suy giảm|suy giam|giảm|giam|lỗ|lo|nợ xấu|no xau|bán tháo|ban thao|lao dốc|lao doc",
            "growth_news": r"tăng trưởng|tang truong|tăng|tang|lãi|lai|lợi nhuận|loi nhuan|doanh thu|kỷ lục|ky luc|mở rộng|mo rong",
            "earnings_news": r"lợi nhuận|loi nhuan|doanh thu|kết quả kinh doanh|ket qua kinh doanh|báo lãi|bao lai|báo lỗ|bao lo",
            "policy_news": r"lãi suất|lai suat|tỷ giá|ty gia|chính sách|chinh sach|ngân hàng nhà nước|ngan hang nha nuoc|thuế|thue",
            "market_news": r"cổ phiếu|co phieu|vn-index|thanh khoản|thanh khoan|khối ngoại|khoi ngoai|tự doanh|tu doanh",
        }
        for col, pattern in topic_keywords.items():
            news[col] = text.str.contains(pattern, regex=True, na=False).astype(float)

        event_keywords = {
            "event_capital_increase": r"tăng vốn|tang von|phát hành|phat hanh|chào bán|chao ban|cổ tức cổ phiếu|co tuc co phieu|quyền mua|quyen mua",
            "event_mna": r"\bm&a\b|sáp nhập|sap nhap|mua lại|mua lai|thâu tóm|thau tom|chuyển nhượng|chuyen nhuong|thoái vốn|thoai von|đầu tư chiến lược|dau tu chien luoc",
            "event_profit_growth": r"lợi nhuận.{0,40}tăng|loi nhuan.{0,40}tang|lãi.{0,40}tăng|lai.{0,40}tang|doanh thu.{0,40}tăng|doanh thu.{0,40}tang|báo lãi|bao lai|vượt kế hoạch|vuot ke hoach",
            "event_profit_decline": r"lợi nhuận.{0,40}giảm|loi nhuan.{0,40}giam|lãi.{0,40}giảm|lai.{0,40}giam|doanh thu.{0,40}giảm|doanh thu.{0,40}giam|báo lỗ|bao lo|thua lỗ|thua lo|lỗ sau thuế|lo sau thue",
            "event_leadership_change": r"bổ nhiệm|bo nhiem|miễn nhiệm|mien nhiem|từ nhiệm|tu nhiem|thay đổi lãnh đạo|thay doi lanh dao|chủ tịch|chu tich|tổng giám đốc|tong giam doc|hội đồng quản trị|hoi dong quan tri",
            "event_investigation_penalty": r"điều tra|dieu tra|xử phạt|xu phat|phạt|phat|vi phạm|vi pham|kiểm toán ngoại trừ|kiem toan ngoai tru|thanh tra|khởi tố|khoi to",
            "event_factory_expansion": r"mở rộng nhà máy|mo rong nha may|nhà máy mới|nha may moi|dự án mới|du an moi|tăng công suất|tang cong suat|khởi công|khoi cong|vận hành|van hanh",
            "event_debt_risk": r"trái phiếu|trai phieu|nợ vay|no vay|đáo hạn|dao han|thanh khoản|thanh khoan|áp lực trả nợ|ap luc tra no|chậm trả|cham tra",
            "event_foreign_investor_activity": r"khối ngoại|khoi ngoai|nhà đầu tư nước ngoài|nha dau tu nuoc ngoai|mua ròng|mua rong|bán ròng|ban rong",
            "event_bank_credit_quality": r"nợ xấu|no xau|dự phòng|du phong|tín dụng|tin dung|casa|nim|biên lãi|bien lai",
        }
        for col, pattern in event_keywords.items():
            news[col] = text.str.contains(pattern, regex=True, na=False).astype(float)

        sector_keywords = {
            "event_bank_sector": r"ngân hàng|ngan hang|tín dụng|tin dung|nợ xấu|no xau|casa|nim|dự phòng|du phong|cho vay|tiền gửi|tien gui",
            "event_real_estate_sector": r"bất động sản|bat dong san|dự án|du an|quỹ đất|quy dat|pháp lý|phap ly|trái phiếu|trai phieu|bàn giao|ban giao",
            "event_steel_sector": r"thép|thep|hòa phát|hoa phat|quặng sắt|quang sat|hpg|sản lượng thép|san luong thep|giá thép|gia thep",
            "event_retail_sector": r"bán lẻ|ban le|chuỗi cửa hàng|chuoi cua hang|bách hóa|bach hoa|điện máy|dien may|tiêu dùng|tieu dung",
            "event_technology_sector": r"công nghệ|cong nghe|chuyển đổi số|chuyen doi so|phần mềm|phan mem|ai|bán dẫn|ban dan|viễn thông|vien thong",
        }
        for col, pattern in sector_keywords.items():
            news[col] = text.str.contains(pattern, regex=True, na=False).astype(float)

        news["event_total_count"] = news[list(event_keywords)].sum(axis=1).clip(upper=1)
        news["event_positive_count"] = news[["event_profit_growth", "event_factory_expansion", "event_capital_increase"]].sum(axis=1).clip(upper=1)
        news["event_negative_count"] = news[["event_profit_decline", "event_investigation_penalty", "event_debt_risk", "event_bank_credit_quality"]].sum(axis=1).clip(upper=1)
        news["event_impact_score"] = news["event_positive_count"] - news["event_negative_count"]
        source_text = (
            news.get("source", pd.Series("", index=news.index)).fillna("").astype(str)
            + " "
            + news.get("source_system", pd.Series("", index=news.index)).fillna("").astype(str)
        ).str.lower()
        news["news_source_weight"] = np.select(
            [
                source_text.str.contains(r"hose|hnx|upcom|ssc|doanh nghiệp|cong bo", regex=True, na=False),
                source_text.str.contains(r"vietstock|cafef|tinnhanhchungkhoan|ndh|vneconomy|vietnamfinance|vietnambiz", regex=True, na=False),
                source_text.str.contains(r"google_news_rss|vnexpress", regex=True, na=False),
            ],
            [1.0, 0.8, 0.6],
            default=0.5,
        )
        news["event_weighted_impact_score"] = news["news_source_weight"] * (
            3.0 * news["event_profit_growth"]
            + 2.0 * news["event_factory_expansion"]
            + 2.0 * news["event_mna"]
            + 1.0 * news["event_capital_increase"]
            + 0.5 * news["event_foreign_investor_activity"]
            - 3.0 * news["event_profit_decline"]
            - 3.0 * news["event_investigation_penalty"]
            - 2.0 * news["event_debt_risk"]
            - 2.0 * news["event_bank_credit_quality"]
        )

        grouped = news.groupby(["symbol", "year", "quarter", "yq_index"]).agg(
            news_count=("symbol", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            min_sentiment=("sentiment_score", "min"),
            max_sentiment=("sentiment_score", "max"),
            sentiment_std=("sentiment_score", "std"),
            positive_news_count=("positive_news", "sum"),
            negative_news_count=("negative_news", "sum"),
            neutral_news_count=("neutral_news", "sum"),
            risk_news_count=("risk_news", "sum"),
            growth_news_count=("growth_news", "sum"),
            earnings_news_count=("earnings_news", "sum"),
            policy_news_count=("policy_news", "sum"),
            market_news_count=("market_news", "sum"),
            event_capital_increase_count=("event_capital_increase", "sum"),
            event_mna_count=("event_mna", "sum"),
            event_profit_growth_count=("event_profit_growth", "sum"),
            event_profit_decline_count=("event_profit_decline", "sum"),
            event_leadership_change_count=("event_leadership_change", "sum"),
            event_investigation_penalty_count=("event_investigation_penalty", "sum"),
            event_factory_expansion_count=("event_factory_expansion", "sum"),
            event_debt_risk_count=("event_debt_risk", "sum"),
            event_foreign_investor_activity_count=("event_foreign_investor_activity", "sum"),
            event_bank_credit_quality_count=("event_bank_credit_quality", "sum"),
            event_bank_sector_count=("event_bank_sector", "sum"),
            event_real_estate_sector_count=("event_real_estate_sector", "sum"),
            event_steel_sector_count=("event_steel_sector", "sum"),
            event_retail_sector_count=("event_retail_sector", "sum"),
            event_technology_sector_count=("event_technology_sector", "sum"),
            event_total_count=("event_total_count", "sum"),
            event_positive_count=("event_positive_count", "sum"),
            event_negative_count=("event_negative_count", "sum"),
            event_impact_score=("event_impact_score", "sum"),
            event_weighted_impact_score=("event_weighted_impact_score", "sum"),
            avg_news_source_weight=("news_source_weight", "mean"),
            avg_news_text_length=("news_text_length", "mean"),
        ).reset_index()
        grouped["positive_news_ratio"] = grouped["positive_news_count"] / grouped["news_count"].clip(lower=1)
        grouped["negative_news_ratio"] = grouped["negative_news_count"] / grouped["news_count"].clip(lower=1)
        grouped["risk_news_ratio"] = grouped["risk_news_count"] / grouped["news_count"].clip(lower=1)
        grouped["growth_news_ratio"] = grouped["growth_news_count"] / grouped["news_count"].clip(lower=1)
        grouped["earnings_news_ratio"] = grouped["earnings_news_count"] / grouped["news_count"].clip(lower=1)
        grouped["event_total_ratio"] = grouped["event_total_count"] / grouped["news_count"].clip(lower=1)
        grouped["event_positive_ratio"] = grouped["event_positive_count"] / grouped["news_count"].clip(lower=1)
        grouped["event_negative_ratio"] = grouped["event_negative_count"] / grouped["news_count"].clip(lower=1)
        grouped["event_net_positive_ratio"] = grouped["event_positive_ratio"] - grouped["event_negative_ratio"]
        grouped["event_impact_score_avg"] = grouped["event_impact_score"] / grouped["news_count"].clip(lower=1)
        grouped["event_weighted_impact_score_avg"] = grouped["event_weighted_impact_score"] / grouped["news_count"].clip(lower=1)
        return grouped

    def load_transformer_news_sentiment(self, news_path: str | Path | None) -> pd.DataFrame:
        if not news_path:
            return pd.DataFrame()
        path = Path(news_path)
        data_root = infer_data_root_from_news_path(path)
        if data_root is None:
            return pd.DataFrame()
        transformer_path = data_root / "raw" / "news_external" / "news_transformer_sentiment_quarterly.csv"
        if not transformer_path.exists():
            return pd.DataFrame()
        df = clean_columns(pd.read_csv(transformer_path))
        required = {"symbol", "year", "quarter", "yq_index"}
        if not required.issubset(df.columns):
            return pd.DataFrame()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        for col in ["year", "quarter", "yq_index"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["symbol", "year", "quarter", "yq_index"]).copy()
        df[["year", "quarter", "yq_index"]] = df[["year", "quarter", "yq_index"]].astype(int)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in {"year", "quarter", "yq_index"}]
        return df[["symbol", "year", "quarter", "yq_index", *numeric_cols]]

    def _standardize_time_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "symbol" not in result.columns and "ticker" in result.columns:
            result["symbol"] = result["ticker"]
        if "symbol" not in result.columns and "meta_ticker" in result.columns:
            result["symbol"] = result["meta_ticker"]
        if "year" not in result.columns and "yearreport" in result.columns:
            result["year"] = result["yearreport"]
        if "year" not in result.columns and "meta_yearreport" in result.columns:
            result["year"] = result["meta_yearreport"]
        if "year" not in result.columns and "period" in result.columns:
            result["year"] = result["period"]
        if "quarter" not in result.columns:
            if "quarterreport" in result.columns:
                result["quarter"] = result["quarterreport"]
            elif "lengthreport" in result.columns:
                result["quarter"] = result["lengthreport"]
            elif "meta_lengthreport" in result.columns:
                result["quarter"] = result["meta_lengthreport"]
            elif "period" in result.columns:
                result["quarter"] = result["period"]

        required = {"symbol", "year", "quarter"}
        if not required.issubset(result.columns):
            return pd.DataFrame()

        result["symbol"] = result["symbol"].astype(str).str.upper()
        result["year"] = normalize_year(result["year"])
        result["quarter"] = normalize_quarter(result["quarter"])
        result = result.dropna(subset=["symbol", "year", "quarter"]).copy()
        result["year"] = result["year"].astype(int)
        result["quarter"] = result["quarter"].astype(int)
        result = result[result["quarter"].between(1, 4)]
        result["yq_index"] = make_yq_index(result["year"], result["quarter"])
        return result

    def standardize_income(self, income: pd.DataFrame) -> pd.DataFrame:
        if income.empty:
            return pd.DataFrame()
        df = self._standardize_time_keys(income)
        if df.empty:
            return df

        mapping = {
            "revenue": ["revenue_bn_vnd", "revenue", "net_sales", "sales", "interest_and_similar_income", "total_operating_income", "net_revenue_of_insurance_premium"],
            "revenue_yoy": ["revenue_yoy_pct", "revenue_yoy"],
            "gross_profit": ["gross_profit"],
            "operating_profit": ["operating_profit_loss", "operating_profit"],
            "profit_before_tax": ["profit_before_tax", "net_accounting_profit_loss_before_tax"],
            "net_profit": ["net_profit_for_the_year", "net_profit_loss_after_tax", "profit_after_tax", "net_profit", "attributable_to_parent_company", "net_profit_attributable_to_shareholders_of_the_group"],
            "net_profit_parent": ["attributable_to_parent_company", "net_profit_attributable_to_shareholders_of_the_group", "attribute_to_parent_company_bn_vnd", "attribute_to_parent_company"],
            "eps": ["eps_basis", "eps_basic_vnd", "eps", "eps_vnd"],
            "bank_interest_income": ["interest_and_similar_income"],
            "bank_interest_expense": ["interest_and_similar_expenses"],
            "bank_net_interest_income": ["net_interest_income"],
            "bank_pre_provision_profit": ["net_operating_profit_before_allowance_for_credit_loss"],
            "bank_credit_loss_provision": ["provision_for_credit_losses"],
        }
        for new_col, candidates in mapping.items():
            df = coalesce_columns(df, candidates, new_col)

        keep = ["symbol", "year", "quarter", "yq_index", *mapping.keys()]
        return df[keep].drop_duplicates(["symbol", "year", "quarter"])

    def standardize_balance(self, balance: pd.DataFrame) -> pd.DataFrame:
        if balance.empty:
            return pd.DataFrame()
        df = self._standardize_time_keys(balance)
        if df.empty:
            return df

        mapping = {
            "current_assets": ["current_assets_bn_vnd", "current_assets"],
            "cash": ["cash_and_cash_equivalents_bn_vnd", "cash_and_cash_equivalents"],
            "inventory": ["inventories_net_bn_vnd", "net_inventories", "inventory"],
            "fixed_assets": ["fixed_assets_bn_vnd", "fixed_assets"],
            "total_assets": ["total_assets_bn_vnd", "total_assets"],
            "liabilities": ["liabilities_bn_vnd", "liabilities", "total_liabilities", "other_liabilities"],
            "current_liabilities": ["current_liabilities_bn_vnd", "current_liabilities"],
            "equity": [
                "owner_s_equity_bn_vnd",
                "owner_s_equity",
                "owners_equity_bnvnd",
                "owners_equity_bn_vnd",
                "owners_equity",
                "shareholders_equity",
                "shareholder_s_equity",
                "capital_and_reserves",
                "equity",
            ],
            "short_term_borrowings": ["short_term_borrowings_bn_vnd", "short_term_borrowings"],
            "long_term_borrowings": ["long_term_borrowings_bn_vnd", "long_term_borrowings"],
            "paid_in_capital": ["paid_in_capital_bn_vnd", "paid_in_capital"],
            "bank_customer_loans_gross": ["loans_and_advances_to_customers"],
            "bank_customer_loans_net": ["loans_and_advances_to_customers_net"],
            "bank_loan_loss_allowance": ["less_provision_for_losses_on_loans_and_advances_to_customers"],
            "bank_customer_deposits": ["deposits_from_customers"],
            "bank_interbank_assets": ["placements_with_and_loans_to_other_credit_institutions", "balances_with_other_credit_institutions", "loans_to_other_credit_institutions"],
            "bank_interbank_liabilities": ["deposits_and_loans_from_other_credit_institutions", "loans_from_other_credit_institutions"],
        }
        for new_col, candidates in mapping.items():
            df = coalesce_columns(df, candidates, new_col)

        keep = ["symbol", "year", "quarter", "yq_index", *mapping.keys()]
        return df[keep].drop_duplicates(["symbol", "year", "quarter"])

    def standardize_cashflow(self, cashflow: pd.DataFrame) -> pd.DataFrame:
        if cashflow.empty:
            return pd.DataFrame()
        df = self._standardize_time_keys(cashflow)
        if df.empty:
            return df

        mapping = {
            "cfo": ["net_cash_flows_from_operating_activities", "net_cash_inflows_outflows_from_operating_activities"],
            "cfi": ["net_cash_flows_from_investing_activities"],
            "cff": ["cash_flows_from_financial_activities", "net_cash_flows_from_financing_activities"],
            "capex": ["purchase_of_fixed_assets", "purchase_of_fixed_assets_and_other_long_term_assets", "capital_expenditure"],
            "cash_end": ["cash_and_cash_equivalents_at_the_end_of_period", "cash_and_cash_equivalents_at_end_of_period"],
        }
        for new_col, candidates in mapping.items():
            df = coalesce_columns(df, candidates, new_col)

        keep = ["symbol", "year", "quarter", "yq_index", *mapping.keys()]
        return df[keep].drop_duplicates(["symbol", "year", "quarter"])

    def standardize_ratios(self, ratios: pd.DataFrame) -> pd.DataFrame:
        if ratios.empty:
            return pd.DataFrame()
        df = self._standardize_time_keys(ratios)
        if df.empty:
            return df

        mapping = {
            "roe": ["chi_tieu_kha_nang_sinh_loi_roe_pct", "roe_pct", "roe"],
            "roa": ["chi_tieu_kha_nang_sinh_loi_roa_pct", "roa_pct", "roa"],
            "roic": ["chi_tieu_kha_nang_sinh_loi_roic_pct", "roic_pct", "roic"],
            "gross_margin": ["chi_tieu_kha_nang_sinh_loi_gross_profit_margin_pct", "gross_profit_margin_pct"],
            "net_margin": ["chi_tieu_kha_nang_sinh_loi_net_profit_margin_pct", "net_profit_margin_pct"],
            "current_ratio": ["chi_tieu_thanh_khoan_current_ratio", "current_ratio"],
            "quick_ratio": ["chi_tieu_thanh_khoan_quick_ratio", "quick_ratio"],
            "financial_leverage": ["chi_tieu_thanh_khoan_financial_leverage", "financial_leverage"],
            "debt_to_equity": ["chi_tieu_co_cau_nguon_von_debt_equity", "debt_equity"],
            "market_cap": ["chi_tieu_dinh_gia_market_capital_bn_vnd", "market_capital_bn_vnd"],
            "pe": ["chi_tieu_dinh_gia_p_e", "p_e", "pe"],
            "pb": ["chi_tieu_dinh_gia_p_b", "p_b", "pb"],
            "ps": ["chi_tieu_dinh_gia_p_s", "p_s", "ps"],
            "eps_ratio": ["chi_tieu_dinh_gia_eps_vnd", "eps_vnd"],
            "bvps": ["chi_tieu_dinh_gia_bvps_vnd", "bvps_vnd"],
            "bank_current_deposits": ["current_deposits"],
            "bank_margin_deposits": ["margin_deposits"],
            "bank_special_deposits": ["deposits_for_special_purposes"],
            "bank_ratio_customer_deposits": ["deposits_from_customers"],
        }
        for new_col, candidates in mapping.items():
            df = coalesce_columns(df, candidates, new_col)

        keep = ["symbol", "year", "quarter", "yq_index", *mapping.keys()]
        return df[keep].drop_duplicates(["symbol", "year", "quarter"])

    def compute_market_features(self, price: pd.DataFrame) -> pd.DataFrame:
        if price.empty or not {"date", "close", "volume"}.issubset(price.columns):
            return pd.DataFrame()

        df = price.copy()
        if "symbol" not in df.columns and "ticker" in df.columns:
            df["symbol"] = df["ticker"]
        if "symbol" not in df.columns:
            return pd.DataFrame()

        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["symbol", "date", "close"]).sort_values(["symbol", "date"])

        grouped = df.groupby("symbol", group_keys=False)
        df["return_daily"] = grouped["close"].pct_change()
        df["volatility_30d"] = grouped["return_daily"].transform(lambda x: x.rolling(30).std() * np.sqrt(252) * 100)
        df["momentum_3m"] = grouped["close"].transform(lambda x: x.pct_change(63) * 100)
        df["avg_volume_20d"] = grouped["volume"].transform(lambda x: x.rolling(20).mean())
        df["rel_volume"] = df["volume"] / df["avg_volume_20d"]
        df["sma_20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())
        df["sma_60"] = grouped["close"].transform(lambda x: x.rolling(60).mean())
        df["ema_12"] = grouped["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df["ema_26"] = grouped["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = grouped["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df["sma_20_gap"] = (df["close"] - df["sma_20"]) / (np.abs(df["sma_20"]) + 1e-9) * 100
        df["sma_60_gap"] = (df["close"] - df["sma_60"]) / (np.abs(df["sma_60"]) + 1e-9) * 100

        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())
        avg_loss = loss.groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())
        rs = avg_gain / (avg_loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        if {"high", "low"}.issubset(df.columns):
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
        else:
            df["high"] = df["close"]
            df["low"] = df["close"]
        low_14 = grouped["low"].transform(lambda x: x.rolling(14).min())
        high_14 = grouped["high"].transform(lambda x: x.rolling(14).max())
        df["stochastic_k"] = (df["close"] - low_14) / (high_14 - low_14 + 1e-9) * 100
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14 + 1e-9) * -100
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_sma = typical_price.groupby(df["symbol"]).transform(lambda x: x.rolling(20).mean())
        tp_mad = typical_price.groupby(df["symbol"]).transform(lambda x: x.rolling(20).apply(lambda y: np.mean(np.abs(y - y.mean())), raw=False))
        df["cci_20"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-9)
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["yq_index"] = make_yq_index(df["year"], df["quarter"])

        return df.groupby(["symbol", "year", "quarter", "yq_index"]).agg(
            close_eoq=("close", "last"),
            avg_volume=("volume", "mean"),
            q_return=("return_daily", lambda x: ((1 + x.fillna(0)).prod() - 1) * 100),
            price_volatility=("volatility_30d", "mean"),
            momentum_3m=("momentum_3m", "mean"),
            avg_rel_volume=("rel_volume", "mean"),
            technical_sma_20_gap=("sma_20_gap", "last"),
            technical_sma_60_gap=("sma_60_gap", "last"),
            technical_rsi_14=("rsi_14", "last"),
            technical_macd=("macd", "last"),
            technical_macd_signal=("macd_signal", "last"),
            technical_stochastic_k=("stochastic_k", "last"),
            technical_williams_r=("williams_r", "last"),
            technical_cci_20=("cci_20", "mean"),
        ).reset_index()

    def add_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sort_values(["symbol", "yq_index"]).copy()
        eps = 1e-9

        base_cols = [
            "revenue",
            "gross_profit",
            "operating_profit",
            "profit_before_tax",
            "net_profit",
            "net_profit_parent",
            "bank_interest_income",
            "bank_interest_expense",
            "bank_net_interest_income",
            "bank_pre_provision_profit",
            "bank_credit_loss_provision",
            "bank_customer_loans_gross",
            "bank_customer_loans_net",
            "bank_loan_loss_allowance",
            "bank_customer_deposits",
            "bank_current_deposits",
            "total_assets",
            "equity",
            "liabilities",
            "cfo",
            "market_cap",
        ]
        for col in base_cols:
            if col not in result.columns:
                continue
            grouped = result.groupby("symbol")[col]
            result[f"{col}_g1q"] = grouped.pct_change(1) * 100
            result[f"{col}_g4q"] = grouped.pct_change(4) * 100
            result[f"{col}_lag1q"] = grouped.shift(1)
            result[f"{col}_lag4q"] = grouped.shift(4)

        ratio_specs = {
            "gross_profit_margin_calc": ("gross_profit", "revenue"),
            "net_profit_margin_calc": ("net_profit", "revenue"),
            "roa_calc": ("net_profit", "total_assets"),
            "roe_calc": ("net_profit", "equity"),
            "asset_turnover_calc": ("revenue", "total_assets"),
            "debt_to_asset_calc": ("liabilities", "total_assets"),
            "debt_to_equity_calc": ("liabilities", "equity"),
            "equity_to_asset_calc": ("equity", "total_assets"),
            "cfo_to_asset_calc": ("cfo", "total_assets"),
            "earnings_quality_calc": ("cfo", "net_profit"),
            "capex_to_asset_calc": ("capex", "total_assets"),
            "current_ratio_calc": ("current_assets", "current_liabilities"),
            "borrowings_to_equity_calc": ("short_term_borrowings", "equity"),
        }
        for new_col, (num, den) in ratio_specs.items():
            if {num, den}.issubset(result.columns):
                result[new_col] = result[num] / (np.abs(result[den]) + eps)

        if {"net_profit_parent", "revenue"}.issubset(result.columns):
            result["net_profit_parent_margin_calc"] = result["net_profit_parent"] / (np.abs(result["revenue"]) + eps)
        if {"cfo", "net_profit_parent"}.issubset(result.columns):
            result["earnings_quality_parent_calc"] = result["cfo"] / (np.abs(result["net_profit_parent"]) + eps)

        for col in ("roa_calc", "roe_calc"):
            if col in result.columns:
                result[col] = result[col] * 100

        target_profit_col = profit_target_source(result)
        if target_profit_col is not None and {"total_assets", "equity"}.issubset(result.columns):
            target_profit_ttm = result.groupby("symbol")[target_profit_col].transform(
                lambda values: values.rolling(4, min_periods=1).sum()
            )
            result["target_profit_ttm_calc"] = target_profit_ttm

        if {"net_profit", "total_assets"}.issubset(result.columns):
            net_profit_ttm = result.groupby("symbol")["net_profit"].transform(
                lambda values: values.rolling(4, min_periods=1).sum()
            )
            assets_lag4q = result.groupby("symbol")["total_assets"].shift(4)
            avg_assets = (result["total_assets"] + assets_lag4q) / 2
            avg_assets = avg_assets.fillna(
                result.groupby("symbol")["total_assets"].transform(lambda values: values.rolling(4, min_periods=1).mean())
            )
            result["roa_ttm_calc"] = net_profit_ttm / (np.abs(avg_assets) + eps) * 100

        if {"net_profit", "equity"}.issubset(result.columns):
            net_profit_ttm = result.groupby("symbol")["net_profit"].transform(
                lambda values: values.rolling(4, min_periods=1).sum()
            )
            equity_lag4q = result.groupby("symbol")["equity"].shift(4)
            avg_equity = (result["equity"] + equity_lag4q) / 2
            avg_equity = avg_equity.fillna(
                result.groupby("symbol")["equity"].transform(lambda values: values.rolling(4, min_periods=1).mean())
            )
            result["roe_ttm_calc"] = net_profit_ttm / (np.abs(avg_equity) + eps) * 100

        if "roa" in result.columns:
            for fallback_col in ("roa_ttm_calc", "roa_calc"):
                if fallback_col in result.columns:
                    result["roa"] = result["roa"].fillna(result[fallback_col])
        elif "roa_ttm_calc" in result.columns:
            result["roa"] = result["roa_ttm_calc"]
        elif "roa_calc" in result.columns:
            result["roa"] = result["roa_calc"]

        if "roe" in result.columns:
            for fallback_col in ("roe_ttm_calc", "roe_calc"):
                if fallback_col in result.columns:
                    result["roe"] = result["roe"].fillna(result[fallback_col])
        elif "roe_ttm_calc" in result.columns:
            result["roe"] = result["roe_ttm_calc"]
        elif "roe_calc" in result.columns:
            result["roe"] = result["roe_calc"]

        margin_candidates = [
            col
            for col in ("net_margin", "net_profit_margin_calc", "net_profit_parent_margin_calc")
            if col in result.columns
        ]
        if margin_candidates:
            result["net_profit_margin"] = result[margin_candidates].bfill(axis=1).iloc[:, 0]
            if "net_margin" in margin_candidates:
                # Vendor ratios are often expressed in percent, while calculated margins are ratios.
                large_margin = result["net_profit_margin"].abs() > 1.5
                result.loc[large_margin, "net_profit_margin"] = result.loc[large_margin, "net_profit_margin"] / 100

        leverage_candidates = [
            col
            for col in ("debt_to_equity", "debt_to_equity_calc", "total_borrowings_to_equity_calc")
            if col in result.columns
        ]
        if leverage_candidates:
            result["debt_to_equity_quality"] = result[leverage_candidates].bfill(axis=1).iloc[:, 0]

        liquidity_candidates = [
            col
            for col in ("current_ratio", "current_ratio_calc", "quick_ratio")
            if col in result.columns
        ]
        if liquidity_candidates:
            result["liquidity_ratio_quality"] = result[liquidity_candidates].bfill(axis=1).iloc[:, 0]

        target_profit_for_quality = profit_target_source(result)
        if target_profit_for_quality is not None and "cfo" in result.columns:
            result["operating_cashflow_to_profit"] = result["cfo"] / (
                np.abs(result[target_profit_for_quality]) + eps
            )

        if "revenue_g1q" in result.columns:
            result["revenue_growth_stability"] = 1 / (
                1
                + result.groupby("symbol")["revenue_g1q"].transform(
                    lambda values: values.rolling(4, min_periods=2).std()
                ).abs()
            )

        if target_profit_for_quality is not None and f"{target_profit_for_quality}_g1q" in result.columns:
            profit_growth_col = f"{target_profit_for_quality}_g1q"
            result["profit_growth_volatility"] = result.groupby("symbol")[profit_growth_col].transform(
                lambda values: values.rolling(4, min_periods=2).std()
            )

        score_parts = []
        if target_profit_for_quality is not None and f"{target_profit_for_quality}_g4q" in result.columns:
            score_parts.append((result[f"{target_profit_for_quality}_g4q"] > 0).astype(float))
        if "revenue_g4q" in result.columns:
            score_parts.append((result["revenue_g4q"] > 0).astype(float))
        if "net_profit_margin" in result.columns:
            margin_delta_4q = result["net_profit_margin"] - result.groupby("symbol")["net_profit_margin"].shift(4)
            score_parts.append((margin_delta_4q >= 0).astype(float).where(margin_delta_4q.notna()))
        if "operating_cashflow_to_profit" in result.columns:
            score_parts.append((result["operating_cashflow_to_profit"] >= 0.8).astype(float))
        if score_parts:
            result["profit_growth_quality_score"] = pd.concat(score_parts, axis=1).mean(axis=1)

        if {"short_term_borrowings", "long_term_borrowings", "equity"}.issubset(result.columns):
            result["total_borrowings_to_equity_calc"] = (
                result["short_term_borrowings"].fillna(0) + result["long_term_borrowings"].fillna(0)
            ) / (np.abs(result["equity"]) + eps)

        bank_ratio_specs = {
            "bank_nim_proxy": ("bank_net_interest_income", "bank_customer_loans_gross"),
            "bank_interest_spread_proxy": ("bank_net_interest_income", "bank_interest_income"),
            "bank_cost_to_interest_income": ("bank_interest_expense", "bank_interest_income"),
            "bank_provision_to_loans": ("bank_credit_loss_provision", "bank_customer_loans_gross"),
            "bank_provision_to_pre_provision_profit": ("bank_credit_loss_provision", "bank_pre_provision_profit"),
            "bank_loan_loss_allowance_to_loans": ("bank_loan_loss_allowance", "bank_customer_loans_gross"),
            "bank_loan_to_deposit": ("bank_customer_loans_gross", "bank_customer_deposits"),
            "bank_deposit_to_asset": ("bank_customer_deposits", "total_assets"),
            "bank_loan_to_asset": ("bank_customer_loans_gross", "total_assets"),
            "bank_pre_provision_profit_to_asset": ("bank_pre_provision_profit", "total_assets"),
        }
        for new_col, (num, den) in bank_ratio_specs.items():
            if {num, den}.issubset(result.columns):
                result[new_col] = result[num] / (np.abs(result[den]) + eps)

        if {"bank_current_deposits", "bank_customer_deposits"}.issubset(result.columns):
            result["bank_casa_proxy"] = result["bank_current_deposits"] / (
                np.abs(result["bank_customer_deposits"]) + eps
            )
        elif {"bank_current_deposits", "bank_ratio_customer_deposits"}.issubset(result.columns):
            result["bank_casa_proxy"] = result["bank_current_deposits"] / (
                np.abs(result["bank_ratio_customer_deposits"]) + eps
            )

        if {"bank_current_deposits", "bank_margin_deposits", "bank_special_deposits", "bank_customer_deposits"}.issubset(result.columns):
            low_cost_deposits = (
                result["bank_current_deposits"].fillna(0)
                + result["bank_margin_deposits"].fillna(0)
                + result["bank_special_deposits"].fillna(0)
            )
            result["bank_low_cost_deposit_ratio_proxy"] = low_cost_deposits / (
                np.abs(result["bank_customer_deposits"]) + eps
            )

        if {"bank_credit_loss_provision", "bank_customer_loans_gross"}.issubset(result.columns):
            result["bank_abs_provision_to_loans"] = (
                result["bank_credit_loss_provision"].abs() / (np.abs(result["bank_customer_loans_gross"]) + eps)
            )
        if {"bank_credit_loss_provision", "bank_pre_provision_profit"}.issubset(result.columns):
            result["bank_abs_provision_to_pre_provision_profit"] = (
                result["bank_credit_loss_provision"].abs() / (np.abs(result["bank_pre_provision_profit"]) + eps)
            )

        bank_warning_cols = [
            "bank_nim_proxy",
            "bank_casa_proxy",
            "bank_low_cost_deposit_ratio_proxy",
            "bank_abs_provision_to_loans",
            "bank_abs_provision_to_pre_provision_profit",
            "bank_loan_loss_allowance_to_loans",
            "bank_loan_to_deposit",
            "bank_customer_loans_gross",
            "bank_customer_deposits",
        ]
        for col in bank_warning_cols:
            if col not in result.columns:
                continue
            grouped = result.groupby("symbol")[col]
            result[f"{col}_chg1q"] = grouped.diff(1)
            result[f"{col}_chg4q"] = grouped.diff(4)
            result[f"{col}_trend_down_1q"] = (result[f"{col}_chg1q"] < 0).astype(float)
            result[f"{col}_trend_down_4q"] = (result[f"{col}_chg4q"] < 0).astype(float)

        if "bank_nim_proxy_chg4q" in result.columns:
            result["bank_nim_warning"] = (result["bank_nim_proxy_chg4q"] < 0).astype(float)
        if "bank_casa_proxy_chg4q" in result.columns:
            result["bank_casa_warning"] = (result["bank_casa_proxy_chg4q"] < 0).astype(float)
        if "bank_abs_provision_to_loans_chg4q" in result.columns:
            result["bank_provision_pressure_warning"] = (
                result["bank_abs_provision_to_loans_chg4q"] > 0
            ).astype(float)
        if "bank_loan_loss_allowance_to_loans_chg4q" in result.columns:
            result["bank_allowance_pressure_warning"] = (
                result["bank_loan_loss_allowance_to_loans_chg4q"] > 0
            ).astype(float)
        if {"bank_customer_loans_gross_g4q", "bank_customer_deposits_g4q"}.issubset(result.columns):
            result["bank_credit_deposit_growth_gap_4q"] = (
                result["bank_customer_loans_gross_g4q"] - result["bank_customer_deposits_g4q"]
            )
            result["bank_credit_growth_warning"] = (
                result["bank_customer_loans_gross_g4q"] < 0
            ).astype(float)

        if "debt_to_equity_quality" in result.columns:
            leverage_threshold = result.groupby("symbol")["debt_to_equity_quality"].transform(
                lambda values: values.shift(1).rolling(8, min_periods=4).quantile(0.80)
            )
            result["high_leverage_flag"] = (
                result["debt_to_equity_quality"] > leverage_threshold
            ).astype(float).where(leverage_threshold.notna() & result["debt_to_equity_quality"].notna())
        if "liquidity_ratio_quality" in result.columns:
            liquidity_threshold = result.groupby("yq_index")["liquidity_ratio_quality"].transform(
                lambda values: values.quantile(0.20)
            )
            result["weak_liquidity_flag"] = (
                (result["liquidity_ratio_quality"] < 1.0)
                | (result["liquidity_ratio_quality"] < liquidity_threshold)
            ).astype(float).where(result["liquidity_ratio_quality"].notna())
        if "operating_cashflow_to_profit" in result.columns:
            cashflow_threshold = result.groupby("yq_index")["operating_cashflow_to_profit"].transform(
                lambda values: values.quantile(0.20)
            )
            result["low_profit_quality_flag"] = (
                (result["operating_cashflow_to_profit"] < 0.5)
                | (result["operating_cashflow_to_profit"] < cashflow_threshold)
            ).astype(float).where(result["operating_cashflow_to_profit"].notna())
        if "profit_growth_volatility" in result.columns:
            volatility_threshold = result.groupby("symbol")["profit_growth_volatility"].transform(
                lambda values: values.shift(1).rolling(8, min_periods=4).quantile(0.80)
            )
            result["volatile_profit_flag"] = (
                result["profit_growth_volatility"] > volatility_threshold
            ).astype(float).where(volatility_threshold.notna() & result["profit_growth_volatility"].notna())
        if "bank_loan_to_asset" in result.columns:
            loan_asset_threshold = result.groupby("symbol")["bank_loan_to_asset"].transform(
                lambda values: values.shift(1).rolling(8, min_periods=4).quantile(0.80)
            )
            result["high_margin_dependency_flag"] = (
                result["bank_loan_to_asset"] > loan_asset_threshold
            ).astype(float).where(loan_asset_threshold.notna() & result["bank_loan_to_asset"].notna())
        risk_cols = [
            col
            for col in (
                "high_leverage_flag",
                "weak_liquidity_flag",
                "low_profit_quality_flag",
                "volatile_profit_flag",
                "high_margin_dependency_flag",
            )
            if col in result.columns
        ]
        if risk_cols:
            result["financial_risk_flag_count"] = result[risk_cols].fillna(0).sum(axis=1)

        return result

    def add_news_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sort_values(["symbol", "yq_index"]).copy()
        news_cols = [
            col
            for col in result.select_dtypes(include=[np.number]).columns
            if (
                col.startswith("news_")
                or col.startswith("event_")
                or col.endswith("_news_count")
                or col.endswith("_news_ratio")
                or col
                in {
                    "avg_sentiment",
                    "min_sentiment",
                    "max_sentiment",
                    "sentiment_std",
                    "avg_news_text_length",
                    "avg_news_source_weight",
                }
            )
            and not re.search(r"_(lag[124]q|trailing4q)$", col)
        ]
        if not news_cols:
            return result

        grouped = result.groupby("symbol", group_keys=False)
        for col in news_cols:
            grouped_col = grouped[col]
            result[f"{col}_lag1q"] = grouped_col.shift(1)
            result[f"{col}_lag2q"] = grouped_col.shift(2)
            result[f"{col}_lag4q"] = grouped_col.shift(4)
            result[f"{col}_trailing4q"] = grouped_col.transform(lambda values: values.rolling(4, min_periods=1).sum())
        lag_cols = [col for col in result.columns if re.search(r"_(lag[124]q|trailing4q)$", col)]
        result[lag_cols] = result[lag_cols].fillna(0)
        return result

    def add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.sort_values(["symbol", "yq_index"]).copy()
        target_profit_col = profit_target_source(result)
        result["target_profit_source"] = target_profit_col or ""
        for horizon in TARGET_HORIZONS:
            if "revenue" in result.columns:
                future = result.groupby("symbol")["revenue"].shift(-horizon)
                current = result["revenue"]
                result[f"target_revenue_growth_{horizon}q"] = (future - current) / (np.abs(current) + 1e-9) * 100
            if target_profit_col is not None:
                future = result.groupby("symbol")[target_profit_col].shift(-horizon)
                current = result[target_profit_col]
                result[f"target_profit_growth_{horizon}q"] = (future - current) / (np.abs(current) + 1e-9) * 100
                target = result[f"target_profit_growth_{horizon}q"]
                result[f"target_profit_up_{horizon}q"] = np.where(target.notna(), (target > 0).astype(int), np.nan)
                result[f"target_strong_profit_up_{horizon}q"] = np.select(
                    [target > 10, target <= 0],
                    [1, 0],
                    default=np.nan,
                )
                result[f"target_growth_class_{horizon}q"] = pd.cut(
                    target,
                    bins=[-np.inf, 0, 10, np.inf],
                    labels=["low", "medium", "high"],
                )
        return result

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.replace([np.inf, -np.inf], np.nan).copy()
        low, high = self.winsorize_quantiles
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in {"year", "quarter", "yq_index"}:
                continue
            if col.endswith("_flag") or col.endswith("_warning") or col == "financial_risk_flag_count":
                continue
            q_low = result[col].quantile(low)
            q_high = result[col].quantile(high)
            if pd.notna(q_low) and pd.notna(q_high) and q_low < q_high:
                result[col] = result[col].clip(q_low, q_high)
        return result

    def build_feature_matrix_one_company(
        self,
        data: Dict[str, pd.DataFrame],
        macro: Optional[pd.DataFrame] = None,
        news_sentiment: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        income = self.standardize_income(data.get("income", pd.DataFrame()))
        if income.empty:
            return pd.DataFrame()

        join_keys = ["symbol", "year", "quarter", "yq_index"]
        result = income.copy()
        for frame in (
            self.standardize_balance(data.get("balance", pd.DataFrame())),
            self.standardize_cashflow(data.get("cashflow", pd.DataFrame())),
            self.standardize_ratios(data.get("ratios", pd.DataFrame())),
            self.compute_market_features(data.get("price", pd.DataFrame())),
        ):
            if not frame.empty:
                result = result.merge(frame, on=join_keys, how="left")

        if macro is not None and not macro.empty:
            result = result.merge(macro, on="year", how="left")

        if news_sentiment is not None and not news_sentiment.empty:
            result = result.merge(news_sentiment, on=join_keys, how="left")
            news_cols = [
                col
                for col in result.columns
                if col.startswith("news_")
                or col.startswith("event_")
                or col.endswith("_news_count")
                or col.endswith("_news_ratio")
                or col in {"avg_sentiment", "min_sentiment", "max_sentiment", "sentiment_std", "avg_news_text_length", "avg_news_source_weight"}
            ]
            for col in news_cols:
                result[col] = result[col].fillna(0)
            result = self.add_news_time_features(result)

        transformer_news = self.load_transformer_news_sentiment("data/raw/news_external/news_merged.csv")
        if transformer_news is not None and not transformer_news.empty:
            symbol = result["symbol"].iloc[0] if "symbol" in result.columns and not result.empty else None
            if symbol:
                ticker_transformer = transformer_news[transformer_news["symbol"] == symbol]
            else:
                ticker_transformer = transformer_news
            if not ticker_transformer.empty:
                result = result.merge(ticker_transformer, on=join_keys, how="left")
                transformer_cols = [col for col in result.columns if col.startswith("transformer_")]
                for col in transformer_cols:
                    result[col] = result[col].fillna(0)
                result = self.add_news_time_features(result)

        result = self.add_financial_features(result)
        result = self.add_targets(result)
        result = self.clean_features(result)
        return result.sort_values(["symbol", "yq_index"]).reset_index(drop=True)

    def build_dataset(
        self,
        raw_root: str | Path = "data/raw",
        macro_path: str | Path | None = "data/raw/macro/macro_data.csv",
        news_path: str | Path | None = "data/processed/news_merged.csv",
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        raw_root = Path(raw_root)
        macro = self.load_macro_data(macro_path)
        news = self.load_news_sentiment(news_path)
        available_tickers = self.discover_tickers(raw_root, tickers)
        frames: List[pd.DataFrame] = []

        for ticker in available_tickers:
            try:
                data = self.load_stock_data(raw_root / ticker)
                ticker_news = news[news["symbol"] == ticker] if not news.empty else news
                features = self.build_feature_matrix_one_company(data, macro=macro, news_sentiment=ticker_news)
                if not features.empty:
                    frames.append(features)
            except Exception as exc:
                logger.warning("Skip %s because feature build failed: %s", ticker, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["symbol", "year", "quarter"])
        combined = combined.sort_values(["symbol", "yq_index"]).reset_index(drop=True)
        logger.info("Built growth dataset: rows=%s cols=%s tickers=%s", *combined.shape, combined["symbol"].nunique())
        return combined

    def build_dataset_from_ml_tree(
        self,
        data_root: str | Path = "data",
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        data_root = Path(data_root)
        macro_path = data_root / "raw" / "macro" / "macro_data.csv"
        news_path = data_root / "raw" / "news_external" / "news_merged.csv"
        macro = self.load_macro_data(macro_path)
        news = self.load_news_sentiment(news_path)
        available_tickers = self.discover_tickers_ml_tree(data_root, tickers)
        frames: List[pd.DataFrame] = []

        for ticker in available_tickers:
            try:
                data = self.load_stock_data_ml_tree(data_root, ticker)
                ticker_news = news[news["symbol"] == ticker] if not news.empty and "symbol" in news.columns else news
                features = self.build_feature_matrix_one_company(data, macro=macro, news_sentiment=ticker_news)
                if not features.empty:
                    frames.append(features)
            except Exception as exc:
                logger.warning("Skip %s because ML-tree feature build failed: %s", ticker, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["symbol", "year", "quarter"])
        combined = combined.sort_values(["symbol", "yq_index"]).reset_index(drop=True)
        logger.info("Built ML-tree growth dataset: rows=%s cols=%s tickers=%s", *combined.shape, combined["symbol"].nunique())
        return combined


def select_feature_columns(df: pd.DataFrame, target_col: str, min_non_null_ratio: float = 0.15) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = IDENTIFIER_COLS | TARGET_COLS | {target_col}
    candidates = [col for col in numeric_cols if col not in excluded]
    return [col for col in candidates if df[col].notna().mean() >= min_non_null_ratio]
