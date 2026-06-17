"""
Crawler for the compact ML data tree.

Writes directly to:
  data/raw/fundamental/<SYMBOL>/
  data/raw/market/equity/<SYMBOL>/
  data/raw/reference/
  data/raw/macro/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import load_symbols, parse_csv_list, resolve_repo_path

VnstockCompany = None
VnstockFinance = None
VnstockFundamental = None
VnstockListing = None
VnstockMarket = None
VnstockQuote = None
VnstockReference = None
VnstockMacro = None

try:
    from vnstock import Company as VnstockCompany
    from vnstock import Finance as VnstockFinance
    from vnstock import Fundamental as VnstockFundamental
    from vnstock import Listing as VnstockListing
    from vnstock import Market as VnstockMarket
    from vnstock import Quote as VnstockQuote
    from vnstock import Reference as VnstockReference

    VNSTOCK_AVAILABLE = True
    VNSTOCK_DATA_AVAILABLE = True
except Exception:
    VNSTOCK_AVAILABLE = False
    VNSTOCK_DATA_AVAILABLE = False

try:
    from vnstock import Macro as VnstockMacro

    VNSTOCK_MACRO_AVAILABLE = True
except Exception:
    VNSTOCK_MACRO_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_SYMBOLS = [
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "LPB",
    "VIC", "VHM", "BCM", "GVR", "HPG", "DGC", "FPT", "VNM", "MSN", "MWG",
    "GAS", "PLX", "POW", "REE", "VJC", "SAB", "SSI", "SHB", "VIB", "BVH",
]

VNSTOCK_DATA_SOURCES = ["kbs", "vci", "tcbs", "mas"]
VNSTOCK_SOURCES = ["kbs", "vci"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower().replace(" ", "_") for col in out.columns]
    return out


def save_df(df: pd.DataFrame, path: Path) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    ensure_dir(path.parent)
    normalize_columns(df).to_csv(path, index=False, encoding="utf-8-sig")


def safe_call(name: str, fn: Callable[[], pd.DataFrame], retries: int = 2, delay: float = 0.75) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = fn()
            if isinstance(result, pd.DataFrame):
                return normalize_columns(result)
            if result is None:
                return pd.DataFrame()
            return normalize_columns(pd.DataFrame(result))
        except Exception as exc:
            last_error = exc
            logger.warning("%s failed on attempt %s/%s: %s", name, attempt + 1, retries + 1, exc)
            time.sleep(delay * (attempt + 1))
    logger.error("%s failed permanently: %s", name, last_error)
    return pd.DataFrame()


def first_non_empty(calls: list[tuple[str, Callable[[], pd.DataFrame]]]) -> tuple[pd.DataFrame, str]:
    for source_name, fn in calls:
        df = safe_call(source_name, fn)
        if not df.empty:
            return df, source_name
    return pd.DataFrame(), ""


def add_metadata(df: pd.DataFrame, symbol: str | None = None, source: str = "vnstock") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if symbol and "symbol" not in out.columns:
        out["symbol"] = symbol
    out["source"] = source
    out["collected_at"] = datetime.now().isoformat(timespec="seconds")
    return out


class VnstockMLTreeCrawler:
    def __init__(self, data_root: str | Path = "data", start: str = "2015-01-01", end: str | None = None, pause: float = 0.4):
        if not VNSTOCK_AVAILABLE and not VNSTOCK_DATA_AVAILABLE:
            raise ImportError("vnstock_data or vnstock is required for the compact ML crawler.")
        self.data_root = resolve_repo_path(data_root)
        self.start = start
        self.end = end or datetime.now().strftime("%Y-%m-%d")
        self.pause = pause

    def crawl_symbols(self) -> None:
        if VNSTOCK_DATA_AVAILABLE:
            calls = [
                ("vnstock_data:kbs:Reference.equity.list", lambda: VnstockReference().equity.list(source="kbs")),
                ("vnstock_data:vci:Reference.equity.list", lambda: VnstockReference().equity.list(source="vci")),
                ("vnstock:kbs:Listing.all_symbols", lambda: VnstockListing(source="kbs").all_symbols()),
            ]
            df, source = first_non_empty(calls)
        else:
            df, source = first_non_empty([
                ("vnstock:vci:Listing.all_symbols", lambda: VnstockListing(source="vci").all_symbols()),
            ])
        df = add_metadata(df, source=source)
        save_df(df, self.data_root / "catalog" / "symbols.csv")
        save_df(df, self.data_root / "raw" / "reference" / "equity" / "list.csv")

    def crawl_equity_reference_details(self) -> None:
        if not VNSTOCK_DATA_AVAILABLE:
            return
        reference = VnstockReference().equity
        calls = {
            "list_by_group.csv": lambda: reference.list_by_group(source="kbs"),
            "list_by_exchange.csv": lambda: reference.list_by_exchange(source="kbs"),
            "list_by_industry.csv": lambda: reference.list_by_industry(source="vci"),
        }
        for filename, fn in calls.items():
            df = add_metadata(safe_call(f"Reference.equity.{filename}", fn), source="vnstock_data")
            save_df(df, self.data_root / "raw" / "reference" / "equity" / filename)
            time.sleep(self.pause)

    def crawl_company_reference(self, symbol: str) -> None:
        mapping = {
            "overview.csv": "info",
            "shareholders.csv": "shareholders",
            "officers.csv": "officers",
            "subsidiaries.csv": "subsidiaries",
            "news.csv": "news",
        }
        for filename, method_name in mapping.items():
            calls = []
            if VNSTOCK_DATA_AVAILABLE:
                calls.extend(
                    (
                        f"vnstock_data:{source}:Reference.company.{method_name}({symbol})",
                        lambda source=source, method_name=method_name: getattr(
                            VnstockReference().company(symbol),
                            method_name,
                        )(source=source),
                    )
                    for source in VNSTOCK_SOURCES
                )
            if VNSTOCK_AVAILABLE:
                fallback_method = "overview" if method_name == "info" else method_name
                calls.extend(
                    (
                        f"vnstock:{source}:Company.{fallback_method}({symbol})",
                        lambda source=source, fallback_method=fallback_method: getattr(
                            VnstockCompany(symbol=symbol, source=source.upper()),
                            fallback_method,
                        )(),
                    )
                    for source in VNSTOCK_SOURCES
                )
            df, source = first_non_empty(calls)
            df = add_metadata(df, symbol=symbol, source=source or "unavailable")
            save_df(df, self.data_root / "raw" / "reference" / "company" / symbol / filename)
            time.sleep(self.pause)

    def crawl_fundamental(self, symbol: str) -> None:
        mapping = {
            "income_statement.csv": "income_statement",
            "balance_sheet.csv": "balance_sheet",
            "cash_flow.csv": "cash_flow",
            "ratio.csv": "ratio",
        }
        for filename, method_name in mapping.items():
            calls = []
            if VNSTOCK_DATA_AVAILABLE:
                calls.append((
                    f"vnstock_data:Fundamental.equity.{method_name}({symbol})",
                    lambda method_name=method_name: self._crawl_fundamental_vnstock_data(symbol, method_name),
                ))
            if VNSTOCK_AVAILABLE:
                calls.extend(
                    (
                        f"vnstock:{source}:Finance.{method_name}({symbol})",
                        lambda source=source, method_name=method_name: getattr(
                            VnstockFinance(symbol=symbol, source=source),
                            method_name,
                        )(period="quarter"),
                    )
                    for source in VNSTOCK_SOURCES
                )
            df, source = first_non_empty(calls)
            df = add_metadata(df, symbol=symbol, source=source or "unavailable")
            save_df(df, self.data_root / "raw" / "fundamental" / symbol / filename)
            time.sleep(self.pause)

    def _crawl_fundamental_vnstock_data(self, symbol: str, method_name: str) -> pd.DataFrame:
        equity = VnstockFundamental().equity(symbol)
        method = getattr(equity, method_name)
        if method_name == "ratio":
            return method(orient="time_series")
        return method(period="quarter", orient="time_series")

    def crawl_market(self, symbol: str) -> None:
        calls = []
        if VNSTOCK_DATA_AVAILABLE:
            calls.extend(
                (
                    f"vnstock_data:{source}:Market.equity.ohlcv({symbol})",
                    lambda source=source: VnstockMarket().equity(symbol).ohlcv(
                        start=self.start,
                        end=self.end,
                        resolution="1D",
                        source=source,
                    ),
                )
                for source in VNSTOCK_DATA_SOURCES
            )
        if VNSTOCK_AVAILABLE:
            calls.extend(
                (
                    f"vnstock:{source}:Quote.history({symbol})",
                    lambda source=source: VnstockQuote(symbol=symbol, source=source).history(
                        start=self.start,
                        end=self.end,
                        interval="1D",
                    ),
                )
                for source in VNSTOCK_SOURCES
            )
        df, source = first_non_empty(calls)
        df = add_metadata(df, symbol=symbol, source=source or "unavailable")
        save_df(df, self.data_root / "raw" / "market" / "equity" / symbol / "ohlcv.csv")

    def crawl_macro(self) -> None:
        if not VNSTOCK_MACRO_AVAILABLE:
            logger.info("Macro API is unavailable; skip macro crawl.")
            return
        economy = VnstockMacro.economy()
        currency = VnstockMacro.currency()
        calls = {
            "economy_gdp.csv": lambda: economy.gdp(),
            "economy_cpi.csv": lambda: economy.cpi(),
            "currency_interest_rate.csv": lambda: currency.interest_rate(),
            "currency_exchange_rate.csv": lambda: currency.exchange_rate(),
        }
        for filename, fn in calls.items():
            df = add_metadata(safe_call(f"Macro.{filename}", fn), source="vnstock_data")
            save_df(df, self.data_root / "raw" / "macro" / filename)
            time.sleep(self.pause)

    def run(self, symbols: list[str]) -> None:
        self.crawl_symbols()
        self.crawl_equity_reference_details()
        self.crawl_macro()
        for i, symbol in enumerate(symbols, start=1):
            logger.info("[%s/%s] Crawling %s", i, len(symbols), symbol)
            self.crawl_company_reference(symbol)
            self.crawl_fundamental(symbol)
            self.crawl_market(symbol)
            time.sleep(self.pause)


def parse_symbols(value: str) -> list[str]:
    return parse_csv_list(value) or DEFAULT_SYMBOLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl vnstock data into compact ML tree.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbols", default="", help="Comma-separated tickers. Empty means discover from the pipeline data tree, then fallback to DEFAULT_SYMBOLS.")
    parser.add_argument("--symbols-file", default="", help="Optional CSV with a symbol column.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--pause", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawler = VnstockMLTreeCrawler(
        data_root=args.data_root,
        start=args.start,
        end=args.end or None,
        pause=args.pause,
    )
    symbols = load_symbols(args.symbols, data_root=args.data_root, symbols_file=args.symbols_file) or DEFAULT_SYMBOLS
    crawler.run(symbols)


if __name__ == "__main__":
    main()

