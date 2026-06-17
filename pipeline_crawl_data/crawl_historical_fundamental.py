"""
Crawl historical quarterly fundamentals from VCI into the ML data tree.

The public vnstock Finance methods expose only the latest periods. The VCI
provider has an internal limit parameter, so this script uses that provider
directly and stores normalized long-form CSVs compatible with the feature
engineering pipeline.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from vnstock import Fundamental
from vnstock.explorer.vci.financial import Finance as VciFinance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import load_symbols, resolve_repo_path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_SYMBOLS = [
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "LPB",
    "VIC", "VHM", "BCM", "GVR", "HPG", "DGC", "FPT", "VNM", "MSN", "MWG",
    "GAS", "PLX", "POW", "REE", "VJC", "SAB", "SSI", "SHB", "VIB", "BVH",
]

REPORTS = {
    "income_statement.csv": "income_statement",
    "balance_sheet.csv": "balance_sheet",
    "cash_flow.csv": "cash_flow",
    "ratio.csv": "ratio",
}


def clean_colname(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_period_key(period: object) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{4})-Q([1-4])", str(period))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def wide_report_to_long(
    df: pd.DataFrame,
    symbol: str,
    start_year: int,
    end_year: int,
    source: str = "vnstock:vci_private_historical",
) -> pd.DataFrame:
    if df.empty:
        return df

    id_col = "item_en" if "item_en" in df.columns else "item_id"
    if id_col not in df.columns:
        return pd.DataFrame()

    period_cols = [
        col for col in df.columns
        if isinstance(col, str) and re.fullmatch(r"\d{4}-Q[1-4](?:_\d+)?", col)
    ]
    if not period_cols:
        return pd.DataFrame()

    # Prefer final report columns when duplicated variants like 2025-Q4_1 exist.
    base_to_col: dict[str, str] = {}
    for col in period_cols:
        base = col.split("_", 1)[0]
        if base not in base_to_col or "_" not in col:
            base_to_col[base] = col

    kept_cols = []
    for base, col in base_to_col.items():
        year, quarter = parse_period_key(base)
        if year is None or quarter is None:
            continue
        if start_year <= year <= end_year:
            kept_cols.append(col)

    if not kept_cols:
        return pd.DataFrame()

    wide = df[[id_col, *kept_cols]].dropna(subset=[id_col]).copy()
    wide[id_col] = wide[id_col].map(clean_colname)
    wide = wide[wide[id_col].ne("")]
    wide = wide.drop_duplicates(subset=[id_col], keep="first")

    long = wide.set_index(id_col)[kept_cols].T.reset_index().rename(columns={"index": "period"})
    long["period"] = long["period"].astype(str).str.split("_", n=1).str[0]
    long["ticker"] = symbol
    long["yearreport"] = long["period"].str.extract(r"(\d{4})").astype(int)
    long["lengthreport"] = long["period"].str.extract(r"Q([1-4])").astype(int)
    long["symbol"] = symbol
    long["source"] = source
    long["collected_at"] = datetime.now().isoformat(timespec="seconds")

    front_cols = ["ticker", "yearreport", "lengthreport", "period", "symbol", "source", "collected_at"]
    other_cols = [col for col in long.columns if col not in front_cols]
    long = long[front_cols + other_cols]
    long = long.sort_values(["yearreport", "lengthreport"]).reset_index(drop=True)
    return long


def fetch_report(symbol: str, report_type: str, start_year: int, end_year: int, limit: int) -> pd.DataFrame:
    finance = VciFinance(symbol, period="quarter", get_all=True, show_log=False)
    df = finance._get_financial_report(
        report_type,
        period="quarter",
        lang="en",
        get_all=True,
        dropna=False,
        limit=limit,
    )
    return wide_report_to_long(df, symbol, start_year=start_year, end_year=end_year)


def fetch_ratio_fallback(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch latest ratio periods from the public API when VCI history is empty."""
    df = Fundamental().equity(symbol=symbol).ratio(period="quarter", lang="en")
    return wide_report_to_long(
        df,
        symbol,
        start_year=start_year,
        end_year=end_year,
        source="vnstock:fundamental_public_ratio",
    )


def save_df(df: pd.DataFrame, path: Path) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl historical VCI fundamentals into data/raw/fundamental.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbols", default="", help="Comma-separated tickers. Empty means discover from the pipeline data tree.")
    parser.add_argument("--symbols-file", default="", help="Optional CSV with a symbol column.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--pause", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    symbols = load_symbols(args.symbols, data_root=data_root, symbols_file=args.symbols_file) or DEFAULT_SYMBOLS

    for i, symbol in enumerate(symbols, start=1):
        logger.info("[%s/%s] Historical fundamentals %s", i, len(symbols), symbol)
        for filename, report_type in REPORTS.items():
            try:
                df = fetch_report(symbol, report_type, args.start_year, args.end_year, args.limit)
                if df.empty and report_type == "ratio":
                    df = fetch_ratio_fallback(symbol, args.start_year, args.end_year)
                if df.empty:
                    logger.warning("%s %s returned empty", symbol, report_type)
                    continue
                save_df(df, data_root / "raw" / "fundamental" / symbol / filename)
                logger.info("%s %s rows=%s range=%s-%s", symbol, filename, len(df), df["period"].min(), df["period"].max())
            except Exception as exc:
                logger.warning("%s %s failed: %s", symbol, report_type, exc)
            time.sleep(args.pause)


if __name__ == "__main__":
    main()

