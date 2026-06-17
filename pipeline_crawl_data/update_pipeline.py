"""
Run operational update pipelines for the ML data tree.

Modes:
  daily      refresh market/news layers, then materialize insights.
  quarterly  refresh fundamentals, optionally refresh daily layers, then materialize and retrain models.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.vnstock_ml_tree_crawler import (  # noqa: E402
    VnstockMarket,
    VnstockQuote,
    add_metadata,
    first_non_empty,
    normalize_columns,
    save_df,
)
from pipeline_crawl_data.pipeline_common import load_symbols, parse_csv_list, resolve_repo_path  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_INDEX_SYMBOLS = ["VNINDEX", "VN30", "HNX30"]


def run_cmd(command: list[str], cwd: Path | None = None, required: bool = True) -> bool:
    logger.info("Run: %s", " ".join(command))
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command, cwd=str(cwd or REPO_ROOT), check=False, env=env)
    if completed.returncode != 0:
        message = f"Command failed ({completed.returncode}): {' '.join(command)}"
        if required:
            raise RuntimeError(message)
        logger.warning(message)
        return False
    return True


def find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).lower() in {"date", "time", "trading_date"}:
            return col
    return None


def latest_start(path: Path, fallback_start: str, overlap_days: int) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return fallback_start
    try:
        df = pd.read_csv(path)
    except Exception:
        return fallback_start
    date_col = find_date_column(df)
    if not date_col:
        return fallback_start
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return fallback_start
    start = dates.max().to_pydatetime() - timedelta(days=overlap_days)
    return start.strftime("%Y-%m-%d")


def merge_ohlcv(existing_path: Path, new_df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    frames = []
    if existing_path.exists() and existing_path.stat().st_size > 0:
        try:
            frames.append(pd.read_csv(existing_path))
        except Exception as exc:
            logger.warning("Cannot read existing OHLCV %s: %s", existing_path, exc)
    if isinstance(new_df, pd.DataFrame) and not new_df.empty:
        frames.append(add_metadata(new_df, symbol=symbol, source=source))
    if not frames:
        return pd.DataFrame()

    merged = normalize_columns(pd.concat(frames, ignore_index=True, sort=False))
    date_col = find_date_column(merged)
    if date_col:
        parsed_dates = pd.to_datetime(merged[date_col], errors="coerce")
        merged["_date_sort"] = parsed_dates
        merged["_date_key"] = parsed_dates.dt.strftime("%Y-%m-%d")
        merged = merged.sort_values("_date_sort").drop_duplicates(subset=["_date_key"], keep="last")
        merged = merged.drop(columns=["_date_key", "_date_sort"])
    else:
        merged = merged.drop_duplicates()
    return merged.reset_index(drop=True)


def fetch_equity_ohlcv(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    calls = [
        (
            f"vnstock_data:kbs:Market.equity.ohlcv({symbol})",
            lambda: VnstockMarket().equity(symbol).ohlcv(start=start, end=end, resolution="1D", source="kbs"),
        ),
        (
            f"vnstock_data:vci:Market.equity.ohlcv({symbol})",
            lambda: VnstockMarket().equity(symbol).ohlcv(start=start, end=end, resolution="1D", source="vci"),
        ),
        (
            f"vnstock:vci:Quote.history({symbol})",
            lambda: VnstockQuote(symbol=symbol, source="vci").history(start=start, end=end, interval="1D"),
        ),
        (
            f"vnstock:kbs:Quote.history({symbol})",
            lambda: VnstockQuote(symbol=symbol, source="kbs").history(start=start, end=end, interval="1D"),
        ),
    ]
    return first_non_empty(calls)


def fetch_index_ohlcv(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    calls = [
        (
            f"vnstock:vci:Quote.history({symbol})",
            lambda: VnstockQuote(symbol=symbol, source="vci").history(start=start, end=end, interval="1D"),
        ),
        (
            f"vnstock:kbs:Quote.history({symbol})",
            lambda: VnstockQuote(symbol=symbol, source="kbs").history(start=start, end=end, interval="1D"),
        ),
    ]
    return first_non_empty(calls)


def refresh_market(
    data_root: Path,
    symbols: list[str],
    indexes: list[str],
    fallback_start: str,
    end: str,
    overlap_days: int,
    pause: float,
) -> None:
    for i, symbol in enumerate(symbols, start=1):
        path = data_root / "raw" / "market" / "equity" / symbol / "ohlcv.csv"
        start = latest_start(path, fallback_start, overlap_days)
        logger.info("[%s/%s] Market %s from %s to %s", i, len(symbols), symbol, start, end)
        df, source = fetch_equity_ohlcv(symbol, start, end)
        merged = merge_ohlcv(path, df, symbol=symbol, source=source or "unavailable")
        save_df(merged, path)
        time.sleep(pause)

    for i, symbol in enumerate(indexes, start=1):
        path = data_root / "raw" / "market" / "index" / symbol / "ohlcv.csv"
        start = latest_start(path, fallback_start, overlap_days)
        logger.info("[%s/%s] Index %s from %s to %s", i, len(indexes), symbol, start, end)
        df, source = fetch_index_ohlcv(symbol, start, end)
        merged = merge_ohlcv(path, df, symbol=symbol, source=source or "unavailable")
        save_df(merged, path)
        time.sleep(pause)


def refresh_news(raw_news_path: Path, chunks_path: Path, rebuild_index: bool) -> None:
    raw_news_path = raw_news_path.resolve()
    chunks_path = chunks_path.resolve()
    if not raw_news_path.exists():
        logger.warning("News input not found, skip news/RAG update: %s", raw_news_path)
        return

    run_cmd([
        sys.executable,
        "pipeline_crawl_data/normalize_news_external.py",
        "--input",
        str(raw_news_path),
        "--output",
        str(raw_news_path),
        "--coverage-output",
        "outputs/data_quality/news_symbol_coverage.csv",
    ], cwd=REPO_ROOT, required=False)

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-c",
        (
            "from run_pipeline import run_full_pipeline; "
            f"run_full_pipeline(r'{raw_news_path}', r'{chunks_path}')"
        ),
    ]
    run_cmd(command, cwd=REPO_ROOT / "src" / "preprocessing", required=False)
    if rebuild_index and chunks_path.exists():
        run_cmd([sys.executable, "build_index.py"], cwd=REPO_ROOT / "src" / "rag", required=False)


def crawl_news_rss(news_path: Path, symbols: list[str], days: int, pause: float) -> None:
    if not symbols:
        return
    run_cmd([
        sys.executable,
        "pipeline_crawl_data/crawl_news_rss.py",
        "--symbols",
        ",".join(symbols),
        "--output",
        str(news_path),
        "--days",
        str(days),
        "--pause",
        str(pause),
    ], cwd=REPO_ROOT, required=False)


def materialize_clean_audit(data_root: Path) -> None:
    clean_roots = ",".join(
        str(path)
        for path in [
            data_root / "raw",
            data_root / "features",
            data_root / "raw" / "analytics",
            data_root / "raw" / "insights",
        ]
    )
    run_cmd([sys.executable, "pipeline_crawl_data/clean_csv_quality.py", "--roots", clean_roots], cwd=REPO_ROOT)
    run_cmd([sys.executable, "pipeline_crawl_data/materialize_ml_outputs.py", "--data-root", str(data_root)], cwd=REPO_ROOT)
    run_cmd([
        sys.executable,
        "pipeline_crawl_data/audit_ml_tree.py",
        "--data-root",
        str(data_root),
        "--output",
        "outputs/data_quality/ml_tree_audit.csv",
    ], cwd=REPO_ROOT)


def refresh_daily_layers(args: argparse.Namespace, symbols: list[str], data_root: Path) -> None:
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    indexes = parse_csv_list(args.indexes)
    refresh_market(
        data_root=data_root,
        symbols=symbols,
        indexes=indexes,
        fallback_start=args.start,
        end=end,
        overlap_days=args.overlap_days,
        pause=args.pause,
    )
    if args.crawl_news_rss:
        crawl_news_rss(resolve_repo_path(args.news_path), symbols, days=args.news_days, pause=args.pause)
    refresh_news(resolve_repo_path(args.news_path), resolve_repo_path(args.chunks_path), rebuild_index=not args.skip_rag_index)


def run_daily(args: argparse.Namespace, symbols: list[str]) -> None:
    data_root = resolve_repo_path(args.data_root)
    refresh_daily_layers(args, symbols, data_root)
    materialize_clean_audit(data_root)


def run_quarterly(args: argparse.Namespace, symbols: list[str]) -> None:
    data_root = resolve_repo_path(args.data_root)
    end_year = int(args.end_year or datetime.now().year)
    run_cmd([
        sys.executable,
        "pipeline_crawl_data/crawl_historical_fundamental.py",
        "--data-root",
        str(data_root),
        "--symbols",
        ",".join(symbols),
        "--start-year",
        str(args.start_year),
        "--end-year",
        str(end_year),
        "--limit",
        str(args.fundamental_limit),
        "--pause",
        str(args.pause),
    ], cwd=REPO_ROOT)

    if args.include_daily:
        refresh_daily_layers(args, symbols, data_root)

    materialize_clean_audit(data_root)
    if args.skip_train:
        logger.info("Skip training because --skip-train is set.")
        return

    run_cmd([
        sys.executable,
        "-m",
        "train.train_growth_models",
        "--data-root",
        str(data_root),
        "--data-layout",
        "ml_tree",
        "--target",
        args.target,
    ], cwd=REPO_ROOT)
    run_cmd([
        sys.executable,
        "-m",
        "train.compare_growth_models",
        "--data-root",
        str(data_root),
        "--target",
        args.target,
    ], cwd=REPO_ROOT, required=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily or quarterly data update pipeline.")
    parser.add_argument("mode", choices=["daily", "quarterly"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Empty means data/raw/fundamental folders.")
    parser.add_argument("--symbols-file", default="", help="Optional CSV with a symbol column.")
    parser.add_argument("--indexes", default=",".join(DEFAULT_INDEX_SYMBOLS))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--overlap-days", type=int, default=10)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--news-path", default="data/raw/news_external/news_merged.csv")
    parser.add_argument("--chunks-path", default="data/processed/chunks.jsonl")
    parser.add_argument("--crawl-news-rss", action="store_true", help="Append Google News RSS articles before normalizing news.")
    parser.add_argument("--news-days", type=int, default=3650)
    parser.add_argument("--skip-rag-index", action="store_true")
    parser.add_argument("--include-daily", action="store_true", help="Run daily refresh before quarterly fundamental/train.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", default="")
    parser.add_argument("--fundamental-limit", type=int, default=60)
    parser.add_argument("--target", default="target_profit_growth_1q")
    parser.add_argument("--skip-train", action="store_true", help="For quarterly mode, stop after crawl/materialize/audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    symbols = load_symbols(args.symbols, data_root=data_root, symbols_file=args.symbols_file)
    if not symbols:
        raise ValueError("No symbols found. Pass --symbols or populate data/raw/fundamental.")

    if args.mode == "daily":
        run_daily(args, symbols)
    else:
        run_quarterly(args, symbols)


if __name__ == "__main__":
    main()

