"""
Train sector-specific growth models from reference industry labels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def discover_available_symbols(data_root: Path) -> set[str]:
    root = data_root / "raw" / "fundamental"
    if not root.exists():
        return set()
    return {path.name.upper() for path in root.iterdir() if path.is_dir()}


def sector_symbols(
    industry_path: Path,
    sector: str,
    available: set[str],
    exchange_path: Path | None = None,
    exchange: str = "",
) -> list[str]:
    industry = pd.read_csv(industry_path)
    if not {"symbol", "icb_name"}.issubset(industry.columns):
        raise ValueError(f"{industry_path} must contain symbol and icb_name columns.")

    normalized = industry["icb_name"].map(normalize_text)
    sector_norm = normalize_text(sector)
    matched = industry[normalized.str.contains(sector_norm, na=False)].copy()
    symbols = set(matched["symbol"].astype(str).str.upper()) & available
    if exchange:
        if exchange_path is None or not exchange_path.exists():
            raise ValueError("exchange_path is required when --exchange is used.")
        exchange_df = pd.read_csv(exchange_path)
        if not {"symbol", "exchange"}.issubset(exchange_df.columns):
            raise ValueError(f"{exchange_path} must contain symbol and exchange columns.")
        exchange_symbols = set(
            exchange_df[
                exchange_df["exchange"].astype(str).str.upper().eq(exchange.upper())
            ]["symbol"].astype(str).str.upper()
        )
        symbols &= exchange_symbols
    symbols = sorted(symbols)
    if not symbols:
        raise ValueError(f"No available symbols matched sector={sector!r}.")
    return symbols


def run_train(symbols: list[str], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / normalize_text(args.sector).replace(" ", "_").replace("/", "_")
    command = [
        sys.executable,
        "-m",
        "train.train_growth_models",
        "--data-root",
        args.data_root,
        "--data-layout",
        "ml_tree",
        "--target",
        args.target,
        "--tickers",
        ",".join(symbols),
        "--output-dir",
        str(output_dir),
    ]
    print("symbols=" + ",".join(symbols))
    print("output_dir=" + str(output_dir))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {completed.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train growth model for a sector from list_by_industry.csv.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--industry-path", default="data/raw/reference/equity/list_by_industry.csv")
    parser.add_argument("--exchange-path", default="data/raw/reference/equity/list_by_exchange.csv")
    parser.add_argument("--sector", default="Ngân hàng")
    parser.add_argument("--exchange", default="", help="Optional exchange filter, for example HOSE.")
    parser.add_argument("--target", default="target_profit_growth_4q")
    parser.add_argument("--output-dir", default="outputs/growth_forecast_sector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = discover_available_symbols(Path(args.data_root))
    symbols = sector_symbols(
        Path(args.industry_path),
        args.sector,
        available,
        exchange_path=Path(args.exchange_path),
        exchange=args.exchange,
    )
    run_train(symbols, args)


if __name__ == "__main__":
    main()
