"""Shared helpers for the data crawl/update pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def symbols_from_file(path: str | Path) -> list[str]:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return []
    try:
        df = pd.read_csv(resolved)
    except Exception:
        return []
    if "symbol" not in df.columns:
        return []
    return sorted(
        {
            str(symbol).strip().upper()
            for symbol in df["symbol"].dropna()
            if str(symbol).strip() and str(symbol).strip().upper() != "NAN"
        }
    )


def discover_symbols(data_root: str | Path = "data") -> list[str]:
    root = resolve_repo_path(data_root)

    fundamental_root = root / "raw" / "fundamental"
    if fundamental_root.exists():
        symbols = sorted(path.name.upper() for path in fundamental_root.iterdir() if path.is_dir())
        if symbols:
            return symbols

    for relative in [
        root / "catalog" / "symbols.csv",
        root / "raw" / "reference" / "equity" / "list.csv",
    ]:
        symbols = symbols_from_file(relative)
        if symbols:
            return symbols

    return []


def load_symbols(symbols: str = "", data_root: str | Path = "data", symbols_file: str | Path = "") -> list[str]:
    explicit = parse_csv_list(symbols)
    if explicit:
        return explicit

    if str(symbols_file).strip():
        from_file = symbols_from_file(symbols_file)
        if from_file:
            return from_file

    return discover_symbols(data_root)

