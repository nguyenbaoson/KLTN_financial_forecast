"""Normalize external news ticker mapping for the ML feature pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.growth_feature_engineering import (  # noqa: E402
    infer_data_root_from_news_path,
    load_valid_symbols,
    map_news_symbols,
    normalize_symbol_value,
    parse_news_dates,
    strip_accents,
)
from pipeline_crawl_data.pipeline_common import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize news symbol/ticker mapping.")
    parser.add_argument("--input", default="data/raw/news_external/news_merged.csv")
    parser.add_argument("--output", default="data/raw/news_external/news_merged.csv")
    parser.add_argument("--coverage-output", default="outputs/data_quality/news_symbol_coverage.csv")
    parser.add_argument("--backup", action="store_true", help="Create .bak before overwriting --input.")
    parser.add_argument("--keep-low-quality", action="store_true", help="Keep rows that fail relevance/duplicate filters.")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = strip_accents("" if pd.isna(value) else str(value)).lower()
    return " ".join(text.split())


def url_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().str.replace(r"[?#].*$", "", regex=True)


def company_alias_hit(text: pd.Series, company_name: pd.Series) -> pd.Series:
    result = pd.Series(False, index=text.index)
    stopwords = {
        "ctcp",
        "cong",
        "ty",
        "co",
        "phan",
        "tap",
        "doan",
        "tong",
        "ngan",
        "hang",
        "tmcp",
        "tnhh",
        "viet",
        "nam",
    }
    for idx, name in company_name.fillna("").items():
        normalized = normalize_text(name)
        tokens = [token for token in normalized.split() if len(token) >= 3 and token not in stopwords]
        if len(tokens) < 2:
            continue
        long_tokens = sorted(tokens, key=len, reverse=True)[:4]
        hits = sum(1 for token in long_tokens if token in text.loc[idx])
        result.loc[idx] = hits >= min(2, len(long_tokens))
    return result


def quality_filter(news: pd.DataFrame) -> pd.DataFrame:
    result = news.copy()
    title = result.get("title", pd.Series("", index=result.index)).fillna("").astype(str)
    content = result.get("content", pd.Series("", index=result.index)).fillna("").astype(str)
    combined = (title + " " + content).map(normalize_text)
    symbol = result["symbol"].fillna("").astype(str).str.upper()
    company_name = result.get("company_name", pd.Series("", index=result.index)).fillna("").astype(str)

    symbol_pattern_hit = pd.Series(False, index=result.index)
    for idx, sym in symbol.items():
        if not sym:
            continue
        symbol_pattern_hit.loc[idx] = bool(pd.Series([combined.loc[idx]]).str.contains(rf"(?<![a-z0-9]){sym.lower()}(?![a-z0-9])", regex=True).iloc[0])
    alias_hit = company_alias_hit(combined, company_name)

    technical_only = combined.str.contains(
        r"bao cao phan tich ky thuat|phan tich ky thuat|tin hieu ky thuat|technical analysis",
        regex=True,
        na=False,
    )
    business_terms = combined.str.contains(
        r"loi nhuan|doanh thu|co tuc|tang von|trai phieu|du an|m&a|sap nhap|mua lai|no xau|tin dung|"
        r"du phong|ket qua kinh doanh|hop dong|dau tu|nha may|lanh dao|bo nhiem|mien nhiem|xu phat",
        regex=True,
        na=False,
    )
    has_text = title.str.len().gt(0) | content.str.len().gt(0)
    has_date = parse_news_dates(result["date"]) .notna() if "date" in result.columns else pd.Series(False, index=result.index)
    result["news_relevance_score"] = symbol_pattern_hit.astype(int) + alias_hit.astype(int) + business_terms.astype(int)
    result["is_relevant_news"] = result["symbol"].notna() & has_date & has_text & (symbol_pattern_hit | alias_hit) & ~(technical_only & ~business_terms)
    result["news_quality_reason"] = "ok"
    result.loc[result["symbol"].isna(), "news_quality_reason"] = "missing_symbol"
    result.loc[~has_date, "news_quality_reason"] = "missing_date"
    result.loc[~has_text, "news_quality_reason"] = "missing_text"
    result.loc[~(symbol_pattern_hit | alias_hit), "news_quality_reason"] = "symbol_or_company_not_in_text"
    result.loc[technical_only & ~business_terms, "news_quality_reason"] = "technical_generic"

    result["_url_key"] = url_key(result.get("url", pd.Series("", index=result.index)))
    result["_title_key"] = title.map(normalize_text)
    before_dup = result["is_relevant_news"].copy()
    result = result.sort_values(["symbol", "date", "source_system"], na_position="last")
    duplicate_url = result["_url_key"].ne("") & result.duplicated(["symbol", "_url_key"], keep="last")
    duplicate_title = result["_title_key"].ne("") & result.duplicated(["symbol", "_title_key"], keep="last")
    duplicate = duplicate_url | duplicate_title
    result.loc[duplicate, "is_relevant_news"] = False
    result.loc[duplicate & before_dup.reindex(result.index).fillna(False), "news_quality_reason"] = "duplicate"
    return result.drop(columns=["_url_key", "_title_key"]).sort_index()


def normalize_news(input_path: Path) -> pd.DataFrame:
    news = pd.read_csv(input_path)
    data_root = infer_data_root_from_news_path(input_path) or REPO_ROOT / "data"
    valid_symbols = load_valid_symbols(data_root)

    original_ticker = news["ticker"].map(normalize_symbol_value) if "ticker" in news.columns else pd.Series("", index=news.index)
    keyword_symbol = news["keyword"].map(normalize_symbol_value) if "keyword" in news.columns else pd.Series("", index=news.index)
    mapped_symbol = map_news_symbols(news, valid_symbols)

    result = news.copy()
    result["symbol"] = mapped_symbol
    result["ticker"] = result["symbol"]
    result["keyword_symbol"] = keyword_symbol.replace("", pd.NA)
    result["original_ticker"] = original_ticker.replace("", pd.NA)
    result["symbol_mapping_method"] = "unmapped"
    result.loc[result["symbol"].notna() & keyword_symbol.eq(result["symbol"].fillna("")), "symbol_mapping_method"] = "keyword"
    result.loc[
        result["symbol"].notna()
        & result["symbol_mapping_method"].eq("unmapped")
        & original_ticker.eq(result["symbol"].fillna("")),
        "symbol_mapping_method",
    ] = "ticker"
    result.loc[result["symbol"].notna() & result["symbol_mapping_method"].eq("unmapped"), "symbol_mapping_method"] = "text_fallback"
    return quality_filter(result)


def write_coverage(news: pd.DataFrame, coverage_path: Path) -> None:
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    dates = parse_news_dates(news["date"]) if "date" in news.columns else pd.Series(pd.NaT, index=news.index)
    audit = news.copy()
    audit["_date"] = dates
    audit["_yq"] = audit["_date"].dt.year.astype("Int64").astype(str) + "Q" + audit["_date"].dt.quarter.astype("Int64").astype(str)
    if "is_relevant_news" in audit.columns:
        mapped = audit.dropna(subset=["symbol"])
        usable = mapped[mapped["is_relevant_news"].astype(bool)]
    else:
        mapped = audit.dropna(subset=["symbol"])
        usable = mapped
    rows = []
    for symbol, frame in mapped.groupby("symbol"):
        usable_frame = usable[usable["symbol"].eq(symbol)]
        rows.append(
            {
                "symbol": symbol,
                "article_count": int(len(frame)),
                "usable_article_count": int(len(usable_frame)),
                "quarter_count": int(usable_frame["_yq"].dropna().nunique()),
                "first_date": usable_frame["_date"].min(),
                "last_date": usable_frame["_date"].max(),
                "keyword_mapped": int(frame["symbol_mapping_method"].eq("keyword").sum()),
                "ticker_mapped": int(frame["symbol_mapping_method"].eq("ticker").sum()),
                "text_fallback_mapped": int(frame["symbol_mapping_method"].eq("text_fallback").sum()),
            }
        )
    pd.DataFrame(rows).sort_values("article_count", ascending=False).to_csv(coverage_path, index=False)


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    normalized = normalize_news(input_path)
    if not args.keep_low_quality and "is_relevant_news" in normalized.columns:
        normalized = normalized[normalized["is_relevant_news"].astype(bool)].copy()
    if args.backup and output_path.resolve() == input_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    coverage_output = resolve_repo_path(args.coverage_output)
    write_coverage(normalized, coverage_output)

    mapped = int(normalized["symbol"].notna().sum())
    print(f"Rows: {len(normalized)}")
    print(f"Mapped rows: {mapped}")
    print(f"Mapped symbols: {normalized['symbol'].nunique(dropna=True)}")
    print(f"Output: {output_path}")
    print(f"Coverage: {coverage_output}")


if __name__ == "__main__":
    main()

