"""Crawl ticker-related financial news from RSS/search feeds.

The crawler writes rows compatible with data/raw/news_external/news_merged.csv.
It is intentionally conservative: symbol is the requested ticker, while later
normalization can validate symbols and deduplicate against existing news.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_crawl_data.pipeline_common import load_symbols, resolve_repo_path


DEFAULT_DOMAINS = [
    "cafef.vn",
    "vietstock.vn",
    "tinnhanhchungkhoan.vn",
    "vietnamfinance.vn",
    "vietnambiz.vn",
    "ndh.vn",
    "vneconomy.vn",
    "vnexpress.net",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl financial news via Google News RSS.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbols", default="", help="Comma-separated tickers. Empty means discover from the pipeline data tree.")
    parser.add_argument("--symbols-file", default="", help="Optional CSV with a symbol column.")
    parser.add_argument("--company-map", default="data/raw/reference/equity/list.csv")
    parser.add_argument("--output", default="data/raw/news_external/news_merged.csv")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--limit-per-symbol", type=int, default=80)
    parser.add_argument("--pause", type=float, default=0.5)
    return parser.parse_args()


def read_company_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "symbol" not in df.columns:
        return {}
    name_col = "organ_name" if "organ_name" in df.columns else None
    result = {}
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        result[symbol] = str(row[name_col]).strip() if name_col else ""
    return result


def google_news_rss_url(symbol: str, company_name: str, domains: list[str], days: int) -> str:
    domain_query = " OR ".join(f"site:{domain}" for domain in domains)
    terms = [f'"{symbol}"']
    if company_name:
        terms.append(f'"{company_name}"')
    query = f"({' OR '.join(terms)}) ({domain_query}) when:{days}d"
    params = urllib.parse.urlencode({"q": query, "hl": "vi", "gl": "VN", "ceid": "VN:vi"})
    return f"https://news.google.com/rss/search?{params}"


def fetch_xml(url: str) -> ET.Element:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return ET.fromstring(response.read())


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    return " ".join(text.split())


def crawl_symbol(symbol: str, company_name: str, domains: list[str], days: int, limit: int) -> list[dict[str, str]]:
    url = google_news_rss_url(symbol, company_name, domains, days)
    root = fetch_xml(url)
    rows = []
    for item in root.findall(".//item")[:limit]:
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        published = clean_text(item.findtext("pubDate"))
        source_node = item.find("source")
        source = clean_text(source_node.text if source_node is not None else "")
        content = clean_text(item.findtext("description"))
        content_hash = hashlib.sha256(f"{symbol}|{title}|{link}".encode("utf-8", errors="ignore")).hexdigest()
        rows.append(
            {
                "keyword": symbol,
                "symbol": symbol,
                "ticker": symbol,
                "company_name": company_name or symbol,
                "title": title,
                "url": link,
                "date": published,
                "content": content,
                "source": source or "google_news_rss",
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "source_system": "google_news_rss",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    symbols = load_symbols(args.symbols, data_root=args.data_root, symbols_file=args.symbols_file)
    if not symbols:
        raise ValueError("No symbols found. Pass --symbols/--symbols-file or populate data/raw/fundamental.")
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    company_map = read_company_map(resolve_repo_path(args.company_map))
    rows = []
    for symbol in symbols:
        try:
            rows.extend(crawl_symbol(symbol, company_map.get(symbol, ""), domains, args.days, args.limit_per_symbol))
        except Exception as exc:
            rows.append(
                {
                    "keyword": symbol,
                    "symbol": symbol,
                    "ticker": symbol,
                    "company_name": company_map.get(symbol, symbol),
                    "title": "",
                    "url": "",
                    "date": "",
                    "content": "",
                    "source": "google_news_rss",
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "debug_reason": str(exc),
                    "source_system": "google_news_rss",
                }
            )
        time.sleep(args.pause)

    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if output.exists() and output.stat().st_size > 0:
        existing = pd.read_csv(output)
        new_df = pd.concat([existing, new_df], ignore_index=True, sort=False)
    if "content_hash" in new_df.columns:
        new_df = new_df.drop_duplicates(subset=["content_hash"], keep="last")
    elif "url" in new_df.columns:
        new_df = new_df.drop_duplicates(subset=["url"], keep="last")
    new_df.to_csv(output, index=False)
    print(f"Saved {len(new_df)} rows to {output}")


if __name__ == "__main__":
    main()

