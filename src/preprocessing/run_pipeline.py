import re
import json
from pathlib import Path
import pandas as pd

from cleaner import clean_text
from entity_normalizer import deduplicate, enrich_metadata
from sentiment import add_sentiment_to_df
from chunker import create_chunks_with_metadata, save_chunks


DATE_PATTERNS = [
    # Ví dụ: 25-03-2026 - 16:40 PM
    re.compile(r"^\s*(\d{2}-\d{2}-\d{4}\s*-\s*\d{2}:\d{2}\s*[AP]M)\s*"),
    # Ví dụ: 26/03/2026 06:02
    re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})\s*"),
    # Ví dụ: 25/03/2026 - 18:05
    re.compile(r"^\s*(\d{2}/\d{2}/\d{4}\s*-\s*\d{2}:\d{2})\s*"),
]


def is_missing_date(value) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in ["", "nan", "none", "nat"]


def recover_date_from_text(text: str):
    text = str(text) if pd.notna(text) else ""
    for pattern in DATE_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip()
    return None


def remove_date_prefix(text: str) -> str:
    text = str(text) if pd.notna(text) else ""
    for pattern in DATE_PATTERNS:
        if pattern.match(text):
            return pattern.sub("", text).strip()
    return text.strip()


def recover_date_column(df: pd.DataFrame) -> pd.DataFrame:
    recovered_count = 0

    if "date" not in df.columns:
        df["date"] = pd.NA

    for idx in df.index:
        current_date = df.at[idx, "date"]
        content = df.at[idx, "content"] if "content" in df.columns else ""

        if is_missing_date(current_date):
            recovered_date = recover_date_from_text(content)
            if recovered_date:
                df.at[idx, "date"] = recovered_date
                recovered_count += 1

    print(f"       Recover được date cho {recovered_count} bài")
    return df


def clean_date_prefix_from_content(df: pd.DataFrame) -> pd.DataFrame:
    if "content" in df.columns:
        df["content"] = df["content"].apply(remove_date_prefix)
    return df


def run_full_pipeline(input_csv: str, output_jsonl: str):
    print("=" * 50)
    print("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ")
    print("=" * 50)

    # 1. Đọc dữ liệu thô
    df = pd.read_csv(input_csv)
    print(f"[1/6] Đọc dữ liệu: {len(df)} bài")

    # Đảm bảo các cột text tồn tại
    for col in ["title", "content", "date", "source", "url", "keyword"]:
        if col not in df.columns:
            df[col] = pd.NA

    # 2. Làm sạch văn bản
    print("[2/6] Làm sạch văn bản...")
    df["content"] = df["content"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)

    # Loại bài không có nội dung sau khi làm sạch
    df = df[df["content"].fillna("").str.len() > 100].reset_index(drop=True)
    print(f"       Còn {len(df)} bài sau làm sạch")

    # 3. Recover date + remove date khỏi đầu content
    print("[3/6] Recover date và loại date khỏi content...")
    df = recover_date_column(df)
    df = clean_date_prefix_from_content(df)

    # 4. Loại trùng + chuẩn hóa thực thể
    print("[4/6] Loại bài trùng và chuẩn hóa tên công ty...")
    df = deduplicate(df)
    df = enrich_metadata(df)

    # 5. Sentiment analysis
    print("[5/6] Phân tích sentiment...")
    df = add_sentiment_to_df(df)
    print("       Phân bố sentiment:")
    print(df["sentiment_label"].value_counts().to_string())

    # 6. Chunking
    print("[6/6] Chunking cho RAG...")
    chunks = create_chunks_with_metadata(df, chunk_size=5, overlap=1)
    save_chunks(chunks, output_jsonl)

    print("\nHOÀN THÀNH!")
    print(f"Tổng chunks: {len(chunks)}")
    print(f"Trung bình chunks/bài: {len(chunks)/len(df):.1f}")

    # Lưu CSV đã xử lý
    processed_csv = input_csv.replace("raw", "processed")
    Path(processed_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False, encoding="utf-8-sig")
    print(f"Đã lưu CSV processed: {processed_csv}")

    return chunks


if __name__ == "__main__":
    chunks = run_full_pipeline(
        input_csv="../../data/raw/news_merged.csv",
        output_jsonl="../../data/processed/chunks.jsonl"
    )

    # Xem thử 2 chunk đầu tiên
    print("\nMẪU CHUNK:")
    for c in chunks[:2]:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        print("---")
