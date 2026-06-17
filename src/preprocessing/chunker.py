import json
from typing import List, Dict

def split_into_sentences(text: str) -> List[str]:
    """
    Tách câu thông minh cho tiếng Việt.
    Không dùng underthesea để tránh phụ thuộc nặng ở bước này.
    """
    import re
    # Tách theo dấu câu kết thúc câu, nhưng giữ dấu trong câu
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Loại câu quá ngắn (< 10 ký tự) — thường là artifact
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def chunk_by_sentences(
    text: str,
    chunk_size: int = 5,      # Số câu mỗi chunk
    overlap: int = 1,          # Số câu overlap giữa các chunk
) -> List[str]:
    """
    Chunk theo câu — tốt hơn chunk theo từ vì giữ nguyên ngữ nghĩa.
    chunk_size=5 câu ≈ 200-400 từ, phù hợp cho RAG.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i: i + chunk_size]
        chunk_text = " ".join(chunk_sentences)
        if len(chunk_text) > 50:  # Bỏ chunk quá ngắn
            chunks.append(chunk_text)
        i += chunk_size - overlap  # Dịch chuyển có overlap

    return chunks


def create_chunks_with_metadata(
    df,
    chunk_size: int = 5,
    overlap: int = 1,
) -> List[Dict]:
    """
    Tạo danh sách chunk với metadata đầy đủ.
    Đây là dữ liệu đầu vào cho Vector DB.
    """
    all_chunks = []
    chunk_id = 0

    for _, row in df.iterrows():
        content = str(row.get("content", ""))
        if not content or len(content) < 100:
            continue

        chunks = chunk_by_sentences(content, chunk_size, overlap)

        for idx, chunk_text in enumerate(chunks):
            chunk_record = {
                # ID duy nhất cho mỗi chunk
                "chunk_id": f"chunk_{chunk_id:06d}",

                # Nội dung chunk
                "text": chunk_text,

                # Metadata — quan trọng cho RAG filter
                "source_url": row.get("url", ""),
                "title": row.get("title", ""),
                "date": row.get("date", ""),
                "company_name": row.get("company_name", ""),
                "ticker": row.get("ticker", ""),
                "sentiment_label": row.get("sentiment_label", "neutral"),
                "sentiment_score": float(row.get("sentiment_score", 0.0)),
                "source": row.get("source", ""),

                # Vị trí chunk trong bài (dùng để rerank)
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            all_chunks.append(chunk_record)
            chunk_id += 1

    return all_chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Lưu chunks dạng JSONL — mỗi dòng 1 JSON object"""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(chunks)} chunks vào {output_path}")