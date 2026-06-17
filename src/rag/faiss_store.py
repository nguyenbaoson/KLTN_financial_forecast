from pathlib import Path
import pickle

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings phải là ma trận 2 chiều")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    print(f"Built FAISS index with {index.ntotal} vectors")
    return index


def save_faiss_store(index: faiss.Index, chunks: list[dict], save_dir: str) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    index_path = save_path / "faiss.index"
    meta_path = save_path / "chunks_meta.pkl"

    faiss.write_index(index, str(index_path))

    with meta_path.open("wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved metadata to: {meta_path}")


def load_faiss_store(save_dir: str):
    save_path = Path(save_dir)
    index_path = save_path / "faiss.index"
    meta_path = save_path / "chunks_meta.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {meta_path}")

    index = faiss.read_index(str(index_path))

    with meta_path.open("rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {index.ntotal} vectors and {len(chunks)} chunks from FAISS")
    return index, chunks


def search_faiss(
    index: faiss.Index,
    chunks: list[dict],
    query_embedding: np.ndarray,
    top_k: int = 5,
    filter_ticker: str | None = None,
    filter_sentiment: str | None = None,
) -> list[dict]:
    if top_k <= 0:
        return []

    if filter_ticker:
        filter_ticker = filter_ticker.upper()
    search_k = min(index.ntotal, max(top_k * 100, 1000)) if (filter_ticker or filter_sentiment) else top_k
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(score)

        if filter_ticker and str(chunk.get("ticker", "")).upper() != filter_ticker:
            continue

        if filter_sentiment and chunk.get("sentiment_label") != filter_sentiment:
            continue

        results.append(chunk)

        if len(results) >= top_k:
            break

    return results
