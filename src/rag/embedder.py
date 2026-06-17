import json
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_CACHE = {}


def load_model(model_name: str | None = None):
    model_name = model_name or os.getenv("RAG_EMBEDDING_MODEL", MODEL_NAME)
    cache_key = model_name
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    print(f"Loading model: {model_name}")

    from sentence_transformers import SentenceTransformer

    try:
        # Avoid safetensors memory mapping on Windows when the pagefile is small.
        model = SentenceTransformer(
            model_name,
            device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
            model_kwargs={"use_safetensors": False},
        )
    except TypeError:
        model = SentenceTransformer(
            model_name,
            device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
        )

    _MODEL_CACHE[cache_key] = model
    print("Model loaded.")
    return model


def load_chunks(jsonl_path: str) -> list[dict]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {jsonl_path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Đã đọc {len(chunks)} chunks từ {jsonl_path}")
    return chunks


def embed_documents(
    model,
    texts: list[str],
    batch_size: int = 16
) -> np.ndarray:
    print(f"Đang tạo embeddings cho {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Xong embeddings documents. Shape = {embeddings.shape}")
    return embeddings.astype(np.float32)


def embed_query(
    model,
    query: str
) -> np.ndarray:
    query_text = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)
