from pathlib import Path
import numpy as np

from embedder import load_model, load_chunks, embed_documents
from faiss_store import build_faiss_index, save_faiss_store
from chroma_store import build_chroma_store

CHUNKS_PATH = "E:/financial_rag_forecast/data/processed/chunks.jsonl"
VECTOR_ROOT = "../../data/vector_store"
FAISS_DIR = "../../data/vector_store/faiss"
CHROMA_DIR = "../../data/vector_store/chroma"
EMBEDDINGS_PATH = "../../data/vector_store/embeddings.npy"


def main():
    chunks = load_chunks(CHUNKS_PATH)

    valid_chunks = []
    texts = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if text:
            valid_chunks.append(chunk)
            texts.append(text)

    print(f"Số chunks hợp lệ: {len(valid_chunks)}")

    model = load_model()
    embeddings = embed_documents(model, texts, batch_size=16)

    Path(VECTOR_ROOT).mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Đã lưu embeddings tại: {EMBEDDINGS_PATH}")

    print("\n--- BUILD FAISS ---")
    faiss_index = build_faiss_index(embeddings)
    save_faiss_store(faiss_index, valid_chunks, FAISS_DIR)

    print("\n--- BUILD CHROMA ---")
    build_chroma_store(valid_chunks, embeddings, CHROMA_DIR)

    print("\nHOÀN THÀNH: đã build cả FAISS và Chroma")


if __name__ == "__main__":
    main()