from pathlib import Path

import chromadb


COLLECTION_NAME = "news_chunks"


def _safe_str(value):
    if value is None:
        return ""
    return str(value)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def build_chroma_store(chunks: list[dict], embeddings, save_dir: str):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(save_path))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 500
    total = len(chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {
                    "title": _safe_str(c.get("title")),
                    "ticker": _safe_str(c.get("ticker")),
                    "company_name": _safe_str(c.get("company_name")),
                    "date": _safe_str(c.get("date")),
                    "sentiment_label": _safe_str(c.get("sentiment_label")),
                    "sentiment_score": _safe_float(c.get("sentiment_score")),
                    "source": _safe_str(c.get("source")),
                    "source_url": _safe_str(c.get("source_url")),
                }
                for c in batch_chunks
            ]
        )
        print(f"Đã thêm {end}/{total} chunks vào Chroma")

    print(f"Đã build Chroma tại: {save_dir}")
    return collection


def load_chroma_collection(save_dir: str):
    client = chromadb.PersistentClient(path=str(save_dir))
    collection = client.get_collection(COLLECTION_NAME)
    print("Đã load Chroma collection")
    return collection


def search_chroma(
    collection,
    query_embedding,
    top_k: int = 5,
    filter_ticker: str | None = None,
    filter_sentiment: str | None = None,
) -> list[dict]:
    where_filter = {}

    if filter_ticker:
        where_filter["ticker"] = filter_ticker

    if filter_sentiment:
        where_filter["sentiment_label"] = filter_sentiment

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        item = {
            "chunk_id": ids[i],
            "text": docs[i],
            "similarity_score": float(1 - distances[i]),
            **metas[i]
        }
        output.append(item)

    return output