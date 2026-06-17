from pathlib import Path
import os
import pickle
import re
import unicodedata

try:
    from .embedder import load_model, embed_query
    from .faiss_store import load_faiss_store, search_faiss
    from .chroma_store import load_chroma_collection, search_chroma
except ImportError:
    from embedder import load_model, embed_query
    from faiss_store import load_faiss_store, search_faiss
    from chroma_store import load_chroma_collection, search_chroma


ROOT_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = ROOT_DIR / "data" / "vector_store"


class RAGRetriever:
    def __init__(self, backend: str = "faiss"):
        self.backend = backend.lower()
        self.model = None
        print(f"Initializing retriever with backend = {self.backend}")

        if self.backend == "faiss":
            try:
                self.index, self.chunks = load_faiss_store(VECTOR_STORE_DIR / "faiss")
            except (OSError, MemoryError, RuntimeError) as exc:
                self.index = None
                self.chunks = self._load_faiss_chunks_meta(VECTOR_STORE_DIR / "faiss")
                print(
                    "Cannot read FAISS index; falling back to lightweight keyword search. "
                    f"Reason: {exc}"
                )
            self.collection = None

        elif self.backend == "chroma":
            self.collection = load_chroma_collection(VECTOR_STORE_DIR / "chroma")
            self.index = None
            self.chunks = None

        else:
            raise ValueError("backend phải là 'faiss' hoặc 'chroma'")

        if self.index is None:
            print("No usable FAISS index; skipping embedding model.")
        elif os.getenv("RAG_DISABLE_EMBEDDING", "").strip() == "1":
            print("RAG_DISABLE_EMBEDDING=1; using lightweight keyword search.")
        else:
            try:
                self.model = load_model()
            except (OSError, MemoryError, RuntimeError) as exc:
                if self.backend != "faiss":
                    raise
                print(
                    "Cannot load embedding model; falling back to lightweight keyword search. "
                    f"Reason: {exc}"
                )

        print("Retriever ready.")

    @property
    def symbols(self) -> set[str]:
        if self.backend == "faiss" and self.chunks:
            return {
                str(chunk.get("ticker")).upper()
                for chunk in self.chunks
                if chunk.get("ticker")
            }
        return set()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
        sentiment: str | None = None,
    ) -> list[dict]:
        if self.model is None or self.index is None:
            return self._search_keywords(
                query=query,
                top_k=top_k,
                filter_ticker=ticker,
                filter_sentiment=sentiment,
            )

        query_embedding = embed_query(self.model, query)

        if self.backend == "faiss":
            return search_faiss(
                index=self.index,
                chunks=self.chunks,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_ticker=ticker,
                filter_sentiment=sentiment,
            )

        return search_chroma(
            collection=self.collection,
            query_embedding=query_embedding,
            top_k=top_k,
            filter_ticker=ticker,
            filter_sentiment=sentiment,
        )

    @staticmethod
    def _load_faiss_chunks_meta(save_dir: Path) -> list[dict]:
        meta_path = save_dir / "chunks_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing file: {meta_path}")
        with meta_path.open("rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks metadata from {meta_path}")
        return chunks

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = value.replace("đ", "d").replace("Đ", "D")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _search_keywords(
        self,
        query: str,
        top_k: int = 5,
        filter_ticker: str | None = None,
        filter_sentiment: str | None = None,
    ) -> list[dict]:
        if not self.chunks or top_k <= 0:
            return []

        if filter_ticker:
            filter_ticker = filter_ticker.upper()

        query_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", self._normalize_text(query))
            if len(term) >= 2
        }
        if not query_terms and filter_ticker:
            query_terms = {filter_ticker.lower()}

        scored = []
        for chunk in self.chunks:
            if filter_ticker and str(chunk.get("ticker", "")).upper() != filter_ticker:
                continue
            if filter_sentiment and chunk.get("sentiment_label") != filter_sentiment:
                continue

            text = self._normalize_text(
                " ".join(
                    [
                        str(chunk.get("ticker", "")),
                        str(chunk.get("company_name", "")),
                        str(chunk.get("title", "")),
                        str(chunk.get("text", "")),
                    ]
                )
            )
            score = sum(text.count(term) for term in query_terms)
            if score <= 0:
                continue

            item = chunk.copy()
            item["similarity_score"] = float(score)
            scored.append(item)

        scored.sort(key=lambda item: item["similarity_score"], reverse=True)
        return scored[:top_k]

    def format_context(self, results: list[dict]) -> str:
        blocks = []

        for i, r in enumerate(results, start=1):
            block = (
                f"[Nguồn {i}]\n"
                f"Tiêu đề: {r.get('title', 'N/A')}\n"
                f"Ngày: {r.get('date', 'N/A')}\n"
                f"Công ty: {r.get('company_name', 'N/A')}\n"
                f"Mã: {r.get('ticker', 'N/A')}\n"
                f"Sentiment: {r.get('sentiment_label', 'N/A')} "
                f"({r.get('sentiment_score', 'N/A')})\n"
                f"URL: {r.get('source_url', 'N/A')}\n"
                f"Nội dung: {r.get('text', '')}\n"
                f"Similarity: {round(r.get('similarity_score', 0.0), 4)}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    query = "FPT có kế hoạch tăng trưởng doanh thu không?"

    print("\n===== TEST FAISS =====")
    retriever_faiss = RAGRetriever(backend="faiss")
    results_faiss = retriever_faiss.retrieve(query=query, top_k=3, ticker="FPT")
    for r in results_faiss:
        print(f"[FAISS] {r.get('ticker')} | {r.get('similarity_score'):.4f} | {r.get('title')}")

    print("\n===== TEST CHROMA =====")
    retriever_chroma = RAGRetriever(backend="chroma")
    results_chroma = retriever_chroma.retrieve(query=query, top_k=3, ticker="FPT")
    for r in results_chroma:
        print(f"[CHROMA] {r.get('ticker')} | {r.get('similarity_score'):.4f} | {r.get('title')}")
