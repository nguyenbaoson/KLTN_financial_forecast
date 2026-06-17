import re
import unicodedata
import csv
from pathlib import Path

try:
    from .retriever import RAGRetriever
    from .llm_client import LLMClient
    from .prompts import build_rag_prompt, build_summary_prompt
    from .forecast_retriever import ForecastRetriever
except ImportError:
    from retriever import RAGRetriever
    from llm_client import LLMClient
    from prompts import build_rag_prompt, build_summary_prompt
    from forecast_retriever import ForecastRetriever

try:
    from .financial_graph_retriever import FinancialGraphRetriever
except Exception:
    try:
        from financial_graph_retriever import FinancialGraphRetriever
    except Exception:
        FinancialGraphRetriever = None


ROOT_DIR = Path(__file__).resolve().parents[2]
SECTOR_REFERENCE_PATH = ROOT_DIR / "data" / "raw" / "reference" / "equity" / "list_by_industry.csv"


class RAGChatbot:
    def __init__(
        self,
        backend: str = "faiss",
        llm_provider: str = "gemini",
        top_k: int = 5,
        enable_graph: bool = True,
        enable_forecast: bool = True,
        forecast_dir: str | None = None,
    ):
        print("Starting RAG Chatbot...")
        self.retriever = RAGRetriever(backend=backend)
        self.graph_retriever = FinancialGraphRetriever() if enable_graph and FinancialGraphRetriever else None
        self.forecast_retriever = ForecastRetriever(forecast_dir) if enable_forecast else None
        self.llm = LLMClient(provider=llm_provider)
        self.top_k = top_k
        self.history = []
        self.known_symbols = self._load_known_symbols()
        print("Chatbot ready.")

    def _normalize_text(self, value: str) -> str:
        text = value.replace("đ", "d").replace("Đ", "D")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _needs_forecast_context(self, query: str) -> bool:
        text = self._normalize_text(query)
        forecast_keywords = [
            "du bao",
            "forecast",
            "mo hinh",
            "model",
            "xac suat",
            "probability",
            "kha nang",
            "trien vong",
            "tuong lai",
            "ky toi",
            "quy toi",
            "nam toi",
            "tang truong manh",
            "nhom tang truong",
            "tin hieu",
            "signal",
        ]
        return any(keyword in text for keyword in forecast_keywords)

    def _needs_news_context(self, query: str) -> bool:
        text = self._normalize_text(query)
        news_keywords = [
            "tin tuc",
            "tin moi",
            "bai bao",
            "nguon tin",
            "sentiment",
            "cam xuc",
            "truyen thong",
            "su kien",
        ]
        return any(keyword in text for keyword in news_keywords)

    def _load_known_symbols(self) -> set[str]:
        symbols = set(self.retriever.symbols)
        if self.graph_retriever is not None:
            symbols.update(self.graph_retriever.detect_symbols("", None))
            _ = self.graph_retriever.dataset
            symbols.update(getattr(self.graph_retriever, "_symbols", set()))
        if self.forecast_retriever is not None:
            symbols.update(self.forecast_retriever.symbols)
        return {symbol.upper() for symbol in symbols if symbol}

    def detect_ticker(self, query: str) -> str | None:
        text = query.upper()
        for symbol in sorted(self.known_symbols, key=len, reverse=True):
            if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", text):
                return symbol

        ticker_keywords = {
            "FPT": ["FPT", "CTCP FPT", "Tập đoàn FPT"],
            "VNM": ["VNM", "Vinamilk", "CTCP Sữa Việt Nam", "Sữa Việt Nam"],
            "VCB": ["VCB", "Vietcombank", "Ngân hàng TMCP Ngoại thương Việt Nam"],
            "HPG": ["HPG", "Hòa Phát", "Tập đoàn Hòa Phát"],
            "MWG": ["MWG", "Thế Giới Di Động", "CTCP Đầu tư Thế Giới Di Động"],
            "SSI": ["SSI", "Chứng khoán SSI"],
            "MBB": ["MBB", "MB", "Ngân hàng Quân đội"],
            "BID": ["BID", "BIDV", "Ngân hàng Đầu tư và Phát triển Việt Nam"],
            "TCB": ["TCB", "Techcombank"],
            "ACB": ["ACB", "Ngân hàng Á Châu"],
            "PNJ": ["PNJ", "Vàng bạc Đá quý Phú Nhuận"],
            "REE": ["REE", "Cơ điện lạnh"],
            "GAS": ["GAS", "PV Gas", "Tổng Công ty Khí Việt Nam"],
            "DIG": ["DIG", "DIC Corp", "DIC"],
            "KDH": ["KDH", "Khang Điền"],
            "BCM": ["BCM", "Becamex"],
            "VIC": ["VIC", "Vingroup"],
            "VHM": ["VHM", "Vinhomes"],
            "VRE": ["VRE", "Vincom Retail"],
            "NVL": ["NVL", "Novaland"],
            "KBC": ["KBC", "Kinh Bắc"],
            "CII": ["CII"],
            "DGW": ["DGW", "Digiworld", "Thế Giới Số"],
            "TDM": ["TDM", "Nước Thủ Dầu Một"],
            "SGR": ["SGR", "Saigonres"],
            "TW3": ["TW3"],
            "LIG": ["LIG", "Licogi 13"],
            "HCM": ["HCM", "Chứng khoán TP.HCM", "HSC"],
            "STB": ["STB", "Sacombank"],
            "SHB": ["SHB", "Ngân hàng Sài Gòn - Hà Nội"],
        }
        q = query.lower()
        for ticker, keywords in ticker_keywords.items():
            if any(kw.lower() in q for kw in keywords):
                return ticker
        return None

    def ask(self, query: str, ticker: str | None = None) -> dict:
        if not ticker:
            ticker = self.detect_ticker(query)

        needs_forecast = self._needs_forecast_context(query)
        needs_news = self._needs_news_context(query)
        retrieved = (
            self.retriever.retrieve(
                query=query,
                top_k=self.top_k,
                ticker=ticker,
            )
            if needs_news or not needs_forecast
            else []
        )
        graph_records = (
            self.graph_retriever.retrieve(query=query, ticker=ticker, top_k=4)
            if self.graph_retriever is not None
            else []
        )
        forecast_query = query
        sector_symbols = []
        if not ticker and self.forecast_retriever is not None and needs_forecast:
            sector_symbols = self._detect_sector_symbols(query)
            if sector_symbols:
                forecast_query = f"{query} {' '.join(sector_symbols)}"

        forecast_records = (
            self.forecast_retriever.retrieve(query=forecast_query, ticker=ticker, top_k=1)
            if self.forecast_retriever is not None and needs_forecast
            else []
        )
        if forecast_records and sector_symbols:
            latest_yq = max(record.yq_index for record in forecast_records)
            forecast_records = [record for record in forecast_records if record.yq_index == latest_yq]

        if not forecast_records and not retrieved:
            retrieved = self.retriever.retrieve(
                query=query,
                top_k=self.top_k,
                ticker=ticker,
            )

        if not retrieved and not graph_records and not forecast_records:
            return {
                "answer": (
                    "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu. "
                    "Bạn hãy thử hỏi cụ thể hơn hoặc kiểm tra lại dữ liệu đã crawl."
                ),
                "sources": [],
                "chunks": [],
                "ticker": ticker,
            }

        context_blocks = []
        if forecast_records and self.forecast_retriever is not None:
            context_blocks.append(self.forecast_retriever.format_context(forecast_records))
        if graph_records and self.graph_retriever is not None:
            context_blocks.append(self.graph_retriever.format_context(graph_records))
        if retrieved:
            context_blocks.append(self.retriever.format_context(retrieved))
        context = "\n\n===\n\n".join(context_blocks)
        company_name = retrieved[0].get("company_name") if retrieved else ticker
        prompt = build_rag_prompt(query, context, company=company_name)

        answer = self.llm.generate(prompt)
        if self._is_llm_error(answer):
            answer = self._fallback_answer(
                query=query,
                ticker=ticker,
                forecast_records=forecast_records,
                graph_records=graph_records,
                retrieved=retrieved,
            )

        sources = []
        for r in forecast_records:
            if r.target_year is not None and r.target_quarter is not None:
                title = (
                    f"Latest model forecast: {r.symbol} input {r.year}Q{r.quarter} "
                    f"-> target {r.target_year}Q{r.target_quarter}"
                )
                date = f"input {r.year}Q{r.quarter}; target {r.target_year}Q{r.target_quarter}"
            else:
                title = f"Latest model forecast: {r.symbol} input {r.year}Q{r.quarter}"
                date = f"input {r.year}Q{r.quarter}"
            sources.append(
                {
                    "title": title,
                    "url": "",
                    "date": date,
                    "company": r.symbol,
                    "ticker": r.symbol,
                    "sentiment": "forecast",
                    "score": r.prob_strong_growth if r.prob_strong_growth is not None else 1.0,
                }
            )
        for r in graph_records:
            sources.append(
                {
                    "title": f"Structured financial graph: {r.symbol} {r.year}Q{r.quarter}",
                    "url": "",
                    "date": f"{r.year}Q{r.quarter}",
                    "company": r.symbol,
                    "ticker": r.symbol,
                    "sentiment": "structured",
                    "score": 1.0,
                }
            )
        for r in retrieved:
            sources.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("source_url", ""),
                    "date": r.get("date", ""),
                    "company": r.get("company_name", ""),
                    "ticker": r.get("ticker", ""),
                    "sentiment": r.get("sentiment_label", ""),
                    "score": round(r.get("similarity_score", 0.0), 3),
                }
            )

        self.history.append(
            {
                "query": query,
                "answer": answer,
                "ticker": ticker,
                "n_sources": len(sources),
            }
        )

        return {
            "answer": answer,
            "sources": sources,
            "chunks": retrieved,
            "ticker": ticker,
            "forecast_records": forecast_records,
            "graph_records": graph_records,
        }

    def summarize(self, ticker: str, top_k: int = 8) -> str:
        chunks = self.retriever.retrieve(
            query=f"tin tức mới nhất về {ticker}",
            top_k=top_k,
            ticker=ticker,
        )

        if not chunks:
            return f"Không có dữ liệu về {ticker}"

        company = chunks[0].get("company_name", ticker)
        prompt = build_summary_prompt(company, chunks)
        answer = self.llm.generate(prompt)
        if self._is_llm_error(answer):
            titles = [chunk.get("title", "N/A") for chunk in chunks[:5]]
            return (
                "Dang dung che do tom tat rut gon vi API sinh ngon ngu chua kha dung.\n\n"
                + "\n".join(f"- {title}" for title in titles)
            )
        return answer

    @staticmethod
    def _is_llm_error(answer: str) -> bool:
        return answer == "LLM_API_QUOTA_EXCEEDED" or answer.startswith("Lỗi khi gọi LLM:")

    @staticmethod
    def _format_pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    def _fallback_answer(
        self,
        query: str,
        ticker: str | None,
        forecast_records: list,
        graph_records: list,
        retrieved: list[dict],
    ) -> str:
        evidence = []

        if forecast_records:
            strong_records = [record for record in forecast_records if record.predicted_label == 1]
            target_periods = sorted(
                {
                    f"{record.target_year}Q{record.target_quarter}"
                    for record in forecast_records
                    if record.target_year is not None and record.target_quarter is not None
                }
            )
            target_text = ", ".join(target_periods) if target_periods else "kỳ mục tiêu hiện có"
            if len(forecast_records) > 1:
                avg_prob_values = [
                    record.prob_strong_growth
                    for record in forecast_records
                    if record.prob_strong_growth is not None
                ]
                avg_prob = sum(avg_prob_values) / len(avg_prob_values) if avg_prob_values else None
                analysis = (
                    f"Dựa trên kết quả dự báo mới nhất, có {len(strong_records)}/{len(forecast_records)} "
                    f"mã trong nhóm được mô hình phân loại vào nhóm tăng trưởng lợi nhuận mạnh cho {target_text}. "
                    f"Xác suất tăng trưởng mạnh trung bình của nhóm là {self._format_pct(avg_prob)}."
                )
                top_records = sorted(
                    forecast_records,
                    key=lambda item: item.prob_strong_growth if item.prob_strong_growth is not None else -1,
                    reverse=True,
                )[:4]
                for index, record in enumerate(top_records, start=1):
                    label = (
                        "thuộc nhóm tăng trưởng mạnh"
                        if record.predicted_label == 1
                        else "chưa thuộc nhóm tăng trưởng mạnh"
                        if record.predicted_label == 0
                        else "chưa có nhãn dự báo"
                    )
                    evidence.append(
                        f"{record.symbol} được dự báo {label}, xác suất tăng trưởng mạnh "
                        f"{self._format_pct(record.prob_strong_growth)} so với ngưỡng "
                        f"{self._format_pct(record.decision_threshold)} [Forecast {index}]."
                    )
                conclusion = (
                    "Tích cực về khả năng xuất hiện một số mã ngân hàng thuộc nhóm tăng trưởng mạnh."
                    if strong_records
                    else "Trung tính/tiêu cực về khả năng thuộc nhóm tăng trưởng mạnh."
                )
            else:
                record = forecast_records[0]
                label = (
                    "thuộc nhóm tăng trưởng lợi nhuận mạnh"
                    if record.predicted_label == 1
                    else "chưa thuộc nhóm tăng trưởng lợi nhuận mạnh"
                    if record.predicted_label == 0
                    else "chưa có nhãn dự báo"
                )
                target_period = (
                    f"{record.target_year}Q{record.target_quarter}"
                    if record.target_year is not None and record.target_quarter is not None
                    else "kỳ mục tiêu hiện có"
                )
                analysis = (
                    f"Dựa trên kết quả dự báo mới nhất, mô hình dự báo {record.symbol} {label} "
                    f"cho {target_period}. Xác suất tăng trưởng mạnh là "
                    f"{self._format_pct(record.prob_strong_growth)}, so với ngưỡng quyết định "
                    f"{self._format_pct(record.decision_threshold)}."
                )
                evidence.append(
                    f"Mô hình sử dụng dữ liệu đầu vào đến {record.year}Q{record.quarter} "
                    f"để dự báo cho {target_period} [Forecast 1]."
                )
                evidence.append(
                    f"Nhãn dự báo của {record.symbol} là \"{label}\" [Forecast 1]."
                )
                evidence.append(
                    f"Xác suất tăng trưởng mạnh là {self._format_pct(record.prob_strong_growth)}, "
                    f"ngưỡng quyết định là {self._format_pct(record.decision_threshold)} [Forecast 1]."
                )
                conclusion = (
                    "Tích cực về khả năng thuộc nhóm tăng trưởng mạnh."
                    if record.predicted_label == 1
                    else "Tiêu cực về khả năng thuộc nhóm tăng trưởng mạnh."
                    if record.predicted_label == 0
                    else "Trung tính do thiếu nhãn dự báo."
                )
        else:
            analysis = (
                "Chưa đủ dữ liệu dự báo từ mô hình để kết luận trực tiếp về nhóm tăng trưởng mạnh. "
                "Các thông tin bên dưới chỉ phản ánh nguồn tin và dữ liệu đã truy xuất được trong hệ thống."
            )
            conclusion = "Trung tính do thiếu kết quả dự báo trực tiếp từ mô hình."

        if retrieved:
            for index, item in enumerate(retrieved[:3], start=1):
                title = item.get("title", "N/A")
                date = item.get("date", "N/A")
                symbol = item.get("ticker") or item.get("company_name") or "N/A"
                sentiment = item.get("sentiment_label", "N/A")
                evidence.append(
                    f"{symbol} có nguồn tin ngày {date} với sentiment {sentiment}: "
                    f"{title} [Nguồn {index}]."
                )

        if not evidence:
            evidence.append("Chưa tìm thấy nguồn nội bộ phù hợp với câu hỏi.")

        evidence_text = "\n".join(f"- {item}" for item in evidence[:5])
        return (
            f"**Phân tích:** {analysis}\n\n"
            f"**Bằng chứng chính:**\n{evidence_text}\n\n"
            f"**Kết luận:** {conclusion}"
        )

    def _detect_sector_symbols(self, query: str) -> list[str]:
        text = self._normalize_text(query)
        sector_aliases = {
            "bank": ["nganh ngan hang", "ngan hang", "bank", "banks"],
        }
        sector_name = None
        for name, aliases in sector_aliases.items():
            if any(alias in text for alias in aliases):
                sector_name = name
                break
        if sector_name is None or not SECTOR_REFERENCE_PATH.exists():
            return []

        symbols = set()
        try:
            with SECTOR_REFERENCE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    symbol = str(row.get("symbol", "")).upper().strip()
                    industry = self._normalize_text(row.get("icb_name", ""))
                    if sector_name == "bank" and "ngan hang" in industry and symbol:
                        symbols.add(symbol)
        except Exception:
            return []

        forecast_symbols = self.forecast_retriever.symbols if self.forecast_retriever is not None else set()
        return sorted(symbol for symbol in symbols if symbol in forecast_symbols)

    def print_response(self, result: dict):
        print("\n" + "=" * 70)
        print("TRẢ LỜI:")
        print(result["answer"])

        print("\nNGUỒN THAM KHẢO:")
        if not result["sources"]:
            print("  Không có nguồn.")
        else:
            for i, s in enumerate(result["sources"], start=1):
                print(f"  [{i}] {s['title'][:80]}")
                print(
                    f"      {s['date']} | {s['company']} | "
                    f"{s['ticker']} | sentiment: {s['sentiment']} | score: {s['score']}"
                )
                if s["url"]:
                    print(f"      {s['url']}")
        print("=" * 70)
