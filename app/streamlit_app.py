import sys
from pathlib import Path
import os
import re
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env", override=True)

from src.rag.chatbot import RAGChatbot


st.set_page_config(
    page_title="Chatbot Phân Tích Doanh Nghiệp VN",
    page_icon="📊",
    layout="wide",
)

st.title("Hệ thống dự báo tăng trưởng doanh nghiệp niêm yết Việt Nam")
st.caption("Dự báo tăng trưởng, cảnh báo sớm và hỏi đáp RAG trên dữ liệu tài chính - thị trường - tin tức")

DEFAULT_FORECAST_DIR = ROOT_DIR / "outputs" / "adaptive_strong_1q_all_classifiers_roa_roe"
FORECAST_DIR = Path(os.getenv("FORECAST_OUTPUT_DIR", str(DEFAULT_FORECAST_DIR)))
if not FORECAST_DIR.is_absolute():
    FORECAST_DIR = ROOT_DIR / FORECAST_DIR


@st.cache_resource
def load_chatbot():
    return RAGChatbot(
        backend="faiss",        # đổi "chroma" nếu muốn
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        top_k=5,
        forecast_dir=str(FORECAST_DIR),
    )


@st.cache_data
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def forecast_score_col(df: pd.DataFrame) -> str | None:
    for col in ["predicted_growth_pct", "prob_strong_growth_pct", "prob_1.0", "prob_1", "prediction_confidence"]:
        if col in df.columns:
            return col
    return None


def format_forecast_view(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    if "prob_1.0" in view.columns:
        view["prob_strong_growth_pct"] = view["prob_1.0"] * 100
    if "decision_threshold" in view.columns:
        view["decision_threshold_pct"] = view["decision_threshold"] * 100
    if "prediction_confidence" in view.columns:
        view["prediction_confidence_pct"] = view["prediction_confidence"] * 100
    if "predicted_label" in view.columns:
        view["forecast_signal"] = view["predicted_label"].map(
            {
                1: "Tăng trưởng mạnh",
                1.0: "Tăng trưởng mạnh",
                0: "Chưa tăng trưởng mạnh",
                0.0: "Chưa tăng trưởng mạnh",
            }
        ).fillna("N/A")
    return view


NAV_ITEMS = {
    "forecast": "Dự báo",
    "models": "Đánh giá mô hình",
    "chat": "Chatbot RAG",
}
NAV_LABEL_TO_KEY = {label: key for key, label in NAV_ITEMS.items()}
STATEMENT_FILES = {
    "income_statement": "Báo cáo KQKD",
    "balance_sheet": "Bảng cân đối kế toán",
    "cash_flow": "Lưu chuyển tiền tệ",
    "ratio": "Chỉ số tài chính",
}


def get_query_param(name: str, default: str = "") -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def build_app_link(view: str, **params: object) -> str:
    clean_params = {"view": view}
    clean_params.update({key: value for key, value in params.items() if value not in (None, "")})
    return "?" + urlencode(clean_params)


def normalize_period_param(period: str) -> str:
    match = re.match(r"^(20\d{2})Q([1-4])$", str(period).upper().replace("-", ""))
    if not match:
        return str(period)
    return f"{match.group(1)}Q{match.group(2)}"


def period_to_raw_value(period: str) -> str:
    normalized = normalize_period_param(period)
    match = re.match(r"^(20\d{2})Q([1-4])$", normalized)
    return f"{match.group(1)}-Q{match.group(2)}" if match else normalized


def filter_statement_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    raw_period = period_to_raw_value(period)
    normalized = normalize_period_param(period)
    match = re.match(r"^(20\d{2})Q([1-4])$", normalized)
    if "period" in df.columns:
        filtered = df[df["period"].astype(str).str.upper().eq(raw_period.upper())]
        if not filtered.empty:
            return filtered
    if match and {"yearreport", "lengthreport"}.issubset(df.columns):
        return df[
            (pd.to_numeric(df["yearreport"], errors="coerce") == int(match.group(1)))
            & (pd.to_numeric(df["lengthreport"], errors="coerce") == int(match.group(2)))
        ]
    return pd.DataFrame()


def statement_file_path(symbol: str, statement_key: str) -> Path:
    return ROOT_DIR / "data" / "raw" / "fundamental" / symbol.upper() / f"{statement_key}.csv"


def previous_quarters(period: str, count: int = 2) -> list[str]:
    normalized = normalize_period_param(period)
    match = re.match(r"^(20\d{2})Q([1-4])$", normalized)
    if not match:
        return []
    year = int(match.group(1))
    quarter = int(match.group(2))
    periods = []
    for _ in range(count):
        quarter -= 1
        if quarter == 0:
            year -= 1
            quarter = 4
        periods.append(f"{year}Q{quarter}")
    return periods


def financial_report_pdf_candidates(symbol: str, period: str) -> list[Path]:
    symbol = symbol.upper()
    normalized = normalize_period_param(period)
    match = re.match(r"^(20\d{2})Q([1-4])$", normalized)
    if not match:
        return []
    year = match.group(1)
    short_year = year[-2:]
    quarter = match.group(2)
    tokens = [
        f"{symbol}_{year}Q{quarter}",
        f"{symbol}_{short_year}Q{quarter}",
        f"{symbol}{year}Q{quarter}",
        f"{symbol}{short_year}Q{quarter}",
        f"{symbol}_{year}_Q{quarter}",
        f"{symbol}_{short_year}_Q{quarter}",
    ]
    roots = [
        ROOT_DIR / "data" / "raw" / "financial_reports" / symbol,
        ROOT_DIR / "backups" / "old_data_from_copy3_20260602_044632" / "data" / "raw" / "financial_reports" / symbol,
        ROOT_DIR / "backups" / "old_data_from_copy3_20260602_044632" / "data_parent" / "raw" / "financial_reports" / symbol,
    ]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for pdf_path in root.rglob("*.pdf"):
            compact_name = re.sub(r"[^A-Z0-9]", "", pdf_path.stem.upper())
            if any(re.sub(r"[^A-Z0-9]", "", token.upper()) in compact_name for token in tokens):
                matches.append(pdf_path)
    deduped = []
    seen = set()
    for pdf_path in matches:
        key = str(pdf_path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(pdf_path)
    return deduped


def infer_pdf_period(pdf_path: Path) -> str:
    compact_name = re.sub(r"[^A-Z0-9]", "", pdf_path.stem.upper())
    symbol_match = re.match(r"^[A-Z]{2,4}", compact_name)
    symbol = symbol_match.group(0) if symbol_match else ""
    candidates = re.findall(r"(20\d{2}|[0-9]{2})Q([1-4])", compact_name)
    if not candidates:
        return ""
    year_token, quarter = candidates[-1]
    year = int(year_token) if len(year_token) == 4 else 2000 + int(year_token)
    return f"{year}Q{quarter}"


def all_financial_report_pdfs(symbol: str) -> list[Path]:
    symbol = symbol.upper()
    roots = [
        ROOT_DIR / "data" / "raw" / "financial_reports" / symbol,
        ROOT_DIR / "backups" / "old_data_from_copy3_20260602_044632" / "data" / "raw" / "financial_reports" / symbol,
        ROOT_DIR / "backups" / "old_data_from_copy3_20260602_044632" / "data_parent" / "raw" / "financial_reports" / symbol,
    ]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.pdf"))
    deduped = []
    seen = set()
    for pdf_path in paths:
        key = str(pdf_path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(pdf_path)
    return sorted(deduped, key=lambda path: infer_pdf_period(path) or path.name, reverse=True)


def pdf_markdown_link(symbol: str, pdf_path: Path) -> str:
    period = infer_pdf_period(pdf_path)
    label_period = f" {period}" if period else ""
    return f"[BCTC PDF {symbol.upper()}{label_period}]({pdf_path.resolve().as_uri()})"


def pdf_links_for_symbol_periods(symbol: str, periods: list[str]) -> tuple[list[str], list[str]]:
    links = []
    missing = []
    for period in periods:
        pdf_paths = financial_report_pdf_candidates(symbol, period)
        if not pdf_paths:
            missing.append(period)
            continue
        for pdf_path in pdf_paths:
            links.append(pdf_markdown_link(symbol, pdf_path))
    return links, missing


def render_financial_statement_sources(symbol: str, period: str = "", active_statement: str = "") -> None:
    if not symbol:
        return
    st.subheader("Nguồn BCTC liên quan")
    if period:
        st.caption(f"Mã {symbol.upper()} - kỳ {normalize_period_param(period)}")
    else:
        st.caption(f"Mã {symbol.upper()}")

    for statement_key, label in STATEMENT_FILES.items():
        path = statement_file_path(symbol, statement_key)
        if not path.exists():
            continue
        link = build_app_link(
            "forecast",
            symbol=symbol.upper(),
            period=normalize_period_param(period) if period else "",
            statement=statement_key,
        )
        st.markdown(f"- [{label}]({link}) - `{path.relative_to(ROOT_DIR)}`")
        with st.expander(label, expanded=statement_key == active_statement):
            df = read_csv_if_exists(path)
            display_df = filter_statement_period(df, period) if period else df.tail(4)
            if display_df.empty:
                st.info("Không tìm thấy dòng dữ liệu đúng kỳ trong file này.")
            else:
                st.dataframe(display_df, width="stretch", hide_index=True)


def record_period(record: object, *, target: bool = False) -> str:
    year_attr = "target_year" if target else "year"
    quarter_attr = "target_quarter" if target else "quarter"
    year = getattr(record, year_attr, None)
    quarter = getattr(record, quarter_attr, None)
    if year is None or quarter is None:
        return ""
    return f"{int(year)}Q{int(quarter)}"


def replace_citation_links(answer: str, result: dict) -> str:
    return re.sub(r"\s*\[(Forecast|Graph)\s*\d+\]", "", answer)


def insert_bctc_links_in_evidence(answer: str, result: dict) -> str:
    records = result.get("forecast_records", []) or []
    evidence_lines = []
    for record in records[:3]:
        symbol = getattr(record, "symbol", "")
        input_period = record_period(record)
        target_period = record_period(record, target=True)
        links = statement_links_for_symbol(symbol, input_period)
        if links:
            evidence_lines.append(
                f"- Nguồn BCTC đầu vào của {symbol} {input_period} dùng cho dự báo {target_period}: "
                + " | ".join(links)
            )
    if not evidence_lines:
        return answer

    evidence_text = "\n".join(evidence_lines)
    patterns = [
        r"(\*\*Bằng chứng chính:\*\*\s*)",
        r"(\*\*Bằng chứng hỗ trợ:\*\*\s*)",
    ]
    for pattern in patterns:
        if re.search(pattern, answer):
            return re.sub(pattern, r"\1" + evidence_text + "\n", answer, count=1)
    return answer + "\n\n**Bằng chứng chính:**\n" + evidence_text


def build_statement_links_markdown(result: dict) -> str:
    return ""


def statement_links_for_symbol(symbol: str, period: str) -> list[str]:
    links = []
    for statement_key, label in STATEMENT_FILES.items():
        if statement_file_path(symbol, statement_key).exists():
            links.append(
                f"[{label}]({build_app_link('forecast', symbol=symbol, period=period, statement=statement_key)})"
            )
    return links


def render_reference_sources(result: dict) -> None:
    forecast_records = result.get("forecast_records", []) or []
    news_sources = [
        src
        for src in result.get("sources", []) or []
        if src.get("sentiment") not in {"forecast", "structured"}
    ]

    if forecast_records:
        with st.expander(f"Nguồn BCTC tham khảo ({len(forecast_records)} mã)", expanded=False):
            for index, record in enumerate(forecast_records, start=1):
                symbol = getattr(record, "symbol", "")
                input_period = record_period(record)
                target_period = record_period(record, target=True)
                links = statement_links_for_symbol(symbol, input_period)
                if not links:
                    continue

                col1, col2 = st.columns([2, 5])
                with col1:
                    st.markdown(f"**[{index}] {symbol}**")
                    st.caption(f"BCTC đầu vào {input_period}; dự báo {target_period}")
                with col2:
                    st.markdown(" | ".join(links))

    if news_sources:
        with st.expander(f"Nguồn tham khảo ({len(news_sources)} bài)", expanded=False):
            for index, src in enumerate(news_sources, start=1):
                col1, col2, col3 = st.columns([4, 2, 2])
                with col1:
                    if src["url"]:
                        st.markdown(f"**[{index}]** [{src['title'][:80]}]({src['url']})")
                    else:
                        st.markdown(f"**[{index}]** {src['title'][:80]}")
                with col2:
                    st.caption(src["date"])
                with col3:
                    st.caption(src["sentiment"])


forecast_df = read_csv_if_exists(FORECAST_DIR / "latest_forecast.csv")
results_df = read_csv_if_exists(FORECAST_DIR / "model_results.csv")
predictions_df = read_csv_if_exists(FORECAST_DIR / "predictions.csv")
importance_df = read_csv_if_exists(FORECAST_DIR / "feature_importance.csv")

query_view = get_query_param("view", "forecast")
if query_view not in NAV_ITEMS:
    query_view = "forecast"
if "page_key" not in st.session_state or query_view != st.session_state.get("_last_query_view"):
    st.session_state.page_key = query_view
    st.session_state._last_query_view = query_view

with st.sidebar:
    st.header("Điều hướng")
    for nav_key, nav_label in NAV_ITEMS.items():
        button_type = "primary" if nav_key == st.session_state.page_key else "secondary"
        if st.button(nav_label, key=f"nav_{nav_key}", type=button_type, use_container_width=True):
            st.session_state.page_key = nav_key
            st.session_state._last_query_view = nav_key
            st.query_params["view"] = nav_key
            st.rerun()

page_key = st.session_state.page_key

if page_key == "forecast":
    if forecast_df.empty:
        st.warning(
            f"Chưa có kết quả dự báo tại `{FORECAST_DIR}`. "
            "Có thể đặt biến môi trường `FORECAST_OUTPUT_DIR` để trỏ sang thư mục output khác."
        )
    else:
        forecast_view = format_forecast_view(forecast_df)
        symbols = sorted(forecast_view["symbol"].dropna().unique().tolist())
        query_symbol = get_query_param("symbol", "").upper()
        query_period = normalize_period_param(get_query_param("period", ""))
        active_statement = get_query_param("statement", "")
        default_symbols = [query_symbol] if query_symbol in symbols else symbols[:8]
        selected_symbols = st.multiselect("Mã cổ phiếu", symbols, default=default_symbols)
        only_positive = st.toggle("Chỉ hiển thị dự báo tăng trưởng mạnh", value=False)

        view_df = forecast_view.copy()
        if selected_symbols:
            view_df = view_df[view_df["symbol"].isin(selected_symbols)]
        if only_positive and "predicted_label" in view_df.columns:
            view_df = view_df[view_df["predicted_label"] == 1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số doanh nghiệp", int(forecast_view["symbol"].nunique()))
        if "prob_1.0" in forecast_view.columns:
            c2.metric("Xác suất tăng trưởng mạnh TB", f"{forecast_view['prob_1.0'].mean() * 100:.2f}%")
        else:
            c2.metric("Số bản ghi dự báo", int(len(forecast_view)))
        if "predicted_label" in forecast_view.columns:
            c3.metric("Dự báo tăng trưởng mạnh", int((forecast_view["predicted_label"] == 1).sum()))
        else:
            c3.metric("Dự báo tăng trưởng mạnh", "N/A")
        c4.metric("Kỳ dữ liệu mới nhất", str(int(forecast_view["yq_index"].max())))

        score_col = forecast_score_col(view_df)
        if score_col:
            chart_df = view_df[["symbol", score_col]].sort_values(score_col)
            st.bar_chart(chart_df.set_index("symbol"))

        display_cols = [
            "symbol",
            "year",
            "quarter",
            "forecast_signal",
            "prob_strong_growth_pct",
            "decision_threshold_pct",
            "prediction_confidence_pct",
            "predicted_label",
        ]
        existing_cols = [col for col in display_cols if col in view_df.columns]
        st.dataframe(
            view_df[existing_cols].sort_values(score_col or "symbol", ascending=False),
            width="stretch",
            hide_index=True,
        )

        if query_symbol:
            render_financial_statement_sources(query_symbol, query_period, active_statement)

elif page_key == "models":
    if results_df.empty:
        st.warning("Chưa có file đánh giá mô hình trong `outputs/growth_forecast`.")
    else:
        st.subheader("So sánh mô hình")
        st.dataframe(results_df, width="stretch", hide_index=True)

        if not predictions_df.empty:
            split = st.selectbox("Tập dữ liệu", sorted(predictions_df["split"].unique().tolist()))
            split_df = predictions_df[predictions_df["split"] == split].copy()
            if {"actual_growth_pct", "predicted_growth_pct"}.issubset(split_df.columns):
                st.scatter_chart(
                    split_df,
                    x="actual_growth_pct",
                    y="predicted_growth_pct",
                    color="symbol",
                )
            elif {"actual_label", "prob_1.0"}.issubset(split_df.columns):
                st.scatter_chart(
                    split_df,
                    x="actual_label",
                    y="prob_1.0",
                    color="symbol",
                )
            else:
                st.dataframe(split_df.head(500), width="stretch", hide_index=True)

        if not importance_df.empty:
            st.subheader("Feature importance")
            top_n = st.slider("Số biến hiển thị", min_value=5, max_value=30, value=15)
            top_df = importance_df.head(top_n).sort_values("importance")
            st.bar_chart(top_df.set_index("feature"))

elif page_key == "chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Hỏi về doanh nghiệp, xu hướng, lợi nhuận, tăng trưởng..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích..."):
                bot = load_chatbot()
                result = bot.ask(query)

            display_answer = replace_citation_links(result["answer"], result)
            display_answer = insert_bctc_links_in_evidence(display_answer, result)
            display_answer += build_statement_links_markdown(result)
            st.markdown(display_answer)

            render_reference_sources(result)

            st.session_state.messages.append(
                {"role": "assistant", "content": display_answer}
            )
