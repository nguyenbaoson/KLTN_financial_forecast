from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCX = ROOT / "Promt" / "KLTN_22022613_Nguyễn Bảo Sơn.docx"
OUT_DIR = ROOT / "latex_kltn_nbs"
FIG_DIR = OUT_DIR / "figures"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


@dataclass
class ParagraphBlock:
    text: str
    images: list[str]
    is_list: bool = False


@dataclass
class TableBlock:
    rows: list[list[str]]


Block = ParagraphBlock | TableBlock


TEXT_REPLACEMENTS = [
    ("nguyên cứu", "nghiên cứu"),
    ("giá trình", "quá trình"),
    ("đựa trên", "dựa trên"),
    ("tang trưởng", "tăng trưởng"),
    ("kế quả", "kết quả"),
    ("tịa thời điểm", "tại thời điểm"),
    ("cô rphieeus", "cổ phiếu"),
    ("tài chín", "tài chính"),
    ("Thư viên", "Thư viện"),
    ("classification threshold", "ngưỡng phân loại"),
    ("Classification threshold", "Ngưỡng phân loại"),
    ("Confusion Matrix", "ma trận nhầm lẫn"),
    ("confusion matrix", "ma trận nhầm lẫn"),
    ("actual vs predicted", "thực tế và dự báo"),
    ("target adaptive", "biến mục tiêu thích nghi theo ngành"),
    ("Target adaptive", "Biến mục tiêu thích nghi theo ngành"),
    ("baseline", "mô hình cơ sở"),
    ("tập train", "tập huấn luyện"),
    ("tập validation", "tập xác thực"),
    ("tập test", "tập kiểm thử"),
    ("file kết quả", "tệp kết quả"),
    ("pipeline", "quy trình xử lý"),
    ("hyperparameter", "siêu tham số"),
    ("feature importance", "mức độ quan trọng của đặc trưng"),
    ("feature filtering", "lọc đặc trưng"),
    ("regularization term", "thành phần điều chuẩn"),
    ("regularization", "điều chuẩn"),
    ("overfitting", "quá khớp"),
    ("weak learner", "bộ học yếu"),
    ("learning rate", "tốc độ học"),
    ("majority voting", "bỏ phiếu đa số"),
    ("node split", "nút phân tách"),
]


FORMULA_MAP = {
    "Profit Growth=Profitt-Profitt-1Profitt-1×100%": r"\[\mathrm{Profit\ Growth} = \frac{\mathrm{Profit}_t-\mathrm{Profit}_{t-1}}{\mathrm{Profit}_{t-1}}\times 100\%\]",
    "y=1neu doanh nghiệp tang trưởng lợi nhuận mạnh0ngược lại": r"\[y=\begin{cases}1, & \text{nếu doanh nghiệp tăng trưởng lợi nhuận mạnh}\\0, & \text{ngược lại}\end{cases}\]",
    "Xt=[x1,x2,x3,...,xn]": r"\[X_t=[x_1,x_2,x_3,\ldots,x_n]\]",
    "ROA= Lợi nhuận sau thuếTổng tài sản bình quân ×100%": r"\[\mathrm{ROA}=\frac{\text{Lợi nhuận sau thuế}}{\text{Tổng tài sản bình quân}}\times100\%\]",
    "ROE= Lợi nhuận sau thuếVốn chủ sở hữu bình quân × 100%": r"\[\mathrm{ROE}=\frac{\text{Lợi nhuận sau thuế}}{\text{Vốn chủ sở hữu bình quân}}\times100\%\]",
    "Net Profit Margin= Lợi nhuận sau thuếDoanh thu thuần × 100%": r"\[\mathrm{Net\ Profit\ Margin}=\frac{\text{Lợi nhuận sau thuế}}{\text{Doanh thu thuần}}\times100\%\]",
    "Returnt= Pt-Pt-1Pt-1 × 100%": r"\[\mathrm{Return}_t=\frac{P_t-P_{t-1}}{P_{t-1}}\times100\%\]",
    "Volatility= 1n-1i=1n(ri-r)2": r"\[\mathrm{Volatility}=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(r_i-\bar r)^2}\]",
    "Revenue Growth= Revenuet-Revenuet-1Revenuet-1 ×100%": r"\[\mathrm{Revenue\ Growth}=\frac{\mathrm{Revenue}_t-\mathrm{Revenue}_{t-1}}{\mathrm{Revenue}_{t-1}}\times100\%\]",
    "Rolling Meant= 1k i=t-k+1tProfiti": r"\[\mathrm{Rolling\ Mean}_t=\frac{1}{k}\sum_{i=t-k+1}^{t}\mathrm{Profit}_i\]",
    "yt=c+i=1pϕiyt-i+j=1qθjεt-j+εt": r"\[y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}+\varepsilon_t\]",
    "ϕi: hệ số tự hồi quy": r"$\phi_i$: hệ số tự hồi quy",
    "θj: hệ số trung bình trượt": r"$\theta_j$: hệ số trung bình trượt",
    "εt: nhiễu ngẫu nhiên": r"$\varepsilon_t$: nhiễu ngẫu nhiên",
    "y=mode⁡(h1(x),h2(x),...,hT(x))": r"\[\hat y=\operatorname{mode}(h_1(x),h_2(x),\ldots,h_T(x))\]",
    "yit=yit1+ft(xi)": r"\[\hat y_i^{(t)}=\hat y_i^{(t-1)}+f_t(x_i)\]",
    "Obj=i=1nl(yi,yi)+k=1KΩ(fk)": r"\[\mathrm{Obj}=\sum_{i=1}^{n}l(y_i,\hat y_i)+\sum_{k=1}^{K}\Omega(f_k)\]",
    "Ω(fk): thành phần điều chuẩn": r"$\Omega(f_k)$: thành phần điều chuẩn",
    "Fm(x)=Fm-1(x)+γmhm(x)": r"\[F_m(x)=F_{m-1}(x)+\gamma_m h_m(x)\]",
    "γm: tốc độ học": r"$\gamma_m$: tốc độ học",
    "P(y=1∣x)=i=1Mwipi(x)i=1Mwi": r"\[P(y=1\mid x)=\frac{\sum_{i=1}^{M}w_i p_i(x)}{\sum_{i=1}^{M}w_i}\]",
    "y=1P(y=1∣x)≥threshold0otherwise": r"\[\hat y=\begin{cases}1, & P(y=1\mid x)\ge\gamma\\0, & \text{otherwise}\end{cases}\]",
    "Accuracy=TP+TNTP+TN+FP+FN": r"\[\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}\]",
    "Precision=TPTP+FP": r"\[\mathrm{Precision}=\frac{TP}{TP+FP}\]",
    "Recall=TPTP+FN": r"\[\mathrm{Recall}=\frac{TP}{TP+FN}\]",
    "F1=2×Precision×RecallPrecision+Recall": r"\[F1=2\times\frac{\mathrm{Precision}\times\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}\]",
    "Balanced Accuracy=TPR+TNR2": r"\[\mathrm{Balanced\ Accuracy}=\frac{TPR+TNR}{2}\]",
    "TPR=TPTP+FN": r"\[TPR=\frac{TP}{TP+FN}\]",
    "TNR=TNTN+FP": r"\[TNR=\frac{TN}{TN+FP}\]",
    "AUC=01TPR(FPR)d(FPR)": r"\[\mathrm{AUC}=\int_0^1 TPR(FPR)\,d(FPR)\]",
    "xi,t∈Rd": r"\[x_{i,t}\in\mathbb{R}^{d}\]",
    "yi,t∈{0,1}": r"\[y_{i,t}\in\{0,1\}\]",
    "pi,t=P(yi,t=1∣xi,t;θ)": r"\[\hat p_{i,t}=P(y_{i,t}=1\mid x_{i,t};\theta)\]",
    "yi,t=1,pi,t≥γ0,pi,t<γ": r"\[\hat y_{i,t}=\begin{cases}1, & \hat p_{i,t}\ge\gamma\\0, & \hat p_{i,t}<\gamma\end{cases}\]",
    "xi,t=[xi,tfin,xi,tmarket,xi,ttech,xtmacro,xi,tindustry,xi,tnews]": r"\[x_{i,t}=[x^{fin}_{i,t},x^{market}_{i,t},x^{tech}_{i,t},x^{macro}_{t},x^{industry}_{i,t},x^{news}_{i,t}]\]",
    "τcgrowth=Qs(gi,t∣industryi=c)": r"\[\tau^{growth}_{c}=Q_s(g_{i,t}\mid industry_i=c)\]",
    "τcgrowth=Qsgi,t∣industryi=c": r"\[\tau^{growth}_{c}=Q_s(g_{i,t}\mid industry_i=c)\]",
    "ROAi,t≥τcROA hoặc ROEi,t≥τcROE": r"\[\mathrm{ROA}_{i,t}\ge\tau^{ROA}_{c}\quad \text{hoặc}\quad \mathrm{ROE}_{i,t}\ge\tau^{ROE}_{c}\]",
    "NetProfitTTMi,t≥τcprofit": r"\[\mathrm{NetProfitTTM}_{i,t}\ge\tau^{profit}_{c}\]",
    "yi,t=1,nếu gi,t>τcgrowth va thỏa đieu kiện chat lượng tài chính0,nếu gi,t≤0Loại bỏ,cac trường hợp trung gian": r"\[y_{i,t}=\begin{cases}1, & \text{nếu } g_{i,t}>\tau^{growth}_{c}\text{ và thỏa điều kiện chất lượng tài chính}\\0, & \text{nếu } g_{i,t}\le0\\\text{Loại bỏ}, & \text{các trường hợp trung gian}\end{cases}\]",
    "pi,t(m)=fm(xi,t;θm)": r"\[\hat p^{(m)}_{i,t}=f_m(x_{i,t};\theta_m)\]",
    "pi,tm=fm(xi,t;θm)": r"\[\hat p^{(m)}_{i,t}=f_m(x_{i,t};\theta_m)\]",
    "pi,t=m=1Mwmpi,t(m)m=1Mwm": r"\[\hat p_{i,t}=\frac{\sum_{m=1}^{M}w_m\hat p^{(m)}_{i,t}}{\sum_{m=1}^{M}w_m}\]",
}


def normalise_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("..", ".")
    text = text.replace("XGBoost (Extreme Gradient Boosting) như là", "XGBoost (Extreme Gradient Boosting) là")
    text = text.replace("Random Forest là mô hình ensemble learning", "Random Forest là mô hình học kết hợp")
    text = text.replace("Giới Thiệu", "Giới thiệu")
    text = text.replace("Kết Luận", "Kết luận")
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace("tài chínhh", "tài chính")
    text = text.replace("kĩ thuật", "kỹ thuật")
    return text.strip()


def compact_formula_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def label_from_caption(prefix: str, caption: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", caption)
    if match:
        return f"{prefix}:{match.group(1)}-{match.group(2)}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", caption.lower()).strip("-")[:32]
    return f"{prefix}:{slug or 'item'}"


def clean_caption(text: str, kind: str) -> str:
    text = normalise_text(text)
    if kind == "fig":
        return re.sub(r"^Hình\s+\d+\.\d+\s*[:.]?\s*", "", text).strip()
    return re.sub(r"^Bảng\s+\d+\.\d+\s*[:.]?\s*", "", text).strip()


def get_paragraph_text_and_images(p, rels) -> tuple[str, list[str]]:
    parts: list[str] = []
    images: list[str] = []

    def walk(node):
        tag = node.tag
        if tag == qn("w:t") or tag == qn("m:t"):
            if node.text:
                parts.append(node.text)
        elif tag == qn("w:tab"):
            parts.append(" ")
        elif tag == qn("a:blip"):
            rid = node.get(qn("r:embed"))
            if rid and rid in rels:
                images.append(Path(rels[rid].target_ref).name)
        for child in node:
            walk(child)

    walk(p)
    return "".join(parts).strip(), images


def extract_table_rows(tbl) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in tbl.findall(".//" + qn("w:tr")):
        row = []
        for tc in tr.findall("./" + qn("w:tc")):
            cell_parts = []
            for p in tc.findall(".//" + qn("w:p")):
                txt = "".join(
                    t.text or ""
                    for t in p.iter()
                    if t.tag in {qn("w:t"), qn("m:t")}
                ).strip()
                if txt:
                    cell_parts.append(txt)
            row.append(normalise_text(" ".join(cell_parts)))
        rows.append(row)
    return rows


def iter_blocks(doc: Document) -> Iterable[Block]:
    rels = doc.part.rels
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text, images = get_paragraph_text_and_images(child, rels)
            text = normalise_text(text)
            if not text and not images:
                continue
            is_list = child.find(".//w:numPr", NS) is not None
            yield ParagraphBlock(text=text, images=images, is_list=is_list)
        elif tag == "tbl":
            rows = extract_table_rows(child)
            if rows:
                yield TableBlock(rows=rows)


def emit_paragraph(text: str) -> str:
    key = compact_formula_key(text)
    if key in FORMULA_MAP:
        return FORMULA_MAP[key] + "\n"
    return latex_escape(text) + "\n\n"


def emit_table(rows: list[list[str]], caption: str | None) -> str:
    max_cols = max(len(r) for r in rows)
    norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]
    width = max(0.12, 0.92 / max_cols)
    cols = "|" + "|".join([f"p{{{width:.2f}\\textwidth}}" for _ in range(max_cols)]) + "|"
    lines = ["\\begin{longtable}{" + cols + "}", "\\hline"]
    if caption:
        cap = clean_caption(caption, "tab")
        label = label_from_caption("tab", caption)
        lines.insert(1, f"\\caption{{{latex_escape(cap)}}}\\label{{{label}}}\\\\")
    for idx, row in enumerate(norm_rows):
        cells = [latex_escape(cell) for cell in row]
        if idx == 0:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
        lines.append("\\hline")
    lines.append("\\end{longtable}\n")
    return "\n".join(lines)


def emit_figure(image_name: str, caption: str | None) -> str:
    label = label_from_caption("fig", caption or image_name)
    cap = clean_caption(caption, "fig") if caption else image_name
    return (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.9\\textwidth]{{figures/{latex_escape(image_name)}}}\n"
        f"\\caption{{{latex_escape(cap)}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n\n"
    )


def is_reference(text: str) -> bool:
    return bool(re.match(r"^\[\d+\]\s+", text))


def emit_reference(text: str) -> str:
    text = normalise_text(text)
    m = re.match(r"^\[(\d+)\]\s*(.*)", text)
    if not m:
        return ""
    return f"\\bibitem{{ref{m.group(1)}}} {latex_escape(m.group(2))}\n"


def emit_heading(text: str) -> str | None:
    if text in {"LỜI CẢM ƠN", "TÓM TẮT", "LỜI CAM ĐOAN"}:
        return f"\\chapter*{{{latex_escape(text)}}}\n\\addcontentsline{{toc}}{{chapter}}{{{latex_escape(text)}}}\n"
    if text == "Tài liệu tham khảo":
        return "\\begin{thebibliography}{99}\n"
    m = re.match(r"^Chương\s+(\d+)[.:]\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        # Lines in "Cấu trúc của khóa luận" have the form
        # "Chương 2. ...: Trình bày ..." and should remain normal prose.
        if ":" in m.group(2):
            return None
        title = normalise_text(m.group(2))
        title = title[0].upper() + title[1:] if title else title
        return f"\\chapter{{{latex_escape(title)}}}\n"
    m = re.match(r"^\d+\.\d+\.\d+\.\s*(.+)$", text)
    if m:
        return f"\\subsection{{{latex_escape(normalise_text(m.group(1)))}}}\n"
    m = re.match(r"^\d+\.\d+\.\s*(.+)$", text)
    if m:
        return f"\\section{{{latex_escape(normalise_text(m.group(1)))}}}\n"
    return None


def preamble() -> str:
    return r"""\documentclass[14pt,a4paper]{extreport}
\usepackage{fontspec}
\usepackage[vietnamese]{babel}
\setmainfont{Times New Roman}
\usepackage{geometry}
\geometry{left=3cm,right=2cm,top=2.5cm,bottom=2.5cm}
\usepackage{setspace}
\onehalfspacing
\usepackage{indentfirst}
\setlength{\parindent}{1.25cm}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\renewcommand{\contentsname}{Mục lục}
\renewcommand{\listfigurename}{Danh sách hình vẽ}
\renewcommand{\listtablename}{Danh sách bảng}
\renewcommand{\chaptername}{Chương}
\captionsetup[figure]{name=Hình}
\captionsetup[table]{name=Bảng}
\begin{document}
\begin{titlepage}
\begin{center}
{\bfseries ĐẠI HỌC QUỐC GIA HÀ NỘI}\\
{\bfseries TRƯỜNG ĐẠI HỌC CÔNG NGHỆ}\\[1cm]
\includegraphics[width=3cm]{figures/image1.png}\\[1cm]
{\bfseries Nguyễn Bảo Sơn}\\[2cm]
{\bfseries\MakeUppercase{Ứng dụng mô hình học máy trong xây dựng hệ thống dự báo xu hướng tăng trưởng doanh nghiệp niêm yết tại Việt Nam trong ngắn hạn và trung hạn dựa trên phân tích tin tức tài chính}}\\[2cm]
{\bfseries KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC HỆ CHÍNH QUY}\\
Ngành: Trí tuệ nhân tạo\\[2cm]
{\bfseries HÀ NỘI - 2026}
\end{center}
\end{titlepage}

\begin{titlepage}
\begin{center}
{\bfseries ĐẠI HỌC QUỐC GIA HÀ NỘI}\\
{\bfseries TRƯỜNG ĐẠI HỌC CÔNG NGHỆ}\\[2cm]
{\bfseries Nguyễn Bảo Sơn}\\[2cm]
{\bfseries\MakeUppercase{Ứng dụng mô hình học máy trong xây dựng hệ thống dự báo xu hướng tăng trưởng doanh nghiệp niêm yết tại Việt Nam trong ngắn hạn và trung hạn dựa trên phân tích tin tức tài chính}}\\[2cm]
{\bfseries KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC HỆ CHÍNH QUY}\\
Ngành: Trí tuệ nhân tạo\\[1.5cm]
Cán bộ hướng dẫn: TS. Trần Hồng Việt\\[2cm]
{\bfseries HÀ NỘI - 2026}
\end{center}
\end{titlepage}

\pagenumbering{roman}
"""


def frontmatter_lists() -> str:
    return r"""
\tableofcontents
\listoffigures
\listoftables
\clearpage
\pagenumbering{arabic}
"""


def build_latex() -> str:
    doc = Document(SOURCE_DOCX)
    blocks = list(iter_blocks(doc))

    # Skip the two DOCX cover pages because LaTeX creates clean title pages above.
    start_idx = next(i for i, b in enumerate(blocks) if isinstance(b, ParagraphBlock) and b.text == "LỜI CẢM ƠN")

    out: list[str] = [preamble()]
    pending_figure_caption: str | None = None
    pending_table_caption: str | None = None
    in_itemize = False
    in_references = False
    inserted_lists = False

    def close_itemize():
        nonlocal in_itemize
        if in_itemize:
            out.append("\\end{itemize}\n")
            in_itemize = False

    for block in blocks[start_idx:]:
        if isinstance(block, TableBlock):
            close_itemize()
            out.append(emit_table(block.rows, pending_table_caption))
            pending_table_caption = None
            continue

        text = block.text
        if text in {"Danh sách hình vẽ", "Danh sách bảng"}:
            break

        if text == "Chương 1. Giới Thiệu" and not inserted_lists:
            close_itemize()
            out.append(frontmatter_lists())
            inserted_lists = True

        if text.startswith("Hình "):
            pending_figure_caption = text
            continue
        if text.startswith("Bảng ") and not text.startswith("Bảng dưới"):
            pending_table_caption = text
            continue

        if block.images:
            close_itemize()
            for image in block.images:
                if image != "image1.png":
                    out.append(emit_figure(image, pending_figure_caption))
            pending_figure_caption = None
            continue

        if is_reference(text):
            close_itemize()
            if not in_references:
                out.append("\\begin{thebibliography}{99}\n")
                in_references = True
            out.append(emit_reference(text))
            continue

        heading = emit_heading(text)
        if heading:
            close_itemize()
            if text == "Tài liệu tham khảo":
                in_references = True
            out.append(heading)
            continue

        if block.is_list:
            if not in_itemize:
                out.append("\\begin{itemize}\n")
                in_itemize = True
            out.append("\\item " + emit_paragraph(text).strip() + "\n")
        else:
            close_itemize()
            out.append(emit_paragraph(text))

    close_itemize()
    if in_references:
        out.append("\\end{thebibliography}\n")
    out.append("\\end{document}\n")
    return "".join(out)


def write_project() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(SOURCE_DOCX) as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                target = FIG_DIR / Path(name).name
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    tex = build_latex()
    (OUT_DIR / "main.tex").write_text(tex, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# KLTN LaTeX\n\n"
        "Biên dịch bằng XeLaTeX để hỗ trợ tiếng Việt và font Times New Roman:\n\n"
        "```powershell\n"
        "xelatex main.tex\n"
        "xelatex main.tex\n"
        "```\n\n"
        "Nếu thiếu font Times New Roman trên hệ thống LaTeX, đổi `\\setmainfont{Times New Roman}` trong `main.tex` sang font có sẵn.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_project()
    print(f"Wrote LaTeX project to: {OUT_DIR}")
