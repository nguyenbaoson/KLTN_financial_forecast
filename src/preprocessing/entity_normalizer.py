import pandas as pd
import hashlib

# Bảng mapping: tên gọi khác nhau → tên chuẩn + mã cổ phiếu
COMPANY_MAP = {
    # FPT
    "fpt": ("FPT Corporation", "FPT"),
    "fpt telecom": ("FPT Corporation", "FPT"),
    "tập đoàn fpt": ("FPT Corporation", "FPT"),

    # Vinamilk
    "vinamilk": ("Vinamilk", "VNM"),
    "công ty sữa việt nam": ("Vinamilk", "VNM"),
    "vnm": ("Vinamilk", "VNM"),

    # Vietcombank
    "vietcombank": ("Vietcombank", "VCB"),
    "vcb": ("Vietcombank", "VCB"),
    "ngân hàng ngoại thương": ("Vietcombank", "VCB"),

    # Hòa Phát
    "hòa phát": ("Hòa Phát Group", "HPG"),
    "tập đoàn hòa phát": ("Hòa Phát Group", "HPG"),
    "hpg": ("Hòa Phát Group", "HPG"),

    # Thế Giới Di Động
    "thế giới di động": ("MWG", "MWG"),
    "mwg": ("MWG", "MWG"),
    "tgdd": ("MWG", "MWG"),
}

def normalize_company_name(text: str):
    """
    Tìm và trả về (tên_chuẩn, mã_CK) nếu phát hiện công ty trong text.
    Trả về (None, None) nếu không tìm thấy.
    """
    text_lower = text.lower()
    for keyword, (canonical, ticker) in COMPANY_MAP.items():
        if keyword in text_lower:
            return canonical, ticker
    return None, None

def make_content_hash(text: str) -> str:
    """Tạo hash để phát hiện bài trùng nội dung"""
    # Chuẩn hóa trước khi hash: lowercase, xóa khoảng trắng thừa
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bài trùng dựa trên URL và nội dung"""
    before = len(df)

    # Loại trùng theo URL
    df = df.drop_duplicates(subset=["url"], keep="first")

    # Loại trùng theo nội dung (hash)
    df["content_hash"] = df["content"].apply(make_content_hash)
    df = df.drop_duplicates(subset=["content_hash"], keep="first")

    after = len(df)
    print(f"Loại {before - after} bài trùng. Còn lại: {after} bài.")
    return df

def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột company_name và ticker vào DataFrame"""
    companies, tickers = [], []
    for _, row in df.iterrows():
        # Tìm công ty từ title + content
        full_text = f"{row.get('title', '')} {row.get('content', '')}"
        company, ticker = normalize_company_name(full_text)

        # Nếu không tìm được từ nội dung, dùng keyword gốc
        if not company and "keyword" in row:
            kw = row["keyword"].lower()
            company, ticker = COMPANY_MAP.get(kw, (row.get("keyword"), None))

        companies.append(company)
        tickers.append(ticker)

    df["company_name"] = companies
    df["ticker"] = tickers
    return df