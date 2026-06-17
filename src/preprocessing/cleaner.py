import re
import unicodedata
import pandas as pd

def remove_html(text: str) -> str:
    """Xóa toàn bộ thẻ HTML"""
    text = re.sub(r'<[^>]+>', ' ', text)
    # Xóa các entity HTML như &nbsp; &amp;
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text

def normalize_unicode(text: str) -> str:
    """
    Chuẩn hóa Unicode tiếng Việt về dạng NFC.
    Quan trọng: tiếng Việt có 2 cách mã hóa dấu,
    nếu không chuẩn hóa thì cùng 1 chữ sẽ ra 2 vector khác nhau.
    """
    return unicodedata.normalize('NFC', text)

def clean_whitespace(text: str) -> str:
    """Xóa khoảng trắng thừa, ký tự xuống dòng dư"""
    text = re.sub(r'\n+', '\n', text)       # nhiều dòng → 1 dòng
    text = re.sub(r'[ \t]+', ' ', text)     # nhiều space → 1 space
    text = text.strip()
    return text

def remove_noise(text: str) -> str:
    """Xóa ký tự rác không cần thiết"""
    # Xóa URL
    text = re.sub(r'https?://\S+', '', text)
    # Xóa email
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Xóa ký tự đặc biệt thừa (giữ lại dấu câu tiếng Việt)
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)%/]', ' ', text)
    return text

def clean_text(text: str) -> str:
    """Pipeline làm sạch đầy đủ — gọi hàm này cho mỗi bài báo"""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_html(text)
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = clean_whitespace(text)
    return text


# ========== Test thử ==========
if __name__ == "__main__":
    sample = """
    <p>FPT &nbsp; công bố <b>doanh thu</b> quý 3/2024 tăng 20%
    so với cùng kỳ năm ngoái.  Xem chi tiết tại: https://fpt.com.vn  </p>
    """
    result = clean_text(sample)
    print("Kết quả:", result)
    # → "FPT công bố doanh thu quý 3/2024 tăng 20% so với cùng kỳ năm ngoái."