# Cách đơn giản nhất: dùng từ điển từ khóa tài chính tiếng Việt
# Sau này có thể thay bằng PhoBERT fine-tuned
import pandas as pd


POSITIVE_KEYWORDS = [
    "tăng trưởng", "lợi nhuận tăng", "doanh thu tăng", "mở rộng",
    "thắng lợi", "vượt kế hoạch", "kỷ lục", "tích cực", "khả quan",
    "đột phá", "tăng mạnh", "phục hồi", "cải thiện", "hiệu quả cao",
    "hợp đồng lớn", "thị phần tăng", "đầu tư mới"
]

NEGATIVE_KEYWORDS = [
    "giảm", "lỗ", "nợ xấu", "thua lỗ", "sụt giảm", "khó khăn",
    "rủi ro", "kiện tụng", "vi phạm", "xử phạt", "thoái vốn",
    "cắt giảm", "đình trệ", "chậm tiến độ", "mất hợp đồng"
]

def simple_sentiment(text: str) -> dict:
    """
    Phân tích cảm xúc đơn giản dựa trên từ điển.
    Trả về: {'label': 'positive'|'negative'|'neutral', 'score': float}
    """
    text_lower = text.lower()

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    total = pos_count + neg_count

    if total == 0:
        return {"label": "neutral", "score": 0.0, "pos": 0, "neg": 0}

    score = (pos_count - neg_count) / total  # -1.0 đến +1.0

    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": round(score, 3),
            "pos": pos_count, "neg": neg_count}


def add_sentiment_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột sentiment vào DataFrame"""
    sentiments = df["content"].apply(simple_sentiment)
    df["sentiment_label"] = sentiments.apply(lambda x: x["label"])
    df["sentiment_score"] = sentiments.apply(lambda x: x["score"])
    return df