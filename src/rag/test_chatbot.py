try:
    from .chatbot import RAGChatbot
except ImportError:
    from chatbot import RAGChatbot


def main():
    bot = RAGChatbot(
        backend="faiss",       # đổi thành "chroma" nếu muốn
        llm_provider="gemini", # hoặc "openai"
        top_k=5,
    )

    print("\nTEST 1")
    result = bot.ask("FPT có kế hoạch tăng trưởng gì trong thời gian tới?")
    bot.print_response(result)

    print("\nTEST 2")
    result = bot.ask("doanh thu gần đây thế nào?", ticker="VNM")
    bot.print_response(result)

    print("\nTEST 3 - SUMMARY")
    summary = bot.summarize("HPG")
    print(summary)

    print("\nTEST 4 - ngoài dữ liệu")
    result = bot.ask("Apple sẽ ra iPhone 17 khi nào?")
    bot.print_response(result)


if __name__ == "__main__":
    main()
