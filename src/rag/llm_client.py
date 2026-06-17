import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]


class LLMClient:
    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()

        if self.provider == "gemini":
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Thiếu GEMINI_API_KEY trong file .env")

            genai.configure(api_key=api_key)
            configured_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
            self.model_name = configured_model.replace("models/", "")
            self.model = genai.GenerativeModel(self.model_name)

        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_key_here":
                raise ValueError("Thiếu OPENAI_API_KEY trong file .env")

            self.client = OpenAI(api_key=api_key)
            self.model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()

        else:
            raise ValueError("provider phải là 'gemini' hoặc 'openai'")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            if self.provider == "gemini":
                return self._generate_gemini(prompt)

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or "LLM không trả về nội dung."

        except Exception as e:
            message = str(e)
            if "429" in message or "quota" in message.lower() or "insufficient_quota" in message:
                return "LLM_API_QUOTA_EXCEEDED"
            return f"Lỗi khi gọi LLM: {message}"

    def _generate_gemini(self, prompt: str) -> str:
        tried = []
        model_names = [self.model_name, *GEMINI_FALLBACK_MODELS]

        for model_name in dict.fromkeys(model_names):
            tried.append(model_name)
            try:
                model = self.model if model_name == self.model_name else self._new_gemini_model(model_name)
                response = model.generate_content(prompt)
                if model_name != self.model_name:
                    self.model_name = model_name
                    self.model = model
                return getattr(response, "text", "") or "LLM không trả về nội dung."
            except Exception as exc:
                message = str(exc)
                if "not found" not in message and "not supported" not in message:
                    raise

        return f"Lỗi khi gọi LLM: không có Gemini model khả dụng. Đã thử: {', '.join(tried)}"

    @staticmethod
    def _new_gemini_model(model_name: str):
        import google.generativeai as genai

        return genai.GenerativeModel(model_name.replace("models/", ""))
