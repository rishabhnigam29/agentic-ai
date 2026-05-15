import os
from dotenv import load_dotenv

load_dotenv()

def get_model():
    is_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    if is_offline:
        return os.getenv("OLLAMA_MODEL", "ollama_chat/llama3")
    return os.getenv("LLM_MODEL", "gpt-40-mini")

def get_api_base(model):
    if model.startswith("ollama/"):
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    return None

def get_llm_params(temperature: float = 0, 
                   max_tokens: int | None = None,
                   extra: dict | None = None) -> dict:
    model = get_model()
    params: dict = {
        "model": model,
        "temperature": temperature,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    
    api_base = get_api_base(model)
    if api_base:
        params["api_base"] = api_base
    if extra:
        params.update(extra)
    return params

def get_tavily_api_key() -> str | None:
    key = os.getenv("TAVILY_API_KEY")
    if not key or key == "your_tavily_api_key_here":
        return None
    return key
    