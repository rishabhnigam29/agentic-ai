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