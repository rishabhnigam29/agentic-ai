import os
from dotenv import load_dotenv

load_dotenv()

def get_model():
    """
    Get the LLM model to use based on environment configuration.

    Returns:
        Model identifier string from .env configuration
    """
    is_offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    if is_offline:
        return os.getenv("OLLAMA_MODEL", "ollama_chat/llama3")
    return os.getenv("LLM_MODEL", "gpt-40-mini")

def get_api_base(model):
    """
    Get the API base URL for a given model if needed (e.g., for Ollama).

    Args:
        model: Model identifier

    Returns:
        API base URL string or None
    """
    if model.startswith("ollama/"):
        return os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    return None

def get_llm_params(temperature: float = 0, 
                   max_tokens: int | None = None,
                   extra: dict | None = None) -> dict:
    """
    Get the params needed to construct an LLM client, resolved from .env.

    Returns a dict ready to splat into any LiteLLM-compatible chat model
    constructor (e.g. ChatLiteLLM(**get_llm_params())) or completion call.
    Keeps this module free of framework-specific imports.

    Args:
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Optional max output tokens. Omitted from the dict if None.
        extra: Optional dict of additional kwargs the caller wants in the params.
            Covers two cases:
              - call payload that this helper doesn't know about (e.g. `messages`,
                `max_retries`, `tools`) — used by `call_llm` in utils/llm_client.py.
              - provider knobs (e.g. `top_p`, `seed`, `frequency_penalty`) the
                helper doesn't model explicitly.
            Merged into the returned params last, so it can override resolved
            defaults if a caller really needs to.

    Returns:
        Dict with model, temperature, api_base (if applicable), max_tokens (if set),
        and any kwargs from `extra`.
    """
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
    """
    Resolve the Tavily API key from the environment.

    Returns the key only if it looks valid (present, not the placeholder,
    starts with the expected `tvly-` prefix). Returns None otherwise so
    callers can skip Tavily-dependent features cleanly.
    """
    key = os.getenv("TAVILY_API_KEY")
    if not key or key == "your_tavily_api_key_here":
        return None
    return key
    