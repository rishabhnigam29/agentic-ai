from litellm import completion
from dotenv import load_dotenv
from utils.llm_config import get_model, get_api_base

load_dotenv()


def call_llm(messages, temperature=0):
    model = get_model()
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    api_base=get_api_base(model)
    if api_base:
        params["api_base"] = api_base
    return completion(**params)