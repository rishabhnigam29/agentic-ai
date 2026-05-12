from litellm import completion
from dotenv import load_dotenv
from utils.llm_config import get_model, get_api_base
import instructor

load_dotenv()

client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)

def call_llm(messages, response_model=None,
             temperature=0, max_retries=2):
    model = get_model()
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    api_base=get_api_base(model)
    if api_base:
        params["api_base"] = api_base

    if response_model:
        params["response_model"] = response_model
        params["max_retries"] = max_retries
        return client.chat.completions.create(**params)
      
    return completion(**params)