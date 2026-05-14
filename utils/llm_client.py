from typing import TypeVar, Type, Optional, List, Dict, Any
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv
from utils.llm_config import get_model, get_api_base, get_llm_params
import instructor

load_dotenv()

client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)

# Generic type variable for response models. Bound to BaseModel so callers
# get a properly typed Pydantic instance back, not just 'BaseModel'. 
T = TypeVar("T", bound=BaseModel)

def call_llm(messages: List[Dict[str, Any]],
             response_model: Optional[Type[T]] = None,
             temperature: float = 0,
             max_retries: int = 2,
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: str = "auto",
             return_raw_response: bool = False) -> Any:
    params: Dict[str, Any] = get_llm_params(
        temperature=temperature,
        extra={
            "messages": messages,
            "max_retries": max_retries,
        },
        )
    
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice

    if response_model:
        params["response_model"] = response_model
        params["max_retries"] = max_retries
        return client.chat.completions.create(**params)
      
    response = completion(**params)
    if return_raw_response:
        return response
    return response.choices[0].message.content