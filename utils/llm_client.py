from typing import TypeVar, Type, Optional, List, Dict, Any
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv
from utils.llm_config import get_model, get_api_base, get_llm_params
import instructor

load_dotenv()

# LiteLLM unified interface — supports 100+ providers including Ollama.
# Mode.JSON works across cloud (OpenAI / Anthropic) and local (Ollama) reliably;
# the default TOOLS mode breaks down on small local models like Llama 3.1 8B.
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
    """
    Generic LLM caller that can be used throughout the project.
    
    Model is configured via .env file.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        response_model: Optional Pydantic model for structured output
        temperature: Sampling temperature (0 for deterministic)
        max_retries: Number of retries on validation failure
        tools: Optional list of tool definitions for function calling
        tool_choice: Tool choice mode ("auto", "none", "required", or tool name)
        return_raw_response: If True, returns raw response object (useful for tool calling loops)
    
    Returns:
        If return_raw_response=True: Returns raw response object
        If response_model provided: Returns validated Pydantic model instance
        Otherwise: Returns raw response content string
    """
    # Resolve model + api_base from .env via shared params builder.
    # Pass call-specific payload (messages, max_retries) through `extra`.
    params: Dict[str, Any] = get_llm_params(
        temperature=temperature,
        extra={
            "messages": messages,
            "max_retries": max_retries,
        },
        )
    
    # Add tools if provided (for function calling)
    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice
    # Add response_model if provided (for structured outputs)
    if response_model:
        params["response_model"] = response_model
        params["max_retries"] = max_retries
        return client.chat.completions.create(**params)
      
    response = completion(**params)
    if return_raw_response:
        return response
    return response.choices[0].message.content