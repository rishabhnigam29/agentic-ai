from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM

from utils.llm_config import get_llm_params
from tools import get_all_tools

SYSTEM_PROMPT = """You are a helpful financial assistant. 
Use 'get_stock_price' for live ticker prices.
Use web search tool for news, headlines or anything else that requires current
information from the web.
If neither of the tools is needed, just answer the question directly."""

def build_agent():
    """
    Create a ReAct agent executor with stock price tool and Tavily search.

    Returns:
        Agent instance
    """
    llm = ChatLiteLLM(**get_llm_params(max_tokens=2000))
    
    # Create ReAct agent using LangChain's create_agent utility, which sets up the agent loop and tool calling based on the provided system prompt.
    return create_agent(model=llm, tools=get_all_tools(), system_prompt=SYSTEM_PROMPT)

def run_agent(user_question: str)-> str:
    """
    Run a ReAct agent that can call tools (stock price, Tavily search) based on user query.

    The agent uses the ReAct pattern:
    - Reasoning: Thinks about what tools to use
    - Acting: Executes tools to gather information
    - Observing: Uses tool results to inform next steps
    - Repeating until final answer

    Configuration is handled via .env file.

    Args:
        user_query: The user's question or request

    Returns:
        The agent's final response after potentially calling tools
    """
    # Create agent (all config from .env)
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return result["messages"][-1].content

if __name__ == "__main__":
    question = "What is the current stock price of Apple Inc. and any recent news about the company?"
    answer = run_agent(question)
    print(answer)
    