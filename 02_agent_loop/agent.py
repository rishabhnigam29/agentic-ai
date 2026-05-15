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
    llm = ChatLiteLLM(**get_llm_params(max_tokens=2000))
    return create_agent(model=llm, tools=get_all_tools(), system_prompt=SYSTEM_PROMPT)

def run_agent(user_question: str)-> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return result["messages"][-1].content

if __name__ == "__main__":
    question = "What is the current stock price of Apple Inc. and any recent news about the company?"
    answer = run_agent(question)
    print(answer)
    