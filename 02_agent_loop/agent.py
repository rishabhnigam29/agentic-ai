from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM

from utils.llm_config import get_llm_params
from tools import get_stock_price

SYSTEM_PROMPT = """You are a helpful financial assistant. 
When asked about the stock price, call the 'get_stock_price'
Otherwise, answer directly."""

def build_agent():
    llm = ChatLiteLLM(**get_llm_params(max_tokens=2000))
    return create_agent(model=llm, tools=[get_stock_price], system_prompt=SYSTEM_PROMPT)

def run_agent(user_question: str)-> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_question}]})
    return result["messages"][-1].content

if __name__ == "__main__":
    question = "What is the current stock price of Apple Inc. (AAPL)?"
    answer = run_agent(question)
    print(answer)
    