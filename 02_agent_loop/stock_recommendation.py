from langchain_litellm import ChatLiteLLM
from utils.llm_config import get_llm_params
from agent import run_agent
from models import StockRecommendation

def get_stock_recommendation(symbol: str) -> StockRecommendation:
    gathering_prompt = (
        f"Analyze the stock {symbol}."
        "Get the current price. Get the latest market news. "
        "Then, write a short opinion: Buy, Sell, or Hold. with reasoning and confidence score." 
    )
    free_form_answer = run_agent(gathering_prompt)

    print("Raw agent output:", free_form_answer)

    llm = ChatLiteLLM(**get_llm_params(max_tokens=2000))
    structured_llm = llm.with_structured_output(StockRecommendation)

    messages = [
        {"role": "system", "content": "You convert free-form stock analysis into a structured recommendation."},
        {"role": "user", "content": f"Symbol: {symbol}\n Free-form analysis: {free_form_answer}\n\n Produce the StockRecommendation."}
    ]

    return structured_llm.invoke(messages)

if __name__ == "__main__":
    rec = get_stock_recommendation("AAPL")
    print("Structured Stock Recommendation:")
    print(rec.model_dump_json(indent=2))