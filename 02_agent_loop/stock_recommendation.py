from langchain_litellm import ChatLiteLLM
from utils.llm_config import get_llm_params
from utils.tracing import setup_tracing
from agent import run_agent
from models import StockRecommendation

setup_tracing(project_name="module2-research-agent")

def get_stock_recommendation(symbol: str) -> StockRecommendation:
    """
    Get a structured stock recommendation using the agent's tools and LLM analysis.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

    Returns:
        StockRecommendation with structured recommendation data
    """
    # Step 1: agent gathers messy data via tools.
    gathering_prompt = (
        f"Analyze the stock {symbol}."
        "Get the current price. Get the latest market news. "
        "Then, write a short opinion: Buy, Sell, or Hold. with reasoning and confidence score." 
    )
    free_form_answer = run_agent(gathering_prompt)

    print("Raw agent output:", free_form_answer)

    # Step 2: bind the same LiteLLM model to the Pydantic schema via LangChain.
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