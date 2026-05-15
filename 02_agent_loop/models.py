from typing import List
from pydantic import BaseModel, Field

class StockRecommendation(BaseModel):
    symbol: str = Field(description="The stock symbol, e.g., AAPL for Apple Inc.")
    action: str = Field(description="One of 'buy', 'sell', or 'hold'.")
    rationale: str = Field(description="A brief explanation for the recommendation.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="How confident we are, 0 to 1.")
    key_headlines: List[str] = Field(description="Headlines or facts that drove the recommendation.")