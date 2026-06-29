"""Market analysis subgraph - extracts markets then scores them"""

from __future__ import annotations
from langgraph.graph import StateGraph

from agent.state import MarketSubState
from shared.prompts import MARKET_EXTRACT_PROMPT, MARKET_ANALYZE_PROMPT
from subgraphs.base import build_extract_analyze_subgraph

def build_market_subgraph() -> StateGraph:
    """Builds the market analysis subgraph."""
    return build_extract_analyze_subgraph(
        branch_name = "market",
        state_cls = MarketSubState,
        extract_prompt = MARKET_EXTRACT_PROMPT,
        analyze_prompt = MARKET_ANALYZE_PROMPT,
    )