"""Financial analysis subgraph - extracts financials then scores them"""

from __future__ import annotations
from langgraph.graph import StateGraph

from agent.state import FinancialSubState
from shared.prompts import FINANCIAL_EXTRACT_PROMPT, FINANCIAL_ANALYZE_PROMPT
from subgraphs.base import build_extract_analyze_subgraph

def build_financial_subgraph() -> StateGraph:
    """Builds the financial analysis subgraph."""
    return build_extract_analyze_subgraph(
        branch_name = "financial",
        state_cls = FinancialSubState,
        extract_prompt = FINANCIAL_EXTRACT_PROMPT,
        analyze_prompt = FINANCIAL_ANALYZE_PROMPT,
    )