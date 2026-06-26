"""Main due-diligence orchestration graph"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import DueDiligenceState
from agent.nodes import intake_node, financial_analysis_node, market_analysis_node, team_analysis_node

def build_graph() -> StateGraph:
    """Construct the stategraph and return it (uncompiled)"""
    g = StateGraph(DueDiligenceState)
    
    g.add_node("intake", intake_node)
    g.add_node("financial_analysis", financial_analysis_node)
    g.add_node("market_analysis", market_analysis_node)
    g.add_node("team_analysis", team_analysis_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "financial_analysis")
    g.add_edge("financial_analysis", "market_analysis")
    g.add_edge("market_analysis", "team_analysis")
    g.add_edge("team_analysis", END)

    return g