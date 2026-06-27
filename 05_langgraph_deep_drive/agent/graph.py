"""Main due-diligence orchestration graph"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import DueDiligenceState
from agent.nodes import aggregate_node, intake_node, financial_analysis_node, market_analysis_node, team_analysis_node

def build_graph() -> StateGraph:
    """Construct the stategraph and return it (uncompiled)"""
    g = StateGraph(DueDiligenceState)
    
    g.add_node("intake", intake_node)
    g.add_node("financial_analysis", financial_analysis_node)
    g.add_node("market_analysis", market_analysis_node)
    g.add_node("team_analysis", team_analysis_node)
    g.add_node("aggregate", aggregate_node)

    g.add_edge(START, "intake")

    # Fan-out: intake -> 3 parallel branches
    g.add_edge("intake", "financial_analysis")
    g.add_edge("intake", "market_analysis")
    g.add_edge("intake", "team_analysis")

    # Fan-in: 3 parallel branches -> aggregate
    g.add_edge("financial_analysis", "aggregate")
    g.add_edge("market_analysis", "aggregate")
    g.add_edge("team_analysis", "aggregate")

    g.add_edge("aggregate", END)

    return g