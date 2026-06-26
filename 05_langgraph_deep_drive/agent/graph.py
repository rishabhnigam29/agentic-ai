"""Main due-diligence orchestration graph"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import DueDiligenceState
from agent.nodes import intake_node

def build_graph() -> StateGraph:
    """Construct the stategraph and return it (uncompiled)"""
    g = StateGraph(DueDiligenceState)
    
    g.add_node("intake", intake_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", END)

    return g