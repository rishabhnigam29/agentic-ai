"""Main due-diligence orchestration graph"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import DueDiligenceState
from agent.nodes import aggregate_node, generate_report_node, intake_node, financial_analysis_node, market_analysis_node, team_analysis_node, auto_approve_node, auto_reject_node, human_review_node, route_after_aggregate, visual_analysis_node

def build_graph() -> StateGraph:
    """Construct the stategraph and return it (uncompiled)"""
    g = StateGraph(DueDiligenceState)
    
    g.add_node("intake", intake_node)
    g.add_node("financial_analysis", financial_analysis_node)
    g.add_node("market_analysis", market_analysis_node)
    g.add_node("team_analysis", team_analysis_node)
    g.add_node("visual_analysis", visual_analysis_node)
    g.add_node("aggregate", aggregate_node)

    #Decision nodes
    g.add_node("auto_approve", auto_approve_node)
    g.add_node("auto_reject", auto_reject_node)
    g.add_node("human_review", human_review_node)

    g.add_node("generate_report", generate_report_node)

    g.add_edge(START, "intake")

    # Fan-out: intake -> 4 parallel branches
    g.add_edge("intake", "financial_analysis")
    g.add_edge("intake", "market_analysis")
    g.add_edge("intake", "team_analysis")
    g.add_edge("intake", "visual_analysis")
    # Fan-in: 4 parallel branches -> aggregate
    g.add_edge("financial_analysis", "aggregate")
    g.add_edge("market_analysis", "aggregate")
    g.add_edge("team_analysis", "aggregate")
    g.add_edge("visual_analysis", "aggregate")

    # Conditional routing based on score
    g.add_conditional_edges("aggregate", 
                            route_after_aggregate,
                            {
                                "approve": "auto_approve",
                                "reject": "auto_reject",
                                "review": "human_review"
                            },
                            )
    
    g.add_edge("auto_approve", "generate_report")
    g.add_edge("auto_reject", "generate_report")
    g.add_edge("human_review", "generate_report")

    g.add_edge("generate_report", END)

    return g