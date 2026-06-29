""" Node functions for the due diligence graph """

from __future__ import annotations
from utils.llm_client import call_llm
from shared.prompts import INTAKE_PROMPT
from agent.state import DueDiligenceState, FinancialSubState, MarketSubState, TeamSubState

from subgraphs.financials import build_financial_subgraph
from subgraphs.market import build_market_subgraph
from subgraphs.team import build_team_subgraph

from shared.config import BRANCH_WEIGHTS, AUTO_APPROVE_THRESHOLD, AUTO_REJECT_THRESHOLD

SubState = FinancialSubState | MarketSubState | TeamSubState

def _run_subgraph(subgraph_builder, init_state: SubState, branch_name: str) -> dict:
    """Invoke a compiled subgraph and stash the results in the main state."""
    compiled = subgraph_builder().compile()
    result = compiled.invoke(init_state)
    return {
        "branch_result": {
            "branch": branch_name,   
            "score": result["score"],
            "summary": result["summary"],
            "key_findings": result["key_findings"],
            "risks": result["risks"],
        },
    }

def aggregate_node(state: DueDiligenceState) -> dict:
    """Compute weighted overall score and set recommendation based on thresholds."""
    results = state["branch_results"]
    score_by_branch = {r["branch"]: r["score"] for r in results}

    weighted_score = sum(
        score_by_branch.get(branch, 50.0) * weight
        for branch, weight in BRANCH_WEIGHTS.items()
    ) 

    if weighted_score >= AUTO_APPROVE_THRESHOLD:
        recommendation = "approve"  
    elif weighted_score <= AUTO_REJECT_THRESHOLD:
        recommendation = "reject"
    else:
        recommendation = "review"

    return {
        "overall_score": round(weighted_score, 1),
        "recommendation": recommendation,
    }

def auto_approve_node(state: DueDiligenceState) -> dict:
    """Stamp the deal as auto-approved"""
    return {"human_decision": "approve", "human_feedback" : "Auto-approved based on high overall score."}

def auto_reject_node(state: DueDiligenceState) -> dict:
    """Stamp the deal as auto-rejected"""
    return {"human_decision": "reject", "human_feedback" : "Auto-rejected based on low overall score."}

def human_review_node(state: DueDiligenceState) -> dict:
    """Placeholder - later lesson turns this into a real inerrupt"""
    return {}

def route_after_aggregate(state: DueDiligenceState) -> dict:
    return state["recommendation"]

def financial_analysis_node(state: DueDiligenceState) -> dict:
    """Run the financial analysis subgraph."""
    return _run_subgraph(
        build_financial_subgraph,
        FinancialSubState(
            pitch_deck=state["pitch_deck"],
            extracted_info={},
            score=0.0,
            summary="",
            key_findings=[],
            risks=[],
        ),
        "financial",
    )

def market_analysis_node(state: DueDiligenceState) -> dict:
    """Run the market analysis subgraph."""
    return _run_subgraph(
        build_market_subgraph,
        MarketSubState(
            pitch_deck=state["pitch_deck"],
            extracted_info={},
            score=0.0,
            summary="",
            key_findings=[],
            risks=[],
        ),
        "market",
    )

def team_analysis_node(state: DueDiligenceState) -> dict:
    """Run the team analysis subgraph."""
    return _run_subgraph(
        build_team_subgraph,
        TeamSubState(
            pitch_deck=state["pitch_deck"],
            extracted_info={},
            score=0.0,
            summary="",
            key_findings=[],
            risks=[],
        ),
        "team",
    )



def intake_node(state: DueDiligenceState) -> dict:
    """Parse the pitch deck and extract the startup name."""
    name: str = call_llm(
        messages=[
            {"role": "system", "content": INTAKE_PROMPT}, 
            {"role": "user", "content": state["pitch_deck"][:3000]}
            ],
            temperature=0,
    )
    return {"startup_name": name.strip().strip('"').strip("'")}
