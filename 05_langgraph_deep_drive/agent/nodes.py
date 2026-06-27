""" Node functions for the due diligence graph """

from __future__ import annotations
from utils.llm_client import call_llm
from shared.prompts import INTAKE_PROMPT
from agent.state import DueDiligenceState, FinancialSubState, MarketSubState, TeamSubState

from subgraphs.financials import build_financial_subgraph
from subgraphs.market import build_market_subgraph
from subgraphs.team import build_team_subgraph

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
    """Stub - properly fleshed out next lesson"""
    for r in state["branch_results"]:
        print(f" {r['branch']:<10s} score: {r['score']:.0f}/100")
    return {} 

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
