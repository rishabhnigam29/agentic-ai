""" Node functions for the due diligence graph """

from __future__ import annotations
import time
from utils.llm_client import call_llm
from shared.prompts import INTAKE_PROMPT, AGGREGATE_PROMPT
from agent.state import DueDiligenceState, FinancialSubState, MarketSubState, TeamSubState, VisualSubState

from subgraphs.financials import build_financial_subgraph
from subgraphs.market import build_market_subgraph
from subgraphs.team import build_team_subgraph
from subgraphs.visual import build_visual_subgraph
from shared.models import DueDiligenceReport

from shared.config import BRANCH_WEIGHTS, AUTO_APPROVE_THRESHOLD, AUTO_REJECT_THRESHOLD

SubState = FinancialSubState | MarketSubState | TeamSubState | VisualSubState

def _run_subgraph(subgraph_builder, init_state: SubState, branch_name: str) -> dict:
    """Invoke a compiled subgraph and stash the results in the main state."""
    compiled = subgraph_builder().compile()
    result = compiled.invoke(init_state)
    return {
        "branch_results": [
            {
                "branch": branch_name,   
                "score": result["score"],
                "summary": result["summary"],
                "key_findings": result["key_findings"],
                "risks": result["risks"],
            },
        ],
    }

def aggregate_node(state: DueDiligenceState) -> dict:
    """Compute weighted overall score and set recommendation based on thresholds."""
    results = state["branch_results"]
    print("Branch Results:", state["branch_results"])

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
    """Placeholder node where the graph pauses for human input. 
    In a real application, this would be replaced with a mechanism to collect human feedback."""
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

def visual_analysis_node(state: DueDiligenceState) -> dict:
    """Run the visual analysis subgraph."""
    time.sleep(10)  
    return _run_subgraph(
        build_visual_subgraph,
        VisualSubState(
            pitch_deck=state["pitch_deck"],
            pitch_deck_images=state.get("pitch_deck_images", []),
            extracted_info={},
            score=0.0,
            summary="",
            key_findings=[],
            risks=[],
        ),
        "visual",
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

def generate_report_node(state: DueDiligenceState) -> dict:
    """Produce the final due diligence report based on the aggregated results."""
    time.sleep(10)  
    results = state["branch_results"]
    branch_texts = "\n\n".join(
        f"##Branch: {r['branch'].upper()} \nScore: {r['score']}/100"
        "\nSummary: {r['summary']}"
        "\nFindings: {', '.join(r['key_findings'])}"
        "\nRisks: {', '.join(r['risks'])}"
        for r in results
    )

    human_section = ""
    if state.get("human_decision") and state["recommendation"] == "review":
        human_section = (
            f"\n\nHuman reviewer decision: {state['human_decision'].upper()}"
            f"\n Human Feedback: {state.get('human_feedback', 'N/A')}"
        )

    report: DueDiligenceReport = call_llm(
        messages=[
            {"role": "system", "content": AGGREGATE_PROMPT},
            {"role": "user", 
             "content": (
             f"Startup: {state['startup_name']}\n\n"
             f"Overall weighted score: {state['overall_score']}/100\n\n"
             f"Automated recommendation: {state['recommendation'].upper()}\n\n"
             f"{human_section}\n\n"
             f"Branch Analysis Results:\n{branch_texts}"
             ),
            },
        ],
        temperature=0,
        response_model=DueDiligenceReport,
    )

    final_decision = state.get("human_decision") or state["recommendation"]

    lines = [
        "="*60,
        f" DUE DILIGENCE REPORT FOR: {state['startup_name'].upper()} ",
        "="*60,
        "",
        f"Overall Weighted Score: {state['overall_score']}/100",
        f"Automated Recommendation: {final_decision.upper()}",
        "",
        "Executive Summary:",
        "-"*40,
        report.executive_summary,
        "",
    ]

    for branch, summary in report.branch_summaries.items():
        lines.append(f"{branch.upper()} Summary: {summary} \n" )
    lines.append("")

    lines.append("Top 3 Strengths:")
    lines.append("-"*40)
    for strength in report.top_strengths:
        lines.append(f"- {strength}")
    lines.append("")

    lines.append("Top 3 Risks:")
    lines.append("-"*40)
    for risk in report.top_risks:
        lines.append(f"- {risk}")
    lines.append("")

    lines.append("Final Investment Verdict:")
    lines.append("-"*40)
    lines.append(report.final_verdict)
    lines.append("="*60)

    return {"final_report": "\n".join(lines)}

