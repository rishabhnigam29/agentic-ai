"""Common factory for building extracts -> analyze subgraph."""

from __future__ import annotations
from typing import Any

from langgraph.graph import StateGraph, END

from shared.models import BranchAnalysis
from utils.llm_client import call_llm

def build_extract_analyze_subgraph(
    *,
    branch_name: str,
    state_cls: type,
    extract_prompt: str,
    analyze_prompt: str,
    )-> StateGraph:
    """Builds a subgraph for extracting and analyzing information from a pitch deck."""

    def extract(state: dict[str, Any]) -> dict:
        extracted: str = call_llm(
            messages = [
                {"role": "system", "content": extract_prompt},
                {"role": "user", "content": state["pitch_deck"]},
            ],
            temperature = 0.0,
        )
        return {"extracted_info": extracted}

    def analyze(state: dict[str, Any]) -> dict:
        result: BranchAnalysis = call_llm(
            messages = [
                {"role": "system", "content": analyze_prompt},
                {"role": "user", "content": f"Extracted {branch_name} data : \n\n {state['extracted_info']}"},
            ],
            temperature = 0.0,
            response_model = BranchAnalysis,
        )
        #print(f"Analysis result for {branch_name}: {result}")
        return {
            "score": result.score,
            "summary": result.summary,
            "key_findings": result.key_findings,
            "risks": result.risks,
        }
    
    g = StateGraph(state_cls)
    g.add_node("extract", extract)
    g.add_node("analyze", analyze)
    g.set_entry_point("extract")
    g.add_edge("extract", "analyze")
    g.add_edge("analyze", END)
    return g