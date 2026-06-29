"""Team analysis subgraph - extracts team information then scores them"""

from __future__ import annotations
from langgraph.graph import StateGraph

from agent.state import TeamSubState
from shared.prompts import TEAM_EXTRACT_PROMPT, TEAM_ANALYZE_PROMPT
from subgraphs.base import build_extract_analyze_subgraph

def build_team_subgraph() -> StateGraph:
    """Builds the team analysis subgraph."""
    return build_extract_analyze_subgraph(
        branch_name = "team",
        state_cls = TeamSubState,
        extract_prompt = TEAM_EXTRACT_PROMPT,
        analyze_prompt = TEAM_ANALYZE_PROMPT,
    )