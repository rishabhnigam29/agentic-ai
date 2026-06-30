"""State types for the due deligence graph"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

#------Main graph state-------------------------------------------
class DueDiligenceState(TypedDict):
    startup_name: str
    pitch_deck: str
    branch_results: Annotated[list[dict], operator.add]
    overall_score: float
    recommendation: str # "approve", "reject", or "review"
    human_decision: str # "approve", "reject", or "review"
    human_feedback: str
    final_report: str

#------Subgraph state (each branch has its own state)-------------------------------------------

class FinancialSubState(TypedDict):
    pitch_deck: str
    extracted_info: dict
    score: float
    summary: str
    key_findings: list[str]
    risks: list[str]

class MarketSubState(TypedDict):
    pitch_deck: str
    extracted_info: dict
    score: float
    summary: str
    key_findings: list[str]
    risks: list[str]

class TeamSubState(TypedDict):
    pitch_deck: str
    extracted_info: dict
    score: float
    summary: str
    key_findings: list[str]
    risks: list[str]