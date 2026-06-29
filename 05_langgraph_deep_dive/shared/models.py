"""Pydantic models for structured LLM output."""

from pydantic import BaseModel, Field, field_validator

class BranchAnalysis(BaseModel):
    """Structured output for branch analysis."""
    branch: str = Field(description="Branch name: financial, market, team or visuals")
    score: float = Field(ge=0, le=100, description="Score from 0(terrible) to 100(excellent). Example : 70 ")
    summary: str = Field(description="2-3 sentences summary of findings")
    key_findings: list[str] = Field(description="List of notable positive findings")
    risks: list[str] = Field(description="List of notable risks or concerns or red flags")

    @field_validator("key_findings", "risks")
    @classmethod
    def at_least_one(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("At least one item is required")
        return v

class DueDiligenceReport(BaseModel):
    """Final structured due diligence report."""

    startup_name: str = Field(description="Name of the startup being evaluated")
    overall_score: float = Field(ge=0, le=100, description="Overall score from 0(terrible) to 100(excellent). Example : 70 ")
    recommendation: str = Field(description="APPROVE, REJECT or REVIEW")
    executive_summary: str = Field(description="2-3 sentences summary of the overall findings")
    branch_summaries: dict[str, str] = Field(description="Per-branch summaries, keyed by branch name")
    top_strengths: list[str] = Field(description="Top 3 strengths of the startup")
    top_risks: list[str] = Field(description="Top 3 risks or concerns of the startup")
    final_verdict: str = Field(description="One paragraph final investment verdict")

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, v: str) -> str:
        allowed = {"APPROVE", "REJECT", "REVIEW"}
        if v.upper() not in allowed:
            raise ValueError(f"recommendation must be one of {allowed}")
        return v.upper()
