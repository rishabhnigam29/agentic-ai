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