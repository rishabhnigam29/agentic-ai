"""Structured-output schemas for the full document audit.

The audit is *pluggable*. Every document type defines its own concrete schema 
by subclassing 'AuditResult'/'AuditFinding'"""

from __future__ import annotations

from enum import Enum
from operator import le
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

class AuditFinding(BaseModel):
    """One issue found in the document"""

    location: str = Field(description="Where in the document this sits (e.g. 'Section 4.2', 'Clause 11')")
    category: str = Field(description="The kind of finding - each document type defines its own categories")
    description: str = Field(description="What the finding is and why it's a problem for the reader")
    severity: int = Field(ge=1, le=10, description="Severity from 1(minor) to 10(critical/poterntially illegal)")
    recommendation: str = Field(description="What the reader should do or demand about it")

    @field_validator("description")
    @classmethod
    def _description_substantive(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("description must be at least 10 characters long")
        return v

class AuditResult(BaseModel):
    """The full-document audit"""

    overall_rating:str = Field(description="Overall risk rating: Low, Medium, High, Critical")
    summary: str = Field(description="A plain english summary of the document's biggest problems")
    findings: list[AuditFinding] = Field(description="Every issues identified in the document")

    title: ClassVar[str] = "DOCUMENT AUDIT"
    finding_noun: ClassVar[str] = "finding"

    @field_validator("overall_rating")
    @classmethod
    def _overall_rating_enum(cls, v: str) -> str:
        if v not in ["Low", "Medium", "High", "Critical"]:
            raise ValueError("overall_rating must be one of: Low, Medium, High, Critical")
        return v

    def render(self) -> str:
        """Human readable rendering used by the CLI runned and the UI"""
        lines = [
            self.title,
            f"Overall rating: {self.overall_rating}",
            f"Summary: {self.summary}",
            "",
            f"Found {len(self.findings)} {self.finding_noun}(s):",
            "=" * 50,
        ]

        for i, f in enumerate(self.findings, 1):
            category = getattr(f.category, "value", f.category)
            lines += [
                "",
                f"{self.finding_noun.capitalize()} #{i}:",
                f"[{category}] {f.location} - Severity {f.severity}/10",
                f"{f.description}",
                f"Recommendation: {f.recommendation}",
            ]

        return "\n".join(lines)

class FlawType(str, Enum):
    """The kind of flaw found in the document"""
    GAP = "Gap"
    EXCLUSION = "Exclusion"
    AMBIGUITY = "Ambiguity"
    LIMITATION = "Limitation"

class PolicyFlaw(AuditFinding):
    category: FlawType = Field(description="Category of the policy flaw")

class AnalysisResult(AuditResult):
    title: ClassVar[str] = "COMPREHENSIVE POLICY ANALYSIS"
    finding_noun: ClassVar[str] = "flaw"

    findings: list[PolicyFlaw] = Field(description="Every flaw found in the policy document")