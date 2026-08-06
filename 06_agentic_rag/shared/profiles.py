"""Per document type profiles- the swappable "lens" the deep reader uses"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from pydantic import Field

from shared import config
from shared.models import AuditFinding, AuditResult, AnalysisResult
from shared.prompts import ANALYST_PROMPT, FULL_SCAN_PROMPT, LEASE_ANALYST_PROMPT, LEASE_FULL_SCAN_PROMPT

@dataclass(frozen=True)
class DocProfile:
    key: str
    label: str
    sample_path: str
    sections: list[str]
    analysis_model: type[AuditResult]
    analyst_prompt: str
    full_scan_prompt: str

    @property
    def collection_name(self) -> str:
        return f"{self.key}_docs"


POLICY_SECTIONS =[
    "Definations & General Provisions",
    "Eligibility & Enrollment",
    "Covered Services",
    "Exclusions & Limitations",
    "Claims & Appeals",
    "Premium & Cost-Sharing",
    "Miscellaneous Provisions",
    "Other"
]

POLICY_PROFILE = DocProfile(
    key="medical_policy",
    label="Medical Insurance Policy",
    sample_path=config.SAMPLE_POLICY,
    sections=POLICY_SECTIONS,
    analysis_model=AnalysisResult,
    analyst_prompt=ANALYST_PROMPT,
    full_scan_prompt=FULL_SCAN_PROMPT
)

class LeaseIssueType(str, Enum):
    UNFAIR = "Unfair"
    ONEROUS = "Onerous"
    AMBIGUOUS = "Ambiguous"
    NON_COMPLIANT = "Non-Compliant"

class UnfairClause(AuditFinding):
    category: LeaseIssueType = Field(description="Category of the problematic clause")

class LeaseAudit(AuditResult):
    title: ClassVar[str] = "RENTAL AGREEMENT AUDIT"
    findings_noun: ClassVar[str] = "clause"
    findings: list[UnfairClause] = Field(description="Every problematic clause found in the lease")

LEASE_SECTIONS = [
    "Parties and Premises",
    "Rent & Deposit",
    "Term & Renewal",
    "Maintenance & Repairs",
    "Use & Occupancy",
    "Entry & Privacy",
    "Termination & Eviction",
    "Liability & Insurance",
    "General Terms",
    "Other"
]

LEASE_PROFILE = DocProfile(
    key="rental_lease",
    label="Residential Rental / Lease Agreement",
    sample_path=config.SAMPLE_LEASE,
    sections=LEASE_SECTIONS,
    analysis_model=LeaseAudit,
    analyst_prompt=LEASE_ANALYST_PROMPT,
    full_scan_prompt=LEASE_FULL_SCAN_PROMPT
)

PROFILES: dict[str, DocProfile] = {
    POLICY_PROFILE.key: POLICY_PROFILE,
    LEASE_PROFILE.key: LEASE_PROFILE
}

def get_active_profile() -> DocProfile:
    """The profile selected by 'config.ACTIVE_PROFILE' (env 'DOC_PROFILE')"""
    key = config.ACTIVE_PROFILE
    if key not in PROFILES:
        raise ValueError(f"Unknown profile key: {key}. Available profiles: {list(PROFILES.keys())}")
    return PROFILES[key]
