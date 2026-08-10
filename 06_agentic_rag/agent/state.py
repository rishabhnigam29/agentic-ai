"""Agent state types and routing/structured-output models."""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from shared.models import AuditResult

from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    route: str
    rewritten_query: str
    retrieved_chunks: list[Document]
    chunks_relevant: bool
    retrieval_attempts: int
    memory_context: str
    conversation_history: list[dict]
    answer: str
    metadata_filter: MetadataFilter | None
    analysis_report: AuditResult | None

class RouteDecision(BaseModel):
    reasoning: str = Field(description="One sentence explaining the routing decision.")
    route: Literal["docs", "memory", "direct", "full_scan"] = Field(
        description=(
            "'docs' = needs a SPECIFIC lookup in the documents,"
            "'memory' = answerable from conversation history,"
            "'direct' = general question answerable without the document, "
            "'full_scan' = broad analysis that needs the ENTIRE document"
            "(e.g. 'analyze this', 'review the whole thing', 'find all problems')"
        )
    )

class RelevanceGrade(BaseModel):
    reasoning: str = Field(description="Why the chunks are or are not relevant to the query.")
    relevant: bool = Field(description="True if the retrieved chunks answer the query")

class MetadataFilter(BaseModel):
    reasoning: str = Field(description="Why this section(or None) was chosen")
    sections: list[str] = Field(default_factory=list, 
                                description="Atmost two exact section labels to filter on. Empty lust to search in whole document.")
