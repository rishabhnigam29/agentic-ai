"""Agent state types and routing/structured-output models."""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from core.retriever import RetrievedChunk

class AgentState(TypedDict):
    query: str
    route: str
    rewritten_query: str
    retrieved_chunks: list[RetrievedChunk]
    chunks_relevant: bool
    retrieval_attempts: int
    memory_context: str
    conversation_history: list[dict]
    answer: str

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