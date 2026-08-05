"""Langgraph node functions and routing edge helpers."""

from __future__ import annotations

from utils.llm_client import call_llm
from shared.prompts import ROUTER_PROMPT, REWRITER_PROMPT, RELEVANCE_GRADER_PROMPT, METADATA_FILTER_PROMPT
from agent.state import RouteDecision, AgentState, RelevanceGrade, MetadataFilter

from core.retriever import ChromaRetriever
from shared import config

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever

def router_node(state: AgentState) -> dict:
    """Classify whether the query needs doc retrieval, memory or direct answer."""
    history_snippet = ""
    if state.get("conversation_history"):
        last_turns = state["conversation_history"][-6:]  # Get the last 6 turns of conversation history
        history_snippet = "\n".join(
            f"{m['role']}: {m['content']}" for m in last_turns
        )

    decision: RouteDecision = call_llm(
        messages=[
            {
                "role": "system",
                "content": ROUTER_PROMPT,
            },
            {
                "role": "user",
                "content": (f"Conversation so far:\n{history_snippet}\n\n"
                            f"New query: {state['query']}\n\n")
            },
        ],
        response_model=RouteDecision,
        temperature=0.0,
    )
    return {"route": decision.route}

def rewrite_node(state: AgentState) -> dict:
    """Rewrite the query into a concise, keyword-rich search query."""
    rewritten: str = call_llm(
        messages=[
            {
                "role": "system",
                "content": REWRITER_PROMPT,
            },
            {
                "role": "user",
                "content": state["query"],
            },
        ],
        temperature=0.0,
    )
    return {"rewritten_query": rewritten}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks from the document store"""
    retriever = _get_retriever()
    search_query = state.get("rewritten_query") or state["query"]
    chunks = retriever.search(search_query, top_k=5)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {"retrieved_chunks": chunks, "retrieval_attempts": attempts}

def evaluate_node(state: AgentState) -> dict:
    """LLM judges whether the retrieved chunks actually answer the query."""
    chunks_text = "\n\n------\n\n".join(c.text for c in state["retrieved_chunks"])

    grade: RelevanceGrade = call_llm(
        messages=[
            {
                "role": "system",
                "content": RELEVANCE_GRADER_PROMPT,
            },
            {
                "role": "user",
                "content": (f"Query: {state['query']}\n\n"
                            f"Retrieved Chunks:\n{chunks_text}\n\n")
            },
        ],
        response_model=RelevanceGrade,
        temperature=0.0,
    )
    return {"chunks_relevant": grade.relevant}

def route_after_evalaute(state: AgentState) -> dict:
    if state["chunks_relevant"]:
        return "generate"
    if state.get("retrieval_attempts", 0) < config.MAX_RETRIEVAL_ATTEMPTS:
        return "generate"
    return "rewrite"  # If chunks are not relevant and max attempts reached, rewrite the query

def filter_node(state: AgentState)-> dict:
    """Decide whether to narrow retrieval to a specific section before searching."""
    prompt = METADATA_FILTER_PROMPT.format(
        sections="\n".join(f"- {s}" for s in config.SECTIONS))

    decision: MetadataFilter = call_llm(
        messages=[{"role": "system","content": prompt,},
            {"role": "user", "content": state["query"]}],
            response_model=MetadataFilter,
            temperature=0.0,
    )

    #Guard against a hallucinated section - only keep real sections or None
    if decision.section and decision.section not in config.SECTIONS:
        decision.section = None

    return {"metadata_filter": decision}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve chunks from the document store, narrowed by the section filter."""
    retriever = _get_retriever()
    search_query = state.get("rewritten_query") or state["query"]

    mfilter = state.get("metadata_filter")
    where_clause = {"section": mfilter} if mfilter and mfilter.section else None

    chunks = retriever.search(search_query, top_k=5, where=where_clause)
    if where_clause and not chunks:
        # Filter was too narrow - fall back to searching everything
        chunks = retriever.search(search_query, top_k=5, where=None)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {"retrieved_chunks": chunks, "retrieval_attempts": attempts}
