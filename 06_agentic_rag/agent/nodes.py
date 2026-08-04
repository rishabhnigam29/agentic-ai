"""Langgraph node functions and routing edge helpers."""

from __future__ import annotations

from utils.llm_client import call_llm
from shared.prompts import ROUTER_PROMPT, REWRITER_PROMPT, RELEVANCE_GRADER_PROMPT
from agent.state import RouteDecision, AgentState, RelevanceGrade

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