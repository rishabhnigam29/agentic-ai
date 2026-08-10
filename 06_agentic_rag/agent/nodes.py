"""Langgraph node functions and routing edge helpers."""

from __future__ import annotations

from utils.llm_client import call_llm
from shared.prompts import ROUTER_PROMPT, REWRITER_PROMPT, RELEVANCE_GRADER_PROMPT, METADATA_FILTER_PROMPT, FULL_SCAN_PROMPT
from agent.state import RouteDecision, AgentState, RelevanceGrade, MetadataFilter

from core.retriever import search, format_chunks_as_context
from shared import config
from shared.models import AnalysisResult
from shared.profiles import get_active_profile

from core.ingest import get_vectorstore
from core.memory import ConversationMemory
from shared.profiles import get_active_profile

from langchain_core.documents import Document

_memory: ConversationMemory | None = None

def _get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory

def set_memory(memory: ConversationMemory) -> None:
    global _memory
    _memory = memory

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
    """Rewrite the query into a standalone, concise, keyword-rich search query."""
    history = state.get("conversation_history",[])
    history_snippet = "\n".join(
        f"{m['role']}:{m['content']}" for m in history[-6:]
    )
    rewritten: str = call_llm(
        messages=[
            {
                "role": "system",
                "content": REWRITER_PROMPT,
            },
            {
                "role": "user",
                "content": (f"Conversation so far:\n{history_snippet or '(nothing yet)'}\n\n"
                            f"Question to rewrite : {state["query"]}"),
            },
        ],
        temperature=0.0,
    )
    return {"rewritten_query": rewritten.strip()}


def evaluate_node(state: AgentState) -> dict:
    """LLM judges whether the retrieved chunks actually answer the query."""
    chunks_text = "\n\n------\n\n".join(c.page_content for c in state["retrieved_chunks"])

    graded_query = state.get("rewritten_query") or state["query"]
    grade: RelevanceGrade = call_llm(
        messages=[
            {
                "role": "system",
                "content": RELEVANCE_GRADER_PROMPT,
            },
            {
                "role": "user",
                "content": (f"Query: {graded_query}\n\n"
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
    if state.get("retrieval_attempts", 0) >= config.MAX_RETRIEVAL_ATTEMPTS:
        return "generate"
    return "rewrite"  # If chunks are not relevant and max attempts reached, rewrite the query

def filter_node(state: AgentState)-> dict:
    """Decide whether to narrow retrieval to a specific section before searching."""
    profile = get_active_profile()
    prompt = METADATA_FILTER_PROMPT.format(
        sections="\n".join(f"- {s}" for s in profile.sections))

    decision: MetadataFilter = call_llm(
        messages=[{"role": "system","content": prompt,},
            {"role": "user", "content": state.get("rewritten_query") or state["query"]}],
            response_model=MetadataFilter,
            temperature=0.0,
    )

    decision.sections = [s for s in decision.sections if s in profile.sections][:2]
    return {"metadata_filter": decision}

def retrieve_node(state: AgentState) -> dict:
    """Retrieve chunks from the document store, narrowed by the section filter."""
    search_query = state.get("rewritten_query") or state["query"]

    mfilter = state.get("metadata_filter")
    sections = mfilter.sections if mfilter else []

    if len(sections) > 1:
        where_clause = {"section": {"$in": sections}}
    elif sections:
        where_clause = {"section": sections[0]}
    else:
        where_clause =  None


    chunks = search(search_query, top_k=5, where=where_clause)
    if where_clause and not chunks:
        # Filter was too narrow - fall back to searching everything
        chunks = search(search_query, top_k=5, where=None)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {"retrieved_chunks": chunks, "retrieval_attempts": attempts}

def full_scan_node(state: AgentState) -> dict:
    """Load ALL chunks so the whole document can be analyzed. """
    data = get_vectorstore().get()
    
    chunks = [Document(text=doc,  score=1.0, metadata=dict(meta)) 
              for doc, meta in zip(data["documents"], data["metadatas"])]
    chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
    return {"retrieved_chunks": chunks}

def full_scan_generate_node(state: AgentState) -> dict:
    """Audit the ENTIRE document and return a structured report of all issues found."""
    all_text = format_chunks_as_context(state["retrieved_chunks"])
    profile = get_active_profile()

    analysis =  call_llm(
        messages=[{"role": "system","content": profile.full_scan_prompt,},
                  {"role": "user", "content": f"Here is complete document text:\n{all_text}"}],
        response_model=AnalysisResult,
        temperature=0.0,
    )

    return {"answer": analysis.render(), "analysis_report": analysis}

def full_scan_generate_node(state: AgentState) -> dict:
    """Audit the ENTIRE document and return a structured report of all issues found."""
    profile = config.get_active_profile()
    all_text = format_chunks_as_context(state["retrieved_chunks"])

    analysis =  call_llm(
        messages=[{"role": "system","content": profile.full_scan_prompt,},
                  {"role": "user", "content": f"Here is complete document text:\n{all_text}"}],
        response_model=profile.analysis_model,
        temperature=0.0,
    )
    return {"answer": analysis.render(), "analysis_report": analysis}

def memory_lookup_node(state: AgentState) -> dict:
    """Pull relevant past conversation context from long term memory."""
    memory = _get_memory()
    relevant_context = memory.get_relevant_memory(state["query"],top_k=3)
    return {"memory_context": relevant_context}

def generate_node(state: AgentState) -> dict:
    """Produce the final answer using all available context"""
    profile = get_active_profile()
    context_parts: list[str] = []

    if state.get("retrieved_chunks"):
        doc_context = "\n\n".join(
            f"[Chunk {c.metadata.get('chunk_index', '?')}] (relevance {c.metadata.get('score',1.0):.2f}): \n{c.page_content}" 
            for c in state["retrieved_chunks"]
        )
        context_parts.append(f"Document Excerts:\n{doc_context}")

    if state.get("memory_context"):
        memory_context = "\n\n".join(f"- {m}" for m in state["memory_context"])
        context_parts.append(f"Relevant Past Conversation:\n{memory_context}")

    combined_context = "\n\n".join(context_parts) if context_parts else "No additional context available."
    history = state.get("conversation_history", [])

    messages = [{"role": "system", "content": f"{profile.analyst_prompt}\n\n CONTEXT: \n{combined_context}"},
                *history[-6:],
                {"role": "user", "content": state["query"]}]
    answer: str = call_llm(messages=messages, temperature=0.0)
    return {"answer": answer}

def route_after_router(state: AgentState) -> dict:
    return state["route"]