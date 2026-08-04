"""Langgraph node functions and routing edge helpers."""

from __future__ import annotations

from utils.llm_client import call_llm
from shared.prompts import ROUTER_PROMPT
from agent.state import RouteDecision, AgentState

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