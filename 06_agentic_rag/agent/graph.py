"""Langgraph graph construction and routing edge helpers."""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from agent.state import AgentState
from agent.nodes import (
    router_node,
    filter_node,
    rewrite_node,
    retrieve_node,
    evaluate_node,
    memory_lookup_node,
    generate_node,
    full_scan_node,
    full_scan_generate_node,
    route_after_router,
    route_after_evalaute,
)

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("filter", filter_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("memory_lookup", memory_lookup_node)
    graph.add_node("generate", generate_node)
    graph.add_node("full_scan_generate", full_scan_generate_node)
    graph.add_node("full_scan", full_scan_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", 
                               route_after_router,
                               {
                                   "docs": "rewrite",
                                    "memory": "memory_lookup",
                                    "direct": "generate",
                                    "full_scan": "full_scan",
                               },
    )

    graph.add_edge("rewrite", "filter",)
    graph.add_edge("filter", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges("evaluate", 
                               route_after_evalaute,
                               {
                                   "generate": "generate",
                                   "rewrite": "rewrite",
                               },
    )

    graph.add_edge("memory_lookup", "generate")
    graph.add_edge("full_scan", "full_scan_generate")
    graph.add_edge("generate", END)
    graph.add_edge("full_scan_generate", END)

    return graph

def compile_graph():
    return build_graph().compile(checkpointer=InMemorySaver())