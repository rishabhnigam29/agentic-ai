"""Stage C: Agentic RAG - the agent decides how to answer."""

from utils.tracing import setup_tracing

from agent.graph import compile_graph
from stages.helpers import ensure_ingested
from shared.profiles import get_active_profile

ROUTE_LABELS = {
    "docs": "Search the document (targeted retrieval)",
    "memory": "Check conversation memory",
    "direct": "Direct answer (no retrieval)",
    "full_scan": "Full scan of all documents (loading ALL chunks for comprehensive analysis)",
}

def run_agent(app, query:str, thread_id: str) -> str:
    """Run the LangGraph agent and print the decision path"""
    result = app.invoke({
        "query": query, "route": "", "rewritten_query": "", "metadata_filter": None, 
        "retrieved_chunks" : [], "chunks_relevant": False, "retrieval_attempts": 0,
        "memory_context": [],
        "answer": "", "analysis_result": None,
    },
    config={"configurable": {"thread_id": thread_id}}
    )

    route = result.get("route", "unknown")
    rewritten = result.get("rewritten_query", "")
    attempts = result.get("retrieval_attempts", 0)
    num_chunks = len(result.get("retrieved_chunks", []))
    num_memory = len(result.get("memory_context", []))
    analysis = result.get("analysis_result", None)
    mf = result.get("metadata_filter", None)

    print("\n Agent Decision Path:")
    print(f"  Route: {route} ({ROUTE_LABELS.get(route, 'Unknown')})")
    if route == "full_scan":
        print(f"  Scanned: {num_chunks} chunks (Full scan of all documents)")
        if analysis:
            print(f"  Found: {len(analysis.findings)} findings, risk={analysis.overall_rating}")

    else:
        if mf and mf.sections:
            print(f"  Metadata Filter: sections={mf.sections}")
        if rewritten:
            print(f"  Rewritten Query: {rewritten}")
        if attempts > 0:
            print(f"  Retrieval: {num_chunks} chunks in {attempts} attempts, relevant={result.get('chunks_relevant', False)}")

    if num_memory > 0:
        print(f"  Memory: {num_memory} relevant context items retrieved")

    return result["answer"]

def main() -> None:
    """Run the agentic RAG stage."""
    setup_tracing("M6 : agentic_rag")
    profile = get_active_profile()
    print("Stage C: Agentic RAG - Automated Deep Reader",
                 f"Document: {profile.label}",
                 "The agent decides how to answer your query.",
                 "Type 'quit' to exit, 'clear' to clear memory")

    ensure_ingested()

    app = compile_graph()

    session = 1

    turn_counter = 1
    while True:
        user_input = input(f"\nTurn {turn_counter} - Enter your query: ")
        if user_input.lower() == "quit":
            print("Exiting the agentic RAG stage.")
            break
        elif user_input.lower() == "clear":
            session += 1
            print("Memory cleared (started a new thread).")
            continue

        print(f"\nProcessing your query....")    
        answer = run_agent(app, user_input, thread_id=f"cli-session-{session}")
        print(f"\nAgent's Answer:\n{answer}")

if __name__ == "__main__":
    main()