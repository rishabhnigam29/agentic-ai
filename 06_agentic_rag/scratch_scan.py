from agent.nodes import full_scan_generate_node, full_scan_node
from stages.helpers import ensure_ingested

def main() -> None:
    ensure_ingested()

    state = {
        "query" : "Analyse this entire document and find every problem.",
        "retrieved_chunks":[],
    }

    print("Loading every chunks...")
    state.update(full_scan_node(state))

    print(f"Scanned {len(state['retrieved_chunks'])} chunks. Auditing")
    state.update(full_scan_generate_node(state))
    print(state["answer"])

if __name__ == "__main__":
    main()
