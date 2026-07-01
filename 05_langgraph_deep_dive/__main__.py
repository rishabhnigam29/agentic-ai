"""Entry point for the duediligence graph."""

from __future__ import annotations

from pathlib import Path
from agent.graph import build_graph
from pypdf import PdfReader
import pypdfium2 as pdfium

from langgraph.checkpoint.memory import MemorySaver

from utils.tracing import setup_tracing

import base64
import io

SAMPLE_DECK = str(Path(__file__).resolve().parent / "data" / "sample_deck.pdf")
DECK_IMAGE_SCALE = 1.3

def _render_decl_images(path: str, scale: float = DECK_IMAGE_SCALE) -> list[str]:
    """Render each PDF page to a base64-encoded PNG image for vision branches."""
    images: list[str] = []
    for page in pdfium.PdfDocument(path):
        pil_image = page.render(scale=scale).to_pil()
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        images.append(base64.b64encode(buffered.getvalue()).decode("ascii"))
    return images

def _load_deck(path: str) -> str:
    """Pull the deck text layer or the text branches"""

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise SystemExit(f"Pictg deck has no extractable text: {path}")
    return text

def main() -> None:
    deck_text = _load_deck(SAMPLE_DECK)
    deck_images = _render_decl_images(SAMPLE_DECK)
    checkpointer = MemorySaver()
    app = build_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"])

    thread_config = {"configurable" : {"thread_id" : "duediligence_thread"}}

    initial_state = {
        "startup_name":"",
        "pitch_deck" : deck_text,
        "pitch_deck_images": deck_images,
        "branch_results": [],
        "overall_score": 0.0,
        "recommendation": "",
        "human_decision": "",
        "human_feedback": "",
        "final_report": "",
    }

    result = app.invoke(initial_state, config=thread_config)

    snapshot = app.get_state(thread_config)
    pending_tasks = snapshot.next

    if "human_review" in pending_tasks:
        # Graph Paused - show branch results so far and ask for human input
        print("Branch results so far:")
        for r in sorted(result["branch_results"], key=lambda x: x["branch"]):
            print(f"  - {r['branch']:>10s} : score={r['score']}/100, summary={r['summary'][:50]}")
        print(f"\nOverall weighted score: {result['overall_score']}/100")

        #Collect human input
        decision = input("Your decision (approve/reject): ").strip().lower()
        feedback = input("Your feedback (optional): ").strip()

        #inject human input into the graph state and resume
        app.update_state(thread_config, {"human_decision": decision, "human_feedback": feedback})

        result = app.invoke(None, config=thread_config)

    print(result["final_report"])

    #state_history = list(app.get_state_history(thread_config))
    #print(f"\n[debug] {len(state_history)} checkpoints saved in memory:")
    #for s in state_history:
    #    print(f"  - next = {s.next}, value = {list(s.values.keys())[:3]}")

if __name__ == "__main__":
    setup_tracing("M5: Startup Due Diligence")
    main()
