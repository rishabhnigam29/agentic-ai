"""Entry point for the duediligence graph."""

from __future__ import annotations

from pathlib import Path
from agent.graph import build_graph
from pypdf import PdfReader

SAMPLE_DECK = str(Path(__file__).resolve().parent / "data" / "sample_deck.pdf")

def _load_deck(path: str) -> str:
    """Pull the deck text layer or the text branches"""

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise SystemExit(f"Pictg deck has no extractable text: {path}")
    return text

def main() -> None:
    deck_text = _load_deck(SAMPLE_DECK)

    app = build_graph().compile()

    initial_state = {
        "startup_name":"",
        "pitch_deck" : deck_text,
        "branch_results": [],
        "overall_score": 0.0,
        "recommendation": "",
        "human_decision": "",
        "human_feedback": "",
        "final_report": "",
    }

    result = app.invoke(initial_state)

    print(result["final_report"])

if __name__ == "__main__":
    main()
