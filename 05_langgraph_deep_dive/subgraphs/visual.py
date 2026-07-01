"""Visual / multimodal analysis subgraph - uses vision LLM for charts"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import VisualSubState
from shared.models import BranchAnalysis
from shared.prompts import VISUAL_EXTRACT_PROMPT, VISUAL_ANALYZE_PROMPT
from utils.llm_client import call_llm

def _build_visual_content(text: str, images_b64: list[str]) -> list[dict]:
    """Build multimodal messages conetnt with text and base 64-encoded images."""
    parts: list[dict] = [{"type": "text", "text": text}]
    for image in images_b64:
        mime = "image/png" 
        if image.startswith("/9j/"):  # Check for JPEG magic number
            mime = "image/jpeg"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})
    return parts

def extract_visual(state: VisualSubState) -> dict:
    images = state.get("pitch_deck_images", [])

    if images:
        content = _build_visual_content(
            f"Analyze the visual elements of this pitch deck:\n\n"
            f"Pitch deck text for context:\n{state['pitch_deck']}",
            images
        )
        messages = [
            {"role": "system", "content": VISUAL_EXTRACT_PROMPT},
            {"role": "user", "content": content}
        ]
    else:
        messages = [
            {"role": "system", "content": VISUAL_EXTRACT_PROMPT},
            {"role": "user", "content": (
                "No images are provided. Evaluate the pitch deck's presentation quality"
                " based on its textual structure:\n\n"
                f"{state['pitch_deck']}"
            ),
            },
        ]

    extracted:str = call_llm(messages, temperature=0) 
    return {"extracted_info": extracted}

def analyze_visual(state: VisualSubState) -> dict:
    res: BranchAnalysis = call_llm(
        messages=[
            {"role": "system", "content": VISUAL_ANALYZE_PROMPT},
            {"role": "user", "content": 
             f"Visual analysis of the extracted information:\n\n{state['extracted_info']}"}
        ],
        response_model=BranchAnalysis,
        temperature=0
    )
    return {
        "score": res.score,
        "summary": res.summary,
        "key_findings": res.key_findings,
        "risks": res.risks
    }

def build_visual_subgraph() -> StateGraph:
    g = StateGraph(VisualSubState)
    g.add_node("extract", extract_visual)
    g.add_node("analyze", analyze_visual)

    g.set_entry_point("extract")

    g.add_edge("extract", "analyze")
    
    g.add_edge("analyze", END)
    return g