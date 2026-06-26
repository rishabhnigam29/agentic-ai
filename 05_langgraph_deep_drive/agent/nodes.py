""" Node functions for the due diligence graph """

from __future__ import annotations
from utils.llm_client import call_llm
from shared.prompts import INTAKE_PROMPT
from agent.state import DueDiligenceState

def intake_node(state: DueDiligenceState) -> dict:
    """Parse the pitch deck and extract the startup name."""
    name: str = call_llm(
        messages=[
            {"role": "system", "content": INTAKE_PROMPT}, 
            {"role": "user", "content": state["pitch_deck"][:3000]}
            ],
            temperature=0,
    )
    return {"startup_name": name.strip().strip('"').strip("'")}
