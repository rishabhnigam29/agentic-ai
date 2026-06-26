"""State types for the due deligence graph"""

from __future__ import annotations
from typing_extensions import TypedDict

class DueDiligenceState(TypedDict):
    startup_name: str
    pitch_deck: str