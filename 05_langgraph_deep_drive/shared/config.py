"""Centralised constants and thresholds for the due diligence agent."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent

#---Score thresholds for routing--------------------
AUTO_APPROVE_THRESHOLD = 70.0
AUTO_REJECT_THRESHOLD = 40.0

#---Branch weights (must sum to 1.0)--------------------
BRANCH_WEIGHTS = {
    "financial": 0.3,
    "market": 0.25,
    "team": 0.25,
    "visuals": 0.2
}

#---Sample data------------------------------
# A real startup hands you a one pdf with all the slides. We pull its text layer
# out for the text branches, and later render its pages to images
# for vision branch - one artifact, two representations. 
SAMPLE_DECK = str(PACKAGE_DIR / "data" / "sample_deck.pdf")