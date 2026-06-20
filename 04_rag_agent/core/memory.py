"""Conversation memory management for RAG agents.
Short term sliding window of turns"""

import time
from dataclasses import dataclass, field

@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

class ConversationMemory:
    """Short term conversation memory with a sliding window of turns."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self.turns: list[Turn] = []
    
    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role, content))

    def get_recent_context(self, k: int | None = None) -> list[dict[str, str]]:
        """Get the most recent k turns as context."""
        k = k or self.window_size
        recent = self.turns[-k:]
        return [{"role": turn.role, "content": turn.content} for turn in recent]
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)
    
if __name__ == "__main__":
    memory = ConversationMemory(window_size=4)

    memory.add_turn("user", "What is the maternity waiting period?")
    memory.add_turn("assistant", "Twelve months")
    memory.add_turn("user", "and what about limits?")
    memory.add_turn("assistant", "48 hour postpartum hospital stay")

    for msg in memory.get_recent_context():
        print(f"{msg['role']:>10}: {msg['content']}")
