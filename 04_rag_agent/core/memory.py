"""Conversation memory management for RAG agents.
Short term sliding window of turns"""

import time
from dataclasses import dataclass, field

from litellm import embedding

from shared import config
from shared.prompts import SUMMARIZE_PROMPT
from core.ingest import get_chroma_client
from utils.llm_client import call_llm
@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

class ConversationMemory:
    """Short term conversation memory with a sliding window of turns."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._turns: list[Turn] = []
        self._summary_count = 0

        client = get_chroma_client()
        self.collection = client.get_or_create_collection(config.MEMORY_COLLECTION_NAME,
                                                          metadata={"hnsw:space": "cosine"},
                                                          )
    # short term memory management    
    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(Turn(role, content))

    def get_recent_context(self, k: int | None = None) -> list[dict[str, str]]:
        """Get the most recent k turns as context."""
        k = k or self.window_size
        recent = self._turns[-k:]
        return [{"role": turn.role, "content": turn.content} for turn in recent]
    
    @property
    def turn_count(self) -> int:
        return len(self._turns)
    
    # long term memory management (vector based)

    def summarize_and_store(self) -> str | None:
        """Summarize the conversation and store it in the vector database."""
        if len(self._turns) < config.SUMMARIZE_EVERY_N_TURNS:
            return None
        
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in self._turns)

        summary: str = call_llm(
            messages = [
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": transcript}
            ],
            temperature=0,
        )

        vec = embedding(model=config.EMBEDDING_MODEL, input=[summary]).data[0]["embedding"]
        doc_id = f"summary_{self._summary_count}"
        self._summary_count += 1

        self.collection.add(
            ids=[doc_id],
            embeddings=[vec],
            documents=[summary],
            metadatas=[{"type": "summary", "timestamp": time.time()}]
        )
        return summary
    
    def get_relevant_summaries(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve the most relevant summaries for a given query."""
        if self.collection.count() == 0:
            return []
        
        query_vec = embedding(model=config.EMBEDDING_MODEL, input=[query]).data[0]["embedding"]
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents"]
        )
        return results["documents"][0] if results["documents"] else []
    
    def clear_short_term_memory(self) -> None:
        """Clear the short term memory."""
        self._turns.clear()

    
if __name__ == "__main__":
    memory = ConversationMemory(window_size=4)

    memory.add_turn("user", "What is the maternity waiting period?")
    memory.add_turn("assistant", "Twelve months")
    memory.add_turn("user", "and what about limits?")
    memory.add_turn("assistant", "48 hour postpartum hospital stay")

    print("Short term context:")
    for msg in memory.get_recent_context():
        print(f"{msg['role']:>10}: {msg['content']}")

    summary = memory.summarize_and_store()
    print("\nSummary stored in vector database:")
    print(summary)

    print("\n Retrieving by similarity to 'tell me about pregnancy benifits':")
    for hit in memory.get_relevant_summaries("tell me about pregnancy benefits"):
        print(f"- {hit}")
