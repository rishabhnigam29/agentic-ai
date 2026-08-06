"""Retrieval layer - semantic search over the indexed policy"""

from dataclasses import dataclass

from litellm import embedding

from shared import config
from core.ingest import get_chroma_client, get_or_create_collection

def format_chunks_as_context(chunks: list["RetrievedChunk"],
                             seperator: str = "\n\n------\n\n") -> str:
    """Format a list of retrieved chunks into a single string for LLM context"""
    return seperator.join(
        f"[chunk {c.metadata.get('chunk_index', '?')}]:\n{c.text}" 
        for c in chunks
    )

@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict

    def __str__(self) -> str:
        idx = self.metadata.get("chunk_index", "?")
        return f"[chunk {idx} | score={self.score:.3f}] \n {self.text}"
    
class EmptyIndexError(RuntimeError):
    """Raised when the collection has no chunks. Run ingest first."""

class EmbeddingModelMismatchError(RuntimeError):
    """Raised when chunks were embedded with a different model that the current one"""
    
class ChromaRetriever:
    """Retriever for semantic search over the indexed policy"""

    def __init__(self) -> None:
        client = get_chroma_client()
        self.collection = get_or_create_collection(client)
        self._verify_index()

    def _verify_index(self)->None:
        count = self.collection.count()
        if count == 0:
            raise EmptyIndexError("Collection is empty. Run ingest.py first")
        
        sample = self.collection.get(limit=1, include=["metadatas"])
        stored_model = (sample["metadatas"][0] or {}).get("embedding_model")

        if stored_model and stored_model != config.EMBEDDING_MODEL :
            raise EmbeddingModelMismatchError("Ingestion and Retrieval models are different")

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[RetrievedChunk]:
        query_vector = embedding(model=config.EMBEDDING_MODEL, input=[query]).data[0]["embedding"]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"]
        )

        chunks: list[RetrievedChunk] = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = 1 - (distance / 2)  # Convert distance to similarity score (0-1)
            chunks.append(RetrievedChunk(text=doc, score=score, metadata=metadata))
        return chunks
    
if __name__ == "__main__":
    retriever = ChromaRetriever()
    query = "What is the maternity waiting period?"
    print(f"Query: {query}\n")
    for chunk in retriever.search(query, top_k=3):
        print(chunk)
        print("-" * 80)