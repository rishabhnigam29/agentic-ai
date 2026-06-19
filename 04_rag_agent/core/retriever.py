"""Retrieval layer - semantic search over the indexed policy"""

from dataclasses import dataclass

from litellm import embedding

from shared import config
from core.ingest import get_chroma_client, get_or_create_collection

@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict

    def __str__(self) -> str:
        idx = self.metadata.get("chunk_index", "?")
        return f"[chunk {idx} | score={self.score:.3f}] \n {self.text}"
    
class ChromaRetriever:
    """Retriever for semantic search over the indexed policy"""

    def __init__(self) -> None:
        client = get_chroma_client()
        self.collection = get_or_create_collection(client)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = embedding(model=config.EMBEDDING_MODEL, input=[query]).data[0]["embedding"]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
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