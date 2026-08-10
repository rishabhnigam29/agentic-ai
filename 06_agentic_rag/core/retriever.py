"""Retrieval layer - semantic search over the indexed policy"""

from __future__ import annotations

from langchain_core.documents import Document

from core.ingest import get_vectorstore

def format_chunks_as_context(chunks: list[Document],
                             seperator: str = "\n\n------\n\n") -> str:
    """Format a list of retrieved chunks into a single string for LLM context"""
    return seperator.join(
        f"[chunk {c.metadata.get('chunk_index', '?')}]:\n{c.page_content}" 
        for c in chunks
    )

def search(query: str, top_k: int = 5, where: dict | None = None) -> list[Document]:
    """Top-k semantic search, optionally filtered by metadata"""
    results = get_vectorstore().similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter=where or None
    )

    documents:list[Document] = []
    for doc, score in results:
        doc.metadata["score"] = score
        documents.append(doc)
    return documents

if __name__ == "__main__":
    query = "What is the maternity waiting period?"
    print(f"Query: {query}\n")
    for doc in search(query, top_k=3):
        meta= doc.metadata
        print(f"Score: {meta.get('score'):.4f}, "
              f"Chunk Index: {meta.get('chunk_index')}, "
              f"Section: {meta.get('section', 'N/A')}\n")

        print(f"{doc.page_content}\n")
        print("-" * 80 + "\n")
