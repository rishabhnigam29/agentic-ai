"""Shared helpers for the stage runners"""

from core.ingest import get_chroma_client, get_or_create_collection, ingest_document
from core.retriever import ChromaRetriever, RetrievedChunk
from shared import config


def ensure_ingested()-> None:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    if collection.count() == 0:
        print("First run - ingesting documents...")
        ingest_document(config.SAMPLE_POLICY)
        print("Documents ingested.")

def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(
        f" [{i}] chunk {c.metadata.get('chunk_index', '?')} (score: {c.score:.3f})"
        for i, c in enumerate(chunks, start=1)
    )

