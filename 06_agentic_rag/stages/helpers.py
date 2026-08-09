"""Shared helpers for the stage runners"""

from core.ingest import get_vectorstore, ingest_document
from core.retriever import ChromaRetriever, RetrievedChunk
from shared import config
from shared.profiles import get_active_profile


def ensure_ingested()-> None:
    if not get_vectorstore().get(limit=1)["ids"]:
        profile = get_active_profile()
        print("First run - ingesting documents...")
        ingest_document(profile.sample_path)
        print("Documents ingested.")

def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(
        f" [{i}] chunk {c.metadata.get('chunk_index', '?')} (score: {c.score:.3f})"
        for i, c in enumerate(chunks, start=1)
    )

