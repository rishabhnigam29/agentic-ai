"""Centralized constants and paths for the RAG agent."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]

#-------------------------------- ChromaDB ---------------------------------#
# Embedded by default (writes to local disk). 
# Set CHROMA_HOST and CHROMA_PORT to connect to a remote ChromaDB instance.
CHROMA_PERSIST_DIR = str(PACKAGE_DIR / "chroma_db")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "policy_docs"
MEMORY_COLLECTION_NAME = "conversation_memory"

#------------------------------- Embedding Model ------------------------------#
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

#------------------------------ Sample Policy Document ------------------------------#
SAMPLE_POLICY = str(PACKAGE_DIR / "data" / "sample_policy.pdf")

#------------------------------- Memory tunables ------------------------------#
SUMMARIZE_EVERY_N_TURNS = 3

#------------------------------- Retrieval tunables ------------------------------#
MAX_RETRIEVAL_ATTEMPTS = 2