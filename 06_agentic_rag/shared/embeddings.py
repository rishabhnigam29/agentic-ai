"""LiteLLM embeddings behind LangChain's Embeddings interface."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from litellm import embedding

from shared import config

class LiteLLMEmbeddings(Embeddings):
    """Adapter: LangChain call this; LiteLLM does the actual work."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = embedding(model=config.EMBEDDING_MODEL, input=texts)
        return [item["embedding"] for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]