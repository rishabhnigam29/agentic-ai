"""Stage A - Basic RAG: question-> retrieve-> augment prompt-> answer"""

from core.ingest import get_chroma_client, get_or_create_collection, ingest_document
from core.retriever import ChromaRetriever, RetrievedChunk
from shared import config
from utils.llm_client import call_llm
from utils.tracing import setup_tracing, traceable

ANALYST_PROMPT = """
You are a medical policy analyst. Answer the user's question using ONLY the document
excerpts below. If the excerts don't contain enough information, say no.
Cite the chunk index for any claim you make.
"""

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

@traceable(NAME="RAG Retrieval")
def retrieve(retriever: ChromaRetriever, question: str, top_k: int = 5) -> list[RetrievedChunk]:
   """Child Span : embed the question and pull the top-k chunks from chromaDb"""
   return retriever.search(question, top_k=top_k)

@traceable(NAME="RAG Generation")
def generate(question: str, context: str)-> str:
    """Child Span : call the LLM with the question and retrieved context to generate an answer"""
    return call_llm(
        messages=[
            {"role": "system", "content": f"{ANALYST_PROMPT}\n\nDocument EXCERTS:\n{context}"},
            {"role": "user", "content": f"Question: {question}"},
        ],
        temperature=0.0,
    )

@traceable(NAME="RAG Query")
def rag_query(retriever: ChromaRetriever, question: str) -> str:
    chunks = retrieve(retriever, question)
    
    context = "\n\n ----- \n\n".join(
        f"[Chunk {c.metadata.get('chunk_index', '?')}]:\n{c.text}"
        for c in chunks
    )

    answer: str = generate(question, context)

    print("\nSources used:")
    print(format_sources(chunks))
    return answer

def main() -> None:
    setup_tracing("module_04-RAG_Agent")
    print("="*60)
    print("Stage A - Basic RAG - Medical Policy Analysis")
    print(" Type 'exit' to quit.")
    print("="*60)

    ensure_ingested()
    retriever = ChromaRetriever()

    while True:
        try:
            question = input("\n You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print("\n Retrieving relevant information...")
        answer = rag_query(retriever, question)
        print("\n Assistant:", answer)

if __name__ == "__main__":
    main()

