"""Stage A - Basic RAG: question-> retrieve-> augment prompt-> answer"""

from core.retriever import ChromaRetriever, RetrievedChunk
from utils.llm_client import call_llm
from utils.tracing import setup_tracing, traceable
from stages.helpers import ensure_ingested, format_sources
from shared.prompts import ANALYST_STAGE_A_PROMPT

@traceable(NAME="RAG Retrieval")
def retrieve(retriever: ChromaRetriever, question: str, top_k: int = 5) -> list[RetrievedChunk]:
   """Child Span : embed the question and pull the top-k chunks from chromaDb"""
   return retriever.search(question, top_k=top_k)

@traceable(NAME="RAG Generation")
def generate(question: str, context: str)-> str:
    """Child Span : call the LLM with the question and retrieved context to generate an answer"""
    return call_llm(
        messages=[
            {"role": "system", "content": f"{ANALYST_STAGE_A_PROMPT}\n\nDocument EXCERTS:\n{context}"},
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

