"""Stage B - RAG + Memory: Doc retrieval + short-term + long term memory"""

from core.memory import ConversationMemory
from core.retriever import ChromaRetriever
from utils.llm_client import call_llm
from utils.tracing import setup_tracing, traceable
from stages.helpers import ensure_ingested, format_sources
from shared.prompts import ANALYST_STAGE_B_PROMPT
from shared import config

@traceable(name="generate")
def generate(messages: list[dict]) -> str:
    """Child Span : call the LLM with the question and retrieved context to generate an answer"""
    return call_llm(
        messages=messages,
        temperature=0.0,
    )

@traceable(name="rag_query_with_memory")
def rag_query_with_memory(retriever: ChromaRetriever, memory: ConversationMemory, question: str) -> str:
    """Child Span : retrieve relevant chunks and generate an answer using the LLM with memory context"""
    # Retrieve relevant chunks from the document
    chunks = retriever.search(question, top_k=5)
    
    # Format the retrieved chunks for context
    context = "\n\n ----- \n\n".join(
        f"[Chunk {c.metadata.get('chunk_index', '?')}]:\n{c.text}"
        for c in chunks
    )

    # Get recent conversation history and summaries from memory
    long_term_memory = memory.get_relevant_memory(question, top_k=3)
    recent_history = memory.get_recent_context(k=4)

    parts = [f"DOCUMENT EXCERPTS:\n{context}"]
    if long_term_memory:
        parts.append(f"RELEVANT SUMMARIES FROM MEMORY:\n" + "\n\n".join(long_term_memory))

    system_prompt = f"{ANALYST_STAGE_B_PROMPT}\n\n" + "\n\n".join(parts)

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        *recent_history,
        {"role": "user", "content": f"Question: {question}"},
    ]

    answer: str = generate(messages)

    #uncomment below logs for debugging
    #print("\nSources used:")
    #print(format_sources(chunks))
    if long_term_memory:
        print(f" + {len(long_term_memory)} relevant summaries from memory")
    
    return answer

def main() -> None:
    setup_tracing("module_04-RAG_Agent")
    print("="*60)
    print("Stage B - RAG + Memory - Converational aware Medical Policy Analysis")
    print(" Type 'exit' to quit.")
    print("="*60)

    ensure_ingested()
    retriever = ChromaRetriever()
    memory = ConversationMemory(window_size=10)

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
        if question.lower == "clear":
            memory.clear_short_term_memory()
            print("Short term memory cleared")
            continue

        print("\n Retrieving relevant information...")
        answer = rag_query_with_memory(retriever, memory, question)
        print("\n Assistant:", answer)

        memory.add_turn("user", question)
        memory.add_turn("assistant", answer)

        if memory.turn_count and memory.turn_count % (config.SUMMARIZE_EVERY_N_TURNS *2 ) == 0:
            print("\n [Summarizing older turns to long term memory...]")
            summary = memory.summarize_and_store()
            if summary:
                print(f" Stored memory : {summary[:80]}...")
                memory.clear_short_term_memory()

if __name__ == "__main__":
    main()