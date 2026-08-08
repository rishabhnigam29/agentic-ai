"""Document ingestion pipeline - load, chunk, embed, store in ChromaDB."""

from pathlib import Path
import chromadb
from chromadb.api import ClientAPI
from litellm import embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import config
from shared.prompts import SECTION_TAG_PROMPT
from shared.profiles import get_active_profile

from utils.llm_client import call_llm

def load_document(file_path: str) -> str:
    """Load a document from a file path."""
    path = Path(file_path)
    if path.suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")

def chunk_text(text: str, 
               chunk_size: int = 1000,
               chunk_overlap: int = 200) -> list[str]:
    """Chunk text into smaller pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", "", "."],
    )
    return splitter.split_text(text)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts."""
    response = embedding(model=config.EMBEDDING_MODEL, input=texts)
    return [item["embedding"] for item in response.data]

def get_chroma_client() -> ClientAPI:
    """Embedded (on-disk) for local dev; HttpClient against a server when
    CHROMA_HOST and CHROMA_PORT are set."""
    if config.CHROMA_HOST:
        return chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

def get_or_create_collection(client: ClientAPI) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(name=config.COLLECTION_NAME,
                                        metadata={"hnsw:space": "cosine"},
                                            )

def ingest_document(file_path: str) -> int:
    print(f"Loading document from {file_path}...")
    text = load_document(file_path)
    print(f"Document length: {len(text)} characters.")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    vectors = get_embeddings(chunks)

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    existing = collection.count()
    if existing > 0:
        print(f" Clearing {existing} existing chunks (re-ingest)")
        collection.delete(ids=collection.get()["ids"])

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Tagging each chunk with its section...")
    sections = [tag_chunk_section(chunk, get_active_profile().sections) for chunk in chunks]

    metadata = [{"source": file_path , 
                 "embedding_model": config.EMBEDDING_MODEL,
                 "section": sections[i],
                 "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadata,
    )
    print(f" Stored {len(chunks)} chunks in ChromaDB collection '{config.COLLECTION_NAME}'.")
    return len(chunks)

def tag_chunk_section(text: str, sections: list[str]) -> str:
    """Classify one chunk into exactly one of 'sections' (cheap LLM call)"""
    prompt = SECTION_TAG_PROMPT.format(sections="\n".join(f"- {s}" for s in sections))
    try:
        raw = call_llm(
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": text[:1500]}],
            temperature=0,
        ).strip()
    except Exception as e:
        print(f"Error occurred while tagging chunk section: {e}")
        return "Other"

    for s in sections:
        if raw.lower() == s.lower():
            return s 

    for s in sections:
        if s.lower() in raw.lower() or raw.lower() in s.lower():
            return s 

    return "Other" 

if __name__ == "__main__":
    # Example usage
    num_chunks = ingest_document(config.SAMPLE_POLICY)
    print(f"Ingested {num_chunks} chunks from sample policy document.")
