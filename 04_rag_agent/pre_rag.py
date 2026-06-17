import os
import math

from litellm import embedding
from utils.llm_client import call_llm

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

POLICY_QUESTION = """
I have an insurance policy from Apex Health Insurance Corporation,
policy number is AHI-2025-MED-47832. What is the maternity waitig period
and what are the listed exclusions for the fertility treatment?
"""

def cosine(a: list[float], b: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def show_embedding_similarity() -> None:
    """Show the embedding similarity between the question and a sample policy document."""
    sentences = [
        "diabetes coverage",
        "blood sugar limits",
        "vacation in bali"
    ]

    print(f"Embedding model: {EMBEDDING_MODEL}")
    response = embedding(model = EMBEDDING_MODEL, input = sentences)
    vectors = [item["embedding"] for item in response.data]

    print(f"Vector length: {len(vectors[0])} dimensions\n")

    related = cosine(vectors[0], vectors[1])
    unrelated_a= cosine(vectors[0], vectors[2])
    unrelated_b = cosine(vectors[1], vectors[2])

    print("Cosine similarity")
    print(f" '{sentences[0]}' vs '{sentences[1]}': {related:.3f} (related)")
    print(f" '{sentences[0]}' vs '{sentences[2]}': {unrelated_a:.3f} (unrelated)")
    print(f" '{sentences[1]}' vs '{sentences[2]}': {unrelated_b:.3f} (unrelated)")

def show_failing_query() -> None:
    answer = call_llm(
        messages=[{"role" : "user", "content": POLICY_QUESTION}],
        temperature=0,
    )
    print("Question : ")
    print(f"{POLICY_QUESTION}\n")
    print("ANSWER from stock LLM (no RAG):")
    print(f"{answer}")

def main() -> None:
    print("=== 1. Failing query without RAG ===\n")
    show_failing_query()
    print("\n=== 2. Embedding similarity demonstration ===\n")
    show_embedding_similarity()

if __name__ == "__main__":
    main()