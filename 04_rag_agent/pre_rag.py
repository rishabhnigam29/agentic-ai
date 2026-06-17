from utils.llm_client import call_llm

POLICY_QUESTION = """
I have an insurance policy from Apex Health Insurance Corporation,
policy number is AHI-2025-MED-47832. What is the maternity waitig period
and what are the listed exclusions for the fertility treatment?
"""

def main() -> None:
    answer = call_llm(
        messages=[{"role" : "user", "content": POLICY_QUESTION}],
        temperature=0,
    )
    print("Question : ")
    print(f"{POLICY_QUESTION}\n")
    print("ANSWER from stock LLM (no RAG):")
    print(f"{answer}")

if __name__ == "__main__":
    main()