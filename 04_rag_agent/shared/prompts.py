"""Reusable system prompts for RAG agents."""

SUMMARIZE_PROMPT = """
Summarize the following conversation in 2-3 concise sentences, 
Focus on key facts, policy sections discussed, and any flaws or 
limits the assistant identified in the policy.
Write it as a self-contained note that would help recall the exchange later.
"""