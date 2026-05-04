from litellm import completion

HARDCODED_KEY = "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"

def call_llm(messages, temperature=0):
    return completion(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        api_key=HARDCODED_KEY,
    )