from utils.llm_client import call_llm

response = call_llm(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
)

print(response.choices[0].message.content)