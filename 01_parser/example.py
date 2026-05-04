from utils.llm_client import call_llm

messy_email = """
Hey, I'm super frustrated. I bought your software last week and it's crashing
every time I open the dashboard. I want my money back ASAP! This is a disaster.
My name is John Doe, by the way. Fix this or refund me.
"""
response = call_llm(
    messages=[
        {"role": "system", "content": "Extract the sender's name and intent from this email.'"},
        {"role": "user", "content": messy_email},
    ],
)

text=response.choices[0].message.content
print("Raw LLM Response:")
print(text)

print("\nParsed Output:")

if "John Doe" in text:
    print("Found 'John Doe' in the response, extracting name as we got lucky")
else:
    print("Did not find 'John Doe' in the response, parsing for name")

print("\n This is the pain pydantic + Instructor will fix")    