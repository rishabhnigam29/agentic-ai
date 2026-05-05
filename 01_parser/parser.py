from utils.llm_client import call_llm

def parse_email_to_crm(email_context,
                       response_model, temperature=0, max_retries=2):
    messages = [
        {
            "role": "system",
            "content": "You are a world-class CRM data entry clerk. Extract the structured data from emails",
        },
        {
            "role": "user",
            "content": f"Extract data from this email: {email_context}",
        },
    ]

    return call_llm(
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_retries=max_retries,
    )