from parser import parse_email_to_crm
from models import CRMEntry

messy_email = """
Hey, I'm super frustrated. I bought your software last week and it's crashing
every time I open the dashboard. I want my money back ASAP! This is a disaster.
My name is John Doe, by the way. Fix this or refund me.
"""

try:
    structured_data = parse_email_to_crm(messy_email, response_model=CRMEntry)
    print(structured_data.model_dump_json(indent=2))
except Exception as e:
    print(f"Validation Error: {e}")
