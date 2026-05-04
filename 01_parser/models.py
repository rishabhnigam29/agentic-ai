from enum import Enum
from Pydantic import BaseModel, Field, field_validator

class Intent(str, Enum):
    SALES = "Sales"
    SUPPORT = "Support"
    REFUND = "Refund"
    PARTNERSHIP = "Partnership"

class CRMEntry(BaseModel):
    sender_name: str = Field(description="The full name of the person who sent the email")
    intent: Intent
    priority_score: int = Field(ge=1, le=10, description="Scale of 1-10")
    action_items: list[str]
    urgency_explanation: str = Field(description="A brief explanation of why this priority was chosen") 

    @field_validator("urgency_explaination")
    @classmethod
    def explaination_must_be_long_enough(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Urgency explanation must be at least 10 words long to be valid")
        return v