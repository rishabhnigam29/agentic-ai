from enum import Enum
from Pydantic import BaseModel

class Intent(str, Enum):
    SALES = "Sales"
    SUPPORT = "Support"
    REFUND = "Refund"
    PARTNERSHIP = "Partnership"

class CRMEntry(BaseModel):
    sender_name: str
    intent: Intent