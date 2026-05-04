from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

alice = Person(name="Alice", age=30)
print("Valid object:", alice)
print("As JSON:", alice.model_dump_json(indent=2))

print("\n ----- Trying invalid data -----")
try:
    bad = Person(name="Bob", age="not a number")
except Exception as e:
    print("Error creating Person with invalid age:", e)