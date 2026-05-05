# 01 — Email Parser (Structured Output)

Module 1 of the course. Turns a messy customer email into a validated CRM record.

## What it does

Feed in unstructured text like:

> "Hey, I'm super frustrated. I bought your software last week and it's crashing... My name is John Doe."

Get back a typed `CRMEntry` object:

```json
{
  "sender_name": "John Doe",
  "intent": "Refund",
  "priority_score": 9,
  "action_items": ["Investigate dashboard crash", "Process refund"],
  "urgency_explanation": "Customer is asking for a refund and reporting a critical software failure."
}
```

The LLM is forced to return data that matches the Pydantic schema. If it doesn't, Instructor retries automatically.

## Files

- [models.py](models.py) — `CRMEntry` Pydantic model + `Intent` enum + a `field_validator` that rejects too-short explanations.
- [parser.py](parser.py) — `parse_email_to_crm()` wraps the shared LLM client with a system prompt for CRM extraction.
- [example.py](example.py) — End-to-end demo: messy email in, JSON out.

The actual LLM call lives in [../utils/llm_client.py](../utils/llm_client.py) (Instructor + LiteLLM). Provider/model is read from `.env` — see the [project README](../README.md) for setup.

## Run it

From the project root:

```bash
uv run python 01_parser/example.py
```

## Key concepts introduced

- **Pydantic models** as a contract for LLM output (`BaseModel`, `Field`)
- **Enums** to constrain categorical fields so the model can't invent new values
- **`field_validator`** for rules Pydantic alone can't express (e.g. minimum word count)
- **Instructor** wrapping LiteLLM to retry on validation failure

## Deferred to Module 2

- `TypeVar` / `Type[T]` generics in `parser.py`
- Tool calling parameters on `call_llm` (`tools=`, `tool_choice=`)
- Running modules with `python -m` from the project root (no more `sys.path` shim)