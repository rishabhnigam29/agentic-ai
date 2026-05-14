"""
Hand-rolled agent loop. No frameworks. Pedagogical anti-pattern by lesson 6.

We call litellm directly. We parse 'tool_calls' ourselves. We run the function
ourselves. We feed the result back ourselves. 

"""

import json
from utils.llm_client import call_llm

# A "tool" is just a Python function. The llm doesn't run it. We do.
def get_stock_price(symbol: str) -> str:
    fake_prices = {"AAPL": "150.00", "GOOG": "2800.00", "MSFT": "300.00"}
    price = fake_prices.get(symbol.upper(), "Unknown symbol")
    return f"The current price of {symbol.upper()} is ${price}."

# This is the schema that LLM sees. It's how the model knows what tools exits
# and what arguments to send. The share is the OpenAi function-calling spec,
# which litellm forwards to every provider
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock symbol to look up (e.g., AAPL, GOOG)."
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

def run_agent(user_question: str, max_iterations: int = 5) -> str:
    messages = [
        {"role": "system", "content": "You are a financial assistant. Use tools when needed"},
        {"role": "user", "content": user_question}
    ]

    for step in range(max_iterations):
        response = call_llm(
            messages=messages,
            tools=TOOLS_SCHEMA,
            return_raw_response=True)
        assistant_message = response.choices[0].message

        #If the LLM didn't ask for a tool, we're done
        tools_calls = getattr(assistant_message, "tool_calls", None)
        if not tools_calls:
            return assistant_message.content
        
        #Otherwise, append the assistant turn and run each tool call by hand
        messages.append(assistant_message.model_dump())
        for call in tools_calls:
            args = json.loads(call.function.arguments)
            print(f" [step {step+1}] LLM called tool {call.function.name} with args {args}")
            try:
                if call.function.name == "get_stock_price":
                    result = get_stock_price(**args)
                else:
                    result = f"Unknown tool: {call.function.name}"
            except TypeError as e:
                result = f"Error calling tool {call.function.name}: {str(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id" : call.id,
                "name": call.function.name,
                "content": result
            })

    return "Hit max iterations without a final answer. "

if __name__ == "__main__":
    question = "What's the current price of AAPL?"
    answer = run_agent(question)
    print("Final answer:", answer)                