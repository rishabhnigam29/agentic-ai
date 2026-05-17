# 02_agent_loop - Beginner's Guide

## 🎯 What is This?

This directory contains an **AI Agent** that can answer questions about stocks by:
1. **Thinking** about what information it needs
2. **Using tools** to get real data (stock prices, news)
3. **Reasoning** about the results
4. **Answering** your question

Think of it like a smart assistant that can:
- Look up stock prices
- Search the web for news
- Analyze information
- Give you structured recommendations

---

## 📚 Key Concepts (Explained Simply)

### 1. **What is an AI Agent?**
An AI agent is like a smart assistant that can:
- Understand your question
- Decide what actions to take
- Use tools to gather information
- Think about the results
- Give you a final answer

**Real-world analogy**: Like asking a financial advisor who can:
- Check stock prices (tool 1)
- Search for news (tool 2)
- Analyze everything
- Give you advice

### 2. **What is the ReAct Pattern?**
ReAct stands for **Reasoning + Acting**. It's a loop:

```
1. REASON: "I need to find the stock price"
2. ACT: Call get_stock_price("AAPL")
3. OBSERVE: "The price is $150"
4. REASON: "Now I need recent news"
5. ACT: Search for "AAPL news"
6. OBSERVE: "Found news articles"
7. REASON: "I have enough info, I can answer now"
8. ACT: Provide final answer
```

The agent repeats this cycle until it has enough information to answer.

### 3. **What are Tools?**
Tools are functions the agent can call to get real-world data:

- **`get_stock_price`**: Gets current stock price from Yahoo Finance
- **`tavily_search`**: Searches the web for news/articles

The agent decides when and how to use these tools automatically!

### 4. **What is Structured Output?**
Instead of getting free-form text, you can ask for data in a specific format:

```python
{
  "symbol": "AAPL",
  "action": "Buy",
  "rationale": "...",
  "confidence_score": 0.85,
  "key_headlines": ["...", "..."]
}
```

This makes it easy to use the data in your code!

---

## 📁 Files Explained

### 1. **`tools.py`** - The Agent's Toolkit

**What it does**: Defines all the tools the agent can use.

**Key Functions**:

#### `get_stock_price(symbol: str)`
- **Purpose**: Gets the current stock price
- **How it works**: Uses `yfinance` library to fetch real-time stock data
- **Example**: `get_stock_price("AAPL")` → "The current price of AAPL is $150.25"

#### `get_tavily_search_tool()`
- **Purpose**: Creates a web search tool
- **How it works**: Uses Tavily API to search the internet
- **Returns**: A search tool that can find news/articles
- **Note**: Only works if you have a Tavily API key in `.env`

#### `get_all_tools()`
- **Purpose**: Collects all available tools
- **Returns**: List of tools `[get_stock_price, tavily_search]`
- **Why**: The agent needs to know what tools it can use

**Key Libraries Used**:
- `yfinance`: Fetches stock market data
- `langchain_tavily`: Web search tool
- `@tool` decorator: Tells LangChain this is a tool the agent can use

---

### 2. **`agent.py`** - The Brain

**What it does**: Creates and runs the AI agent.

**Key Functions**:

#### `create_react_agent_executor()`
- **Purpose**: Sets up the agent with LLM and tools
- **What it does**:
  1. Gets the LLM model from `.env` (e.g., GPT-4)
  2. Loads all available tools
  3. Creates a ReAct agent that can use those tools
  4. Returns the agent ready to use

**Key Components**:
- **LLM (Large Language Model)**: The "brain" that reasons (GPT-4, Claude, etc.)
- **Tools**: What the agent can use (stock price, search)
- **System Prompt**: Instructions for the agent ("You are a financial assistant...")

#### `run_agent_with_tools(user_query: str)`
- **Purpose**: Runs the agent with a user question
- **What happens**:
  1. Creates the agent
  2. Sends your question to the agent
  3. Agent reasons, uses tools, reasons again
  4. Returns the final answer

**Example Flow**:
```python
run_agent_with_tools("What's AAPL's price?")
# Agent thinks: "I need stock price"
# Agent calls: get_stock_price("AAPL")
# Agent gets: "Price is $150"
# Agent responds: "The current price of AAPL is $150"
```

**Key Libraries Used**:
- `langchain`: Framework for building agents
- `langchain_litellm`: Connects LangChain to LiteLLM (which connects to GPT-4, etc.)

---

### 3. **`stockRecommendationModels.py`** - Data Structure

**What it does**: Defines the shape of structured data.

**Key Class**:

#### `StockRecommendation`
A Pydantic model that defines what a recommendation looks like:

```python
class StockRecommendation(BaseModel):
    symbol: str                    # Stock ticker (e.g., "AAPL")
    action: str                   # "Buy", "Sell", or "Hold"
    rationale: str                # Why this recommendation
    confidence_score: float        # 0.0 to 1.0 (how confident)
    key_headlines: List[str]      # Important news headlines
```

**Why Pydantic?**
- Validates data automatically
- Ensures the LLM returns data in the correct format
- Type-safe (Python knows the types)

**Example Output**:
```json
{
  "symbol": "AAPL",
  "action": "Buy",
  "rationale": "Strong performance...",
  "confidence_score": 0.85,
  "key_headlines": ["Apple reports record sales", "..."]
}
```

---

### 4. **`stock_recommendation.py`** - Putting It All Together

**What it does**: Combines the agent with structured output.

**Key Function**:

#### `get_stock_recommendation(symbol: str)`
- **Purpose**: Gets a structured stock recommendation
- **How it works**:

**Step 1**: Use agent to gather information
```python
query = f"Analyze {symbol} stock. Get the current price and latest news..."
agent_response = run_agent_with_tools(query)
```
- Agent uses tools to get price and news
- Returns a text summary

**Step 2**: Convert to structured format
```python
call_llm(
    messages=[...],
    response_model=StockRecommendation  # Force structured output
)
```
- Takes the agent's text response
- Uses LLM to convert it into `StockRecommendation` format
- Returns a structured object

**Why Two Steps?**
1. **Agent step**: Gathers real-world data (prices, news)
2. **Structured step**: Formats it nicely for your code

---

### 5. How to Run It

The runnable entry point is `stock_recommendation.py` itself. Either run it directly, or import from a Python REPL / script:

```python
from stock_recommendation import get_stock_recommendation

recommendation = get_stock_recommendation("AAPL")
print(recommendation.model_dump_json(indent=2))
```

**What happens**:
1. Calls `get_stock_recommendation("AAPL")`
2. Agent gets stock price and news
3. Returns structured recommendation
4. Prints it as JSON

**Prefer a UI?** See [ui_demos/02_agent_loop_streamlit/](../ui_demos/02_agent_loop_streamlit/) — a Streamlit wrapper around this exact module with sample tickers, model-provider selector, and a styled recommendation card.

---

## 🔄 How Everything Works Together

```
┌─────────────────────────┐
│ stock_recommendation.py │  ← You call get_stock_recommendation()
└────────┬────────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────┐    ┌──────────────┐
│   agent.py   │    │  tools.py    │
│              │    │              │
│ Creates agent│───▶│ Defines tools│
│ Runs ReAct   │    │ (stock price)│
│ loop         │    │ (web search) │
└──────────────┘    └──────────────┘
         │
         ▼
┌─────────────────────────┐
│ stockRecommendationModels│
│                         │
│ Defines data structure  │
└─────────────────────────┘
```

**Flow**:
1. **You** call `get_stock_recommendation("AAPL")`
2. **stock_recommendation.py** calls `run_agent_with_tools()` to gather info
3. **agent.py** creates agent, agent uses tools from **tools.py**
4. Agent returns text summary
5. **stock_recommendation.py** converts text to structured format using **stockRecommendationModels.py**
6. You get a nice structured recommendation!

---

## 🛠️ Key Technologies Used

### 1. **LangChain**
- **What**: Framework for building AI applications
- **Why**: Makes it easy to create agents, tools, and chains
- **In this project**: Used for the ReAct agent

### 2. **LiteLLM**
- **What**: Unified interface to 100+ LLM providers
- **Why**: Switch between GPT-4, Claude, Ollama, etc. easily
- **In this project**: Connects to OpenAI/other LLMs

### 3. **Pydantic**
- **What**: Data validation library
- **Why**: Ensures structured outputs are correct
- **In this project**: Defines `StockRecommendation` model

### 4. **yfinance**
- **What**: Python library for stock market data
- **Why**: Gets real stock prices
- **In this project**: Used in `get_stock_price` tool

### 5. **Tavily**
- **What**: Search API optimized for AI agents
- **Why**: Gets relevant news/articles
- **In this project**: Used in `get_tavily_search_tool`

---

---

## 🎓 Learning Path

### Beginner Level
1. ✅ Understand what an agent is
2. ✅ Understand what tools are
3. ✅ Run `example.py` and see it work

### Intermediate Level
1. Modify `tools.py` to add a new tool
2. Change the system prompt in `agent.py`
3. Modify `StockRecommendation` model

### Advanced Level
1. Create custom agent loops
2. Add more complex tools
3. Implement error handling and retries
4. Add streaming responses

---

## 💡 Common Questions

### Q: Why do we need both `agent.py` and `stock_recommendation.py`?
**A**: 
- `agent.py`: Generic agent that can answer any question
- `stock_recommendation.py`: Specific use case that uses the agent + adds structure

### Q: What's the difference between `run_agent_with_tools()` and `get_stock_recommendation()`?
**A**:
- `run_agent_with_tools()`: Returns free-form text
- `get_stock_recommendation()`: Returns structured data (easier to use in code)

### Q: Can I add more tools?
**A**: Yes! Add a function in `tools.py` with `@tool` decorator, and it will be available to the agent automatically.

### Q: How does the agent know which tool to use?
**A**: The LLM (GPT-4, etc.) reads the tool descriptions and decides based on your question. It's smart!

---

## 🚀 Next Steps

1. **Run it**: `python stock_recommendation.py` (or import `get_stock_recommendation` from another script)
2. **Try different stocks**: Change "AAPL" to "TSLA", "MSFT", etc.
3. **Add a tool**: Create a new function in `tools.py` with `@tool` decorator
4. **Modify the model**: Change fields in `StockRecommendation`
5. **Experiment**: Change the system prompt in `agent.py` to see different behaviors

---

---

## Tracing with LangSmith

LangSmith is an observability platform that records every LLM call, tool invocation, and agent decision in a visual UI — like a debugger for your agent.

### Setup (2 minutes, free tier available)

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Go to **Settings → API Keys → Create API Key**
3. Add your key to `.env`:

```bash
LANGSMITH_API_KEY=ls__your_key_here
```

The project name is set in code by `setup_tracing("module-2-research-agent")` — no `.env` entry needed.

That's it. The next time you run `python 02_agent_loop/stock_recommendation.py`, you'll see:

```
✓  LangSmith tracing: ACTIVE → project 'module-2-research-agent'
   View traces at: https://smith.langchain.com
```

### What you'll see in the UI

The stock recommendation run produces a trace that looks like this:

```
stock_recommendation  (root span, ~800ms)
├── ChatLiteLLM          — initial ReAct reasoning step
├── get_stock_price      — tool call → returns "$182.50"
├── tavily_search        — tool call → returns news snippets
├── ChatLiteLLM          — agent synthesises findings
└── ChatLiteLLM          — structured output → StockRecommendation
```

For each span you can see:
- **Input / output**: the exact prompt sent and response received
- **Latency**: how long each step took
- **Token usage**: input tokens, output tokens, total cost
- **Model**: which model was called at each step

### Why this matters

Without tracing, when the agent gives a wrong answer, you're debugging blind. With tracing you can open the specific run, click on the retrieval step that failed, and see exactly what chunks it found and why it went wrong. This is how real agent engineering is done in production.

---

## 📖 Additional Resources

- **LangChain Docs**: https://python.langchain.com/
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **ReAct Paper**: https://arxiv.org/abs/2210.03629
- **LangSmith Docs**: https://docs.smith.langchain.com/