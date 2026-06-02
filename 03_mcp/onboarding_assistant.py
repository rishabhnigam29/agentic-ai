import asyncio
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM

from utils.llm_config import get_llm_params
from utils.tracing import setup_tracing

THIS_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """
You are an engineering onboarding assistant for new joiners and on-call
engineers. When asked about a specific services, owner or runbook, use the
catalog tools and filesystem tools available to you. Otherwise, answer directly. Once you have
enough information to answer, stop calling tools.
"""

def get_catalog_connection() -> dict:
    """stdio (default) or HTTP if MCP_SERVER_URL is set."""
    url = os.environ.get("MCP_SERVER_URL", "").strip()
    if url:
        return {
            "catalog" : {
                "transport" : "streamable_http",
                "url" : url.rstrip("/"),
            }
        }
    return {
        "catalog" : {
            "command": sys.executable,
            "args" : [str(THIS_DIR / "catalog_server.py")],
            "transport": "stdio",
            "cwd": str(THIS_DIR),
        },
        "filesystem": {
            "command" : "npx",
            "args" : [
                "-y", "@modelcontextprotocol/server-filesystem",
                "C:\\agentic-ai\\project_docs"
            ],
            "transport" : "stdio"
        }
    }

async def ask_onboarding_assistant(user_question: str) -> str:
    """Ask the onboarding assistant a question."""
    
    client = MultiServerMCPClient(get_catalog_connection())

    tools = await client.get_tools()
    print(f"[MCP] Discovered tools: {len(tools)} tool(s) from server")
    for tool in tools:
        desc = (tool.description or "").replace("\n", " ")
        print(f"  - {tool.name}: {desc[:100]}")

    llm = ChatLiteLLM(**get_llm_params(temperature=0, max_tokens=2000))

    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config={"recursion_limit": 8},
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    setup_tracing("module-3-mcp-onboarding-assistant")
    question = "find runbook url for the billing service, then list any files in project_docs that have 'billing in there name'. Give me its full path"
    answer = asyncio.run(ask_onboarding_assistant(question))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
