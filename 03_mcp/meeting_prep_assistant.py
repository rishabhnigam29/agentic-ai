import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM

from utils.llm_config import get_llm_params
from utils.tracing import setup_tracing

from rich.console import Console
from rich.markdown import Markdown

console = Console()

SYSTEM_PROMPT = """
You are a meeting-prep assistant. When asked to prepare for a meeting.
1. Get the current time from the calendar tool.
2. Find the users next upcoming event from the calendar tool.
3. Take the event title, attendes and any details that it carries.
4. Search notion for documents related to the meeting.
5. If a relevant document is found, read its page content before writinga anything. Do not 
write brief from the event title alsone.
6. Write short brief: what the meeting is about, the key points from the doc and any open question to raise.
7. If no relevant document is found, say so plainly
8. Stop calling tools once you have enough to write brief.

Calendar tool rules:
- Format timeMin and timeMax as YYYY-MM-DDTHH:MM:SS, seconds precision,
  no milliseconds.
  Valid example: 2026-01-01T00:00:00
- Leave the account parameter out. Never intent the account names.
"""

def get_connection() -> dict:
    """Two community server: Google Calendar and Notion, both over stdio"""
    return {
        "notion": {
            "command" : "npx",
            "args" : ["-y", "@notionhq/notion-mcp-server"],
            "transport" : "stdio",
            "env" : {
                "NOTION_TOKEN": os.environ.get("NOTION_TOKEN", "")
            }
        },
        "calendar":{
            "command" : "npx",
            "args" : ["@cocal/google-calendar-mcp"],
            "transport" : "stdio",
            "env" : {
                "GOOGLE_OAUTH_CREDENTIALS" : os.environ.get("GOOGLE_OAUTH_CREDENTIALS", "")
            }
        }
    }

async def ask_meeting_prep_assistant(user_question: str) -> str:
    """Ask the meeting prep assistant a question."""
    
    client = MultiServerMCPClient(get_connection())

    tools = await client.get_tools()
    print(f"[MCP] Discovered tools: {len(tools)} tool(s) from server")

    llm = ChatLiteLLM(**get_llm_params(temperature=0, max_tokens=2000))

    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config={"recursion_limit": 20},
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    setup_tracing("module-3-meeting-prep-assistant")
    question = "Prepare me for next meeting."
    answer = asyncio.run(ask_meeting_prep_assistant(question))
    print(f"Question: {question}")
    md= Markdown(answer)
    console.print(md)
