"""The Deep Reader - Chainlit chat UI (Module 6 finale).

Wraps the whole project in one chat app: the memory-aware RAG from
Part 1 (Module 4) plus the agentic routing, self-correction, metadata
filtering, and full-document audit from Part 2 (Module 6). Pick a
document type from the chat-profile selector, then ask questions or say
"audit the whole document".

Memory is the graph's checkpointer: each chat session is one thread, so
follow-ups just work - the app never touches a memory object.

Run: uv run --extra ui chainlit run 06_agentic_rag/app.py -w
"""

import asyncio

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "06_agentic_rag")]

for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from utils.tracing import setup_tracing

setup_tracing("module-6-deep-reader-ui")

import chainlit as cl
from agent.graph import compile_graph
from stages.helpers import ensure_ingested
from shared.profiles import PROFILES
from shared import config

ROUTE_LABELS = {
    "docs": "🔍 Searched the document",
    "memory": "🧠 Recalled from memory",
    "direct": "💬 Answered directly",
    "full_scan": "📄 Scanned the entire document",
}

# The chosen document lives in one module-level setting, so it belongs to the
# whole process, not to one chat. Two browser tabs on different documents would
# fight over it and the last tab to open would win for everybody. So each chat
# remembers its own choice, and every call into the agent re-applies that choice
# while holding this lock: one run can never read another chat's document.
DEFAULT_PROFILE = config.ACTIVE_PROFILE
_profile_lock = asyncio.Lock()


async def _run_on_profile(profile_key: str, fn, *args):
    """Run a blocking agent call with this chat's document selected."""
    async with _profile_lock:
        config.ACTIVE_PROFILE = profile_key
        return await cl.make_async(fn)(*args)


def _decision_path(result: dict) -> str:
    """One compact line describing how the agent reached its answer."""
    route = result.get("route", "unknown")
    bits = [ROUTE_LABELS.get(route, route)]

    if route == "full_scan":
        analysis = result.get("analysis_result")
        bits.append(f"{len(result.get('retrieved_chunks', []))} chunks")
        if analysis:
            bits.append(f"{len(analysis.findings)} findings, risk: {analysis.overall_rating}")
    else:
        mf = result.get("metadata_filter")
        if mf and mf.sections:
            bits.append(f"sections: {', '.join(mf.sections)}")
        attempts = result.get("retrieval_attempts", 0)
        if attempts:
            bits.append(f"{attempts} retrieval attempt(s)")
    mem = len(result.get("memory_context", []))
    if mem:
        bits.append(f"{mem} memory snippet(s)")
    return " · ".join(bits)


@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(name=p.key, markdown_description=f"**{p.label}**")
        for p in PROFILES.values()
    ]


@cl.on_chat_start
async def on_chat_start():
    profile_key = cl.user_session.get("chat_profile") or DEFAULT_PROFILE
    cl.user_session.set("profile_key", profile_key)
    profile = PROFILES[profile_key]

    loading = cl.Message(content=f"Loading **{profile.label}** ...")
    await loading.send()
    # first run embeds + tags the doc
    await _run_on_profile(profile_key, ensure_ingested)

    cl.user_session.set("app", compile_graph())

    loading.content = (
        f"**Deep Reader** is ready on the **{profile.label}**.\n\n"
        f"Ask a specific question, a follow-up, or say "
        f"*\"audit the whole document\"* for a full structured review."
    )
    await loading.update()


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")
    profile_key = cl.user_session.get("profile_key") or DEFAULT_PROFILE

    state = {
        # conversation_history stays out: the checkpointer holds this
        # thread's history, and one chat session = one thread.
        "query": message.content,
        "route": "",
        "rewritten_query": "",
        "metadata_filter": None,
        "retrieved_chunks": [],
        "chunks_relevant": False,
        "retrieval_attempts": 0,
        "memory_context": [],
        "answer": "",
        "analysis_result": None,
    }
    thread = {"configurable": {"thread_id": cl.user_session.get("id")}}

    async with cl.Step(name="agent") as step:
        result = await _run_on_profile(profile_key, app.invoke, state, thread)
        step.output = _decision_path(result)

    answer = result.get("answer", "") or "(no answer produced)"
    await cl.Message(content=answer).send()
