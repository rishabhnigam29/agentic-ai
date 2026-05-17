import os
import warnings

from dotenv import load_dotenv

load_dotenv()

def setup_tracing(project_name: str = "agentic-ai") -> bool:
    """
    Configure LangSmith tracing. Call once at the start of any module entry point.

    For LangChain/LangGraph modules (M2, M4, M5): automatically traces all LLM
    calls, tool invocations, and graph node executions when enabled — no further
    code changes required.

    For non-LangChain modules (e.g. M1 with Instructor): use the @traceable
    decorator exported from this module to manually wrap functions.

    Returns True if tracing is active, False if disabled (no valid API key).
    """
    key = os.getenv("LANGSMITH_API_KEY")
    if not key:
        print("LANGSMITH_API_KEY not found in environment variables. Tracing will be disabled.")
        return False
    
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    os.environ["LANGSMITH_PROJECT"] = project_name

    print(f"LangSmith tracing enabled for project: {project_name}")
    print("To view traces, visit: https://smith.langchain.com/")
    return True

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return args[0] if args and callable(args[0]) else decorator
    