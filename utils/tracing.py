import os
import warnings

from dotenv import load_dotenv

load_dotenv()

def setup_tracing(project_name: str = "agentic-ai") -> bool:
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
    