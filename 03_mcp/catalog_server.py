import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

SERVICES_FILE = Path(__file__).parent / "data" / "services.json"

class Service(BaseModel):
    name: str
    description: str
    owner: str
    runbook: str

def load_services() -> list[Service]:
    with SERVICES_FILE.open() as f:
        return [Service(**row) for row in json.load(f)]
    
SERVICES = load_services()

mcp = FastMCP("ServiceCatalog")

@mcp.tool()
async def list_services() -> str:
    """List all available services in the catalog."""
    return "\n".join(f"- {s.name}: {s.description}" for s in SERVICES)

@mcp.tool()
async def get_service_owner(service_name: str) -> str:
    """Get the owner of a specific service.
    
    Args:
        service_name: The name of the service to look up. e.g. billing-service
    """
    for s in SERVICES:
        if s.name == service_name:
            return f"The owner of {service_name} is {s.owner}."
    return f"Unknown service: {service_name}. Please check the service name and try again."
    
@mcp.tool()
async def get_service_runbook(service_name: str) -> str:
    """Get the runbook for a specific service.
    
    Args:
        service_name: The name of the service to look up. e.g. billing-service
    """
    for s in SERVICES:
        if s.name == service_name:
            return f"The runbook for {service_name} is located at: {s.runbook}"
    return f"Unknown service: {service_name}. Please check the service name and try again."
    
@mcp.resource("catalog://services")
def services_resource() -> str:
    """The full service catalog as JSON. Read-only data the host can pull in."""
    return json.dumps([s.model_dump() for s in SERVICES], indent=2)

if __name__ == "__main__":
    mcp.run()