from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ServiceCatalog")

SERVICES = [
    "auth-api",
    "billing-service",
    "email-worker"
]

@mcp.tool()
async def list_services() -> str:
    """List all available services in the catalog."""
    return "\n".join(f"- {s}" for s in SERVICES)

if __name__ == "__main__":
    mcp.run()