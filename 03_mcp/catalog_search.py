import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

THIS_DIR = Path(__file__).resolve().parent

async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(THIS_DIR / "catalog_server.py")],
        cwd = str(THIS_DIR),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            print(f"Server exposed {len(tools_response.tools)} tools:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")

            print("\nCalling list services...")
            result = await session.call_tool("list_services", arguments={})
            print(result.content[0].text)

            print("\nCalling get_service_owner(service_name='billing-service')...")
            result = await session.call_tool(
                "get_service_owner", arguments={"service_name": "billing-service"})
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())