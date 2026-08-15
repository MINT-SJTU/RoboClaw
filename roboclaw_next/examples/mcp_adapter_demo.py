"""Call a local MCP server through RoboClaw Next's tool adapter."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from roboclaw_next.tools import MCPClientRuntime, StdioMCPServerConfig, ToolRegistry
from roboclaw_next.tools import load_mcp_tools


async def main() -> None:
    server_path = Path(__file__).with_name("mcp_demo_server.py")
    config = StdioMCPServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(server_path)],
    )

    async with MCPClientRuntime(config) as runtime:
        tools = await load_mcp_tools(runtime)
        registry = ToolRegistry(tools)

        print("Registered tools:")
        for name in registry.names:
            print(f"- {name}")

        print("\nOpenAI-compatible tool schemas:")
        print(json.dumps(registry.definitions(), ensure_ascii=False, indent=2))

        add_result = await registry.invoke("demo__add", {"a": 19, "b": 23})
        print("\nCall demo__add:")
        print(add_result.as_text())

        echo_result = await registry.invoke("demo__echo", {"text": "hello from MCP"})
        print("\nCall demo__echo:")
        print(echo_result.as_text())


if __name__ == "__main__":
    asyncio.run(main())
