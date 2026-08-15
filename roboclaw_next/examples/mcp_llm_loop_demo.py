"""Optional demo: let an LLM choose and call the MCP-backed tool.

This requires an API key for the chosen provider:

    OPENAI_API_KEY=... PYTHONPATH=. uv run --no-project \
        --with "mcp[cli]<2" --with openai \
        python roboclaw_next/examples/mcp_llm_loop_demo.py

or:

    DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
        uv run --no-project --with "mcp[cli]<2" --with openai \
        python roboclaw_next/examples/mcp_llm_loop_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from roboclaw_next.agent import run_tool_call_loop
from roboclaw_next.llm import create_llm_provider
from roboclaw_next.tools import MCPClientRuntime, StdioMCPServerConfig, ToolRegistry
from roboclaw_next.tools import load_mcp_tools


async def main() -> None:
    server_path = Path(__file__).with_name("mcp_demo_server.py")
    config = StdioMCPServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(server_path)],
    )
    provider_name = os.environ.get("ROBOCLAW_LLM_PROVIDER", "openai")
    provider = create_llm_provider(provider_name)  # type: ignore[arg-type]

    async with MCPClientRuntime(config) as runtime:
        registry = ToolRegistry(await load_mcp_tools(runtime))
        answer = await run_tool_call_loop(
            provider,
            [
                {"role": "system", "content": "You can use tools when helpful."},
                {"role": "user", "content": "Please calculate 19 + 23 using the tool."},
            ],
            registry,
        )
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
