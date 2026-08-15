"""Let an LLM choose and call tools exposed by a local MCP server.

Run with DeepSeek:

    DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
        uv run --no-project --with openai --with "mcp[cli]<2" \
        python roboclaw_next/examples/mcp_tool_call_demo.py

Run with OpenAI:

    OPENAI_API_KEY=... ROBOCLAW_LLM_PROVIDER=openai PYTHONPATH=. \
        uv run --no-project --with openai --with "mcp[cli]<2" \
        python roboclaw_next/examples/mcp_tool_call_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import cast

from roboclaw_next.agent import run_tool_call_loop
from roboclaw_next.llm import create_llm_provider
from roboclaw_next.llm.types import ProviderName
from roboclaw_next.tools import (
    MCPClientRuntime,
    StdioMCPServerConfig,
    ToolRegistry,
    load_mcp_tools,
)


def resolve_provider_name() -> ProviderName:
    raw_provider = os.environ.get("ROBOCLAW_LLM_PROVIDER", "deepseek").lower()
    if raw_provider not in ("openai", "deepseek"):
        raise ValueError("ROBOCLAW_LLM_PROVIDER must be 'openai' or 'deepseek'.")
    return cast(ProviderName, raw_provider)


async def main() -> None:
    server_path = Path(__file__).with_name("example_mcp_server.py")
    config = StdioMCPServerConfig(
        name="example_mcp",
        command=sys.executable,
        args=[str(server_path)],
    )

    async with MCPClientRuntime(config) as runtime:
        registry = ToolRegistry(await load_mcp_tools(runtime))

        print("Registered MCP-backed tools:")
        for name in registry.names:
            print(f"- {name}")

        provider = create_llm_provider(
            resolve_provider_name(),
            temperature=0,
            max_tokens=1024,
        )
        answer = await run_tool_call_loop(
            provider,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a tool-using assistant. For arithmetic tasks, "
                        "call the provided tool instead of calculating directly. "
                        "After tool execution, answer the user in Chinese."
                    ),
                },
                {
                    "role": "user",
                    "content": "请使用 MCP 工具计算 19 + 23，并告诉我最终结果。",
                },
            ],
            registry,
        )
        if answer is None:
            raise RuntimeError("LLM MCP tool loop did not produce a final answer.")

        print("\nFinal answer:")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
