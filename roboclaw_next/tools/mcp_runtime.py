"""Small MCP client runtime for RoboClaw Next tools.

The first version intentionally supports stdio MCP servers only. This keeps the
learning path small while preserving the same adapter shape that later HTTP/SSE
transports can reuse.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StdioMCPServerConfig:
    """一个通过 stdio 启动的 MCP server 配置。"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class MCPClientRuntime:
    """负责连接 MCP server，并暴露 list_tools/call_tool 这两个核心操作。"""

    def __init__(self, config: StdioMCPServerConfig) -> None:
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: Any | None = None

    async def __aenter__(self) -> MCPClientRuntime:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                )
            )
        )  # 启动并连接 MCP server 子进程
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._stack = stack
        self._session = session
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    @property
    def session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCPClientRuntime is not connected")
        return self._session

    async def list_tools(self) -> list[Any]:
        response = await self.session.list_tools()
        return list(response.tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return await self.session.call_tool(tool_name, arguments=arguments)
