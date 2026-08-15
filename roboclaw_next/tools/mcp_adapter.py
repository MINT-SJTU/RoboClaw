"""Adapters that materialize MCP tools as RoboClaw Next tools."""

from __future__ import annotations

import json
import re
from typing import Any

from roboclaw_next.tools.base import AgentTool, ToolExecutionContext, ToolResult
from roboclaw_next.tools.mcp_runtime import MCPClientRuntime


def safe_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Create a provider-safe tool name from MCP server/tool names."""

    raw = f"{server_name}__{tool_name}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("MCP tool name cannot be empty")
    return safe


class MCPToolAdapter(AgentTool):
    """把一个 MCP server tool 包装成 RoboClaw Next 内部工具。"""

    def __init__(self, runtime: MCPClientRuntime, tool_def: Any) -> None:
        self._runtime = runtime
        self._original_name = tool_def.name
        self._name = safe_mcp_tool_name(runtime.config.name, tool_def.name)
        self._description = (
            tool_def.description
            or f'Tool "{tool_def.name}" provided by MCP server "{runtime.config.name}".'
        )
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def invoke(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        result = await self._runtime.call_tool(self._original_name, arguments)
        return mcp_call_result_to_tool_result(
            server_name=self._runtime.config.name,
            tool_name=self._original_name,
            result=result,
            context=context,
        )


async def load_mcp_tools(runtime: MCPClientRuntime) -> list[MCPToolAdapter]:
    """List MCP tools and convert them into local RoboClaw tools."""

    return [MCPToolAdapter(runtime, tool_def) for tool_def in await runtime.list_tools()]


def mcp_call_result_to_tool_result(
    *,
    server_name: str,
    tool_name: str,
    result: Any,
    context: ToolExecutionContext | None = None,
) -> ToolResult:
    """Convert MCP CallToolResult into RoboClaw Next's ToolResult."""

    structured = getattr(result, "structuredContent", None)
    content_blocks = getattr(result, "content", []) or []
    text_parts = [_content_block_to_text(block) for block in content_blocks]
    text = "\n".join(part for part in text_parts if part)
    if not text and structured is not None:
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    if not text:
        text = f'MCP tool "{server_name}/{tool_name}" returned no content.'
    details = {"mcp_server": server_name, "mcp_tool": tool_name}
    if context is not None and context.tool_call_id is not None:
        details["tool_call_id"] = context.tool_call_id
    return ToolResult(
        content=text,
        is_error=bool(getattr(result, "isError", False)),
        structured_content=structured if isinstance(structured, dict) else None,
        details=details,
    )


def _content_block_to_text(block: Any) -> str:
    block_type = getattr(block, "type", None)
    if block_type == "text" and hasattr(block, "text"):
        return str(block.text)
    if hasattr(block, "model_dump_json"):
        return block.model_dump_json(indent=2)
    if hasattr(block, "json"):
        return block.json(indent=2)
    return str(block)
