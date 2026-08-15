"""RoboClaw Next 工具注册表。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from roboclaw_next.tools.base import AgentTool, ToolExecutionContext, ToolResult


class ToolRegistry:
    """管理当前 Agent 可见的工具，并根据模型 tool call 名称执行工具。"""

    def __init__(self, tools: Iterable[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        tool = self.get(name)
        prepared_arguments = await tool.prepare_arguments(arguments)
        return await tool.invoke(prepared_arguments, context or ToolExecutionContext())

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
