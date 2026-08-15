"""RoboClaw Next 的最小工具抽象。

这一层只描述 Agent Runtime 如何认识和调用工具，不关心工具来自本地函数、
机器人服务还是外部 MCP server。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolExecutionContext:
    """一次工具调用的运行上下文。

    这里保存的是“谁在什么会话中触发了这次调用”这类运行期信息。
    工具本身的业务参数仍然由 `arguments` 传入。
    """

    tool_call_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行后的统一返回结构。

    `content` 是最终会回到 LLM 上下文中的可读文本。
    `structured_content` 和 `details` 用于保留结构化结果，方便后续 UI、日志、
    机器人状态或调试流程使用。
    """

    content: str
    is_error: bool = False
    structured_content: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """把工具结果转换为可以放进 LLM tool message 的文本。"""

        if self.structured_content is None:
            return self.content
        structured = json.dumps(self.structured_content, ensure_ascii=False, indent=2)
        if not self.content:
            return structured
        return f"{self.content}\n\nstructured_content:\n{structured}"


class AgentTool(ABC):
    """Agent Runtime 内部统一使用的工具接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型 tool call 使用的工具名。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """暴露给模型的工具说明。"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具输入参数的 JSON Schema object。"""

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """在正式执行前整理模型传入的参数。

        第一版默认原样返回。后续如果模型传入空值、别名字段或需要轻量归一化，
        可以由具体工具覆写这个方法。
        """

        return arguments

    @abstractmethod
    async def invoke(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """执行工具，并返回统一的 ToolResult。"""

    def to_openai_schema(self) -> dict[str, Any]:
        """投影成 OpenAI-compatible function calling schema。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
