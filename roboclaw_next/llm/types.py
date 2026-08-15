"""Provider-neutral LLM types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


ProviderName = Literal["openai", "deepseek"]


# 保存不同 provider 共同使用的生成参数默认值。
@dataclass(frozen=True)
class GenerationConfig:
    """Generation parameters shared by supported providers."""

    temperature: float = 0.7
    max_tokens: int = 4096


# 描述一个具体 LLM provider 的连接配置。
@dataclass(frozen=True)
class ProviderConfig:
    """Connection settings for one LLM provider."""

    provider: ProviderName
    api_key: str
    model: str
    base_url: str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)


# 表示模型请求调用、但尚未执行的工具调用。
@dataclass
class ToolCall:
    """A model-requested tool call in provider-neutral form."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize for appending an assistant tool-call message."""

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


# 将不同 provider 的返回结果统一包装成标准响应结构。
@dataclass
class LLMResponse:
    """Normalized model response."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
