"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import uuid
from typing import Any

import json_repair
from openai import AsyncOpenAI

from roboclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None,
                   tool_choice: str | dict[str, Any] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            if tools and self._is_tool_payload_rejected(e):
                fallback = dict(kwargs)
                fallback.pop("tools", None)
                fallback.pop("tool_choice", None)
                fallback["messages"] = self._text_tool_protocol_messages(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                )
                try:
                    parsed = self._parse(await self._client.chat.completions.create(**fallback))
                    content, text_tool_calls = self._parse_text_tool_protocol(parsed.content)
                    if text_tool_calls:
                        parsed.content = content
                        parsed.tool_calls = text_tool_calls
                        parsed.finish_reason = "tool_calls"
                    return parsed
                except Exception as fallback_error:
                    raise fallback_error from e
            raise

    @staticmethod
    def _is_tool_payload_rejected(error: Exception) -> bool:
        text = str(error).lower()
        markers = (
            "unsupported server-side tools",
            "oversized tool descriptions",
            "tool_use/tool_result",
            "mismatched tool",
            "improperly formed",
            "tool_choice",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _text_only_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool-call history to plain text for OpenAI-compatible gateways.

        Some Claude gateways expose an OpenAI-compatible chat endpoint but reject
        OpenAI tool schemas or prior tool result pairs. The general RoboClaw chat
        can still answer user questions without tools, so preserve useful history
        as text and drop provider-specific tool fields.
        """
        safe: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "tool":
                name = str(message.get("name") or "tool")
                safe.append({
                    "role": "user",
                    "content": f"Tool result from {name}: {message.get('content') or '(empty)'}",
                })
                continue

            clean = {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "name"}
            }
            if role == "assistant" and message.get("tool_calls") and not clean.get("content"):
                clean["content"] = "(assistant requested tool calls; omitted for this provider)"
            if clean.get("content") is None:
                clean["content"] = ""
            safe.append(clean)
        return LLMProvider._sanitize_empty_content(safe)

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
