"""OpenAI-compatible chat provider.

OpenAI and DeepSeek both expose OpenAI-style chat-completion APIs, so the
future RoboClaw provider boundary can start with a single implementation and a
small provider factory.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from roboclaw_next.llm.types import GenerationConfig, LLMResponse, ProviderConfig, ToolCall


class LLMProvider(ABC):
    """Minimal provider interface used by RoboClaw Next agents."""

    _RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
    )

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send one chat request."""

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Retry transient provider failures with small backoff."""

        last_response: LLMResponse | None = None
        for delay in self._RETRY_DELAYS:
            response = await self.chat(
                messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
            if response.finish_reason != "error":
                return response
            if not self._is_transient_error(response.content):
                return response
            last_response = response
            await asyncio.sleep(delay)

        return await self.chat(
            messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        ) if last_response is None else last_response

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        error = (content or "").lower()
        return any(marker in error for marker in cls._TRANSIENT_ERROR_MARKERS)


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible chat-completion APIs."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenAI-compatible providers require the `openai` Python package."
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**client_kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        generation = self.config.generation
        kwargs: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": self._normalize_messages(messages),
            "temperature": generation.temperature if temperature is None else temperature,
            "max_tokens": max(1, generation.max_tokens if max_tokens is None else max_tokens),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        try:
            response = await self._client.chat.completions.create(**kwargs)
            return self._parse_response(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean empty content that OpenAI-compatible APIs often reject."""

        normalized: list[dict[str, Any]] = []
        for message in messages:
            clean = dict(message)
            content = clean.get("content")
            if content == "":
                clean["content"] = None if clean.get("role") == "assistant" and clean.get("tool_calls") else "(empty)"
            normalized.append(clean)
        return normalized

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=_parse_tool_arguments(tool_call.function.arguments),
            )
            for tool_call in (message.tool_calls or [])
        ]
        usage = response.usage
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            } if usage else {},
            reasoning_content=getattr(message, "reasoning_content", None) or None,
        )


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
