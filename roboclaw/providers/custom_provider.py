"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import json_repair
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from roboclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_DEFAULT_TIMEOUT_S = 60.0
_UNSUPPORTED_PARAM_MARKERS = (
    "unsupported parameter",
    "unknown parameter",
    "unrecognized parameter",
    "extra inputs are not permitted",
    "not supported",
)


class CustomProvider(LLMProvider):

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        extra_headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        default_headers = dict(self.extra_headers)
        default_headers.setdefault("x-session-affinity", uuid.uuid4().hex)
        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers=default_headers,
            timeout=timeout,
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None,
                   tool_choice: str | dict[str, Any] | None = None) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except BadRequestError as exc:
            retry_kwargs = self._compatible_retry_kwargs(kwargs, exc)
            if retry_kwargs is not None:
                try:
                    return self._parse(await self._client.chat.completions.create(**retry_kwargs))
                except Exception as retry_exc:
                    return self._error_response(retry_exc)
            return self._error_response(exc)
        except Exception as exc:
            return self._error_response(exc)

    def _build_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
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
        return kwargs

    def _compatible_retry_kwargs(
        self,
        kwargs: dict[str, Any],
        exc: BadRequestError,
    ) -> dict[str, Any] | None:
        message = str(exc).lower()
        if not any(marker in message for marker in _UNSUPPORTED_PARAM_MARKERS):
            return None

        if "reasoning_effort" in kwargs and "reasoning_effort" in message:
            retry = dict(kwargs)
            retry.pop("reasoning_effort", None)
            return retry

        if "tool_choice" in kwargs and "tool_choice" in message:
            retry = dict(kwargs)
            retry.pop("tool_choice", None)
            return retry

        return None

    @staticmethod
    def _error_response(exc: Exception) -> LLMResponse:
        if isinstance(exc, AuthenticationError):
            label = "authentication failed"
        elif isinstance(exc, RateLimitError):
            label = "rate limited"
        elif isinstance(exc, APITimeoutError) or isinstance(exc, asyncio.TimeoutError):
            label = "request timed out"
        elif isinstance(exc, APIConnectionError):
            label = "connection failed"
        elif isinstance(exc, BadRequestError):
            label = "bad request"
        elif isinstance(exc, APIStatusError):
            label = f"api status {exc.status_code}"
        else:
            label = exc.__class__.__name__
        return LLMResponse(content=f"Error ({label}): {exc}", finish_reason="error")

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
