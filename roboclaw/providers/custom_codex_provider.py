"""Custom Codex-compatible provider - Bearer-key auth against /responses.

For gateways that proxy the OpenAI Codex Responses API but authenticate
with a static API key instead of ChatGPT OAuth (e.g. right.codes and
similar new-api / one-api forks). Reuses the request body shape and SSE
parsing from ``openai_codex_provider``; only URL and auth differ.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from roboclaw.providers.base import LLMProvider, LLMResponse
from roboclaw.providers.openai_codex_provider import (
    DEFAULT_ORIGINATOR,
    _convert_messages,
    _convert_tools,
    _prompt_cache_key,
    _request_codex,
    _strip_model_prefix,
)


class CustomCodexProvider(LLMProvider):
    """Codex Responses API client that authenticates with a user-supplied API key."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        default_model: str,
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key=api_key, api_base=api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        sanitized = self._sanitize_empty_content(messages)
        system_prompt, input_items = _convert_messages(sanitized)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model or self.default_model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "prompt_cache_key": _prompt_cache_key(sanitized),
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": True,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = _convert_tools(tools)

        url = _responses_url(self.api_base)
        headers = self._build_headers()

        try:
            try:
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=True)
            except Exception as exc:
                if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                    raise
                logger.warning("SSL certificate verification failed for {}; retrying with verify=False", url)
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=False)
            return LLMResponse(content=content, tool_calls=tool_calls, finish_reason=finish_reason)
        except Exception as exc:
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "originator": DEFAULT_ORIGINATOR,
            "User-Agent": "roboclaw (python)",
        }
        headers.update(self.extra_headers)
        return headers

    def get_default_model(self) -> str:
        return self.default_model


def _responses_url(api_base: str) -> str:
    """Return the concrete Responses API URL for a custom Codex base."""
    base = api_base.rstrip("/")
    if base.endswith("/responses"):
        return base
    return f"{base}/responses"
