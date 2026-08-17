"""Stub LLM provider for PTY integration tests.

Loaded via ``ROBOCLAW_STUB_LLM=tests.integration.stub_llm`` env var.
Maps user keywords to embodied tool calls so the agent exercises
real tool dispatch without hitting an LLM API.
"""

from __future__ import annotations

from typing import Any

from roboclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class StubProvider(LLMProvider):
    def __init__(self, default_model: str = "stub-model"):
        super().__init__(api_key="stub", api_base="stub")
        self.default_model = default_model

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
        last = messages[-1]
        if last.get("role") == "tool":
            return LLMResponse(content=f"Result: {last.get('content', '')}")

        user_text = _last_user_text(messages).lower()
        if "identify" in user_text:
            return _tool_call("setup", {"action": "identify"})
        if "calibrate" in user_text:
            return _tool_call("calibration", {"action": "calibrate"})
        if "teleoperate" in user_text:
            return _tool_call("teleop", {"action": "teleoperate"})
        if "replay" in user_text:
            return _tool_call(
                "replay",
                {"action": "replay", "dataset_name": "demo"},
            )
        return LLMResponse(content=f"Echo: {_last_user_text(messages)}")

    def get_default_model(self) -> str:
        return self.default_model


def create_provider(config: Any) -> StubProvider:
    """Entry point called by ``_make_provider`` when ROBOCLAW_STUB_LLM is set."""
    return StubProvider(default_model=config.agents.defaults.model)


def _tool_call(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="call_1", name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return ""
