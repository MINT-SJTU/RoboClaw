from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.subagent import SubagentManager
from roboclaw.bus.queue import MessageBus
from roboclaw.providers.base import LLMResponse


class _CapturingProvider:
    def __init__(self) -> None:
        self.tool_names: list[str] = []

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.tool_names = [tool["function"]["name"] for tool in (tools or [])]
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "openai-codex/gpt-5.4"


@pytest.mark.asyncio
async def test_embodied_tools_are_registered_only_in_main_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _CapturingProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )

    assert loop.tools.has("embodied_status")
    assert loop.tools.has("embodied_control")

    subagent_provider = _CapturingProvider()
    manager = SubagentManager(
        provider=subagent_provider,
        workspace=tmp_path,
        bus=MessageBus(),
    )

    async def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(manager, "_announce_result", _noop)

    await manager._run_subagent("task1", "say hi", "label", {"channel": "cli", "chat_id": "direct"})

    assert "embodied_status" not in subagent_provider.tool_names
    assert "embodied_control" not in subagent_provider.tool_names
