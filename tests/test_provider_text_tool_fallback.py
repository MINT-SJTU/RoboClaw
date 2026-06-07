from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from roboclaw.providers.custom_provider import CustomProvider
from roboclaw.providers.litellm_provider import LiteLLMProvider


def _tool_schema() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "start_cloud_training",
            "description": "Start a cloud training job after user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": {"type": "string"},
                    "steps": {"type": "integer"},
                },
                "required": ["workflow"],
            },
        },
    }]


def _consult_and_low_level_tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "evo_studio_agent_consult",
                "description": "Delegate Evo Studio product tasks to the backend agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "mode": {"type": "string"},
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Execute a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evo_studio_cloud_train",
                "description": "Low-level cloud training operation.",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                },
            },
        },
    ]


def _fake_openai_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_custom_provider_emulates_tool_calling_when_gateway_rejects_tools() -> None:
    async def _run() -> None:
        provider = CustomProvider(
            api_key="sk-test",
            api_base="https://gateway.example/v1",
            default_model="claude-test",
        )
        create = AsyncMock(side_effect=[
            RuntimeError("upstream rejected the request as improperly formed: unsupported server-side tools"),
            _fake_openai_response('{"tool_calls":[{"name":"start_cloud_training","arguments":{"workflow":"rlinf_vla","steps":1000}}]}'),
        ])
        provider._client = SimpleNamespace(  # type: ignore[attr-defined]
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        response = await provider.chat(
            messages=[{"role": "user", "content": "Start a smoke training job"}],
            tools=_tool_schema(),
        )

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "start_cloud_training"
        assert response.tool_calls[0].arguments == {"workflow": "rlinf_vla", "steps": 1000}
        fallback_messages = create.call_args_list[1].kwargs["messages"]
        assert "text tool protocol" in fallback_messages[0]["content"]
        assert "start_cloud_training" in fallback_messages[0]["content"]

    asyncio.run(_run())


def test_litellm_provider_emulates_tool_calling_when_gateway_rejects_tools() -> None:
    async def _run() -> None:
        mock_acompletion = AsyncMock(side_effect=[
            RuntimeError("unsupported server-side tools"),
            _fake_openai_response('{"tool_calls":[{"name":"start_cloud_training","arguments":{"workflow":"rlinf_vla"}}]}'),
        ])

        with patch("roboclaw.providers.litellm_provider.acompletion", mock_acompletion):
            provider = LiteLLMProvider(
                api_key="sk-test",
                api_base="https://gateway.example/v1",
                default_model="claude-test",
                provider_name="openrouter",
            )
            response = await provider.chat(
                messages=[{"role": "user", "content": "Start a smoke training job"}],
                tools=_tool_schema(),
            )

        assert response.finish_reason == "tool_calls"
        assert response.tool_calls[0].name == "start_cloud_training"
        assert response.tool_calls[0].arguments == {"workflow": "rlinf_vla"}
        fallback_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
        assert "text tool protocol" in fallback_messages[0]["content"]

    asyncio.run(_run())


def test_text_tool_protocol_converts_tool_history_to_plain_chat_messages() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    messages = [
        {"role": "user", "content": "Read the robot-training skill"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"skills/robot-training/SKILL.md"}'},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "read_file",
            "content": "robot training skill body",
        },
    ]

    fallback = provider._text_tool_protocol_messages(messages, _tool_schema())

    assert fallback[0]["role"] == "system"
    assert fallback[2]["role"] == "assistant"
    assert "tool_calls" not in fallback[2]
    assert "Previous assistant tool requests" in fallback[2]["content"]
    assert fallback[3] == {
        "role": "user",
        "content": "Tool result from read_file: robot training skill body",
    }


def test_text_tool_protocol_uses_consult_surface_for_chat_only_gateways() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )

    fallback = provider._text_tool_protocol_messages(
        [{"role": "user", "content": "帮我在云端跑 OpenVLA-OFT"}],
        _consult_and_low_level_tool_schema(),
    )

    protocol = fallback[0]["content"]
    assert "evo_studio_agent_consult" in protocol
    assert '"name": "exec"' not in protocol
    assert '"name": "evo_studio_cloud_train"' not in protocol
    assert "delegate Evo Studio" in protocol


def test_text_tool_protocol_rewrites_cloud_probe_exec_history() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    messages = [
        {"role": "user", "content": "检查 SSH 后端环境"},
        {
            "role": "assistant",
            "content": "我先检查 SSH 后端。",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"command":"whoami && hostname && pwd && date"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"command":"nvidia-smi --query-gpu=name"}'},
                },
            ],
        },
    ]

    fallback = provider._text_tool_protocol_messages(messages, _tool_schema())

    assistant_content = fallback[2]["content"]
    assert "Previous assistant tool requests" in assistant_content
    assert "exec(" not in assistant_content
    assert 'evo_studio_cloud_train({"action":"backend_probe"})' in assistant_content


def test_text_tool_protocol_rewrites_plain_cloud_probe_history() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    messages = [
        {
            "role": "assistant",
            "content": (
                "Previous assistant tool requests:\n"
                '- exec({"command":"nvidia-smi --query-gpu=name"})\n'
                '- exec({"command":"python3 --version"})'
            ),
        },
    ]

    fallback = provider._text_tool_protocol_messages(messages, _tool_schema())

    assert "exec(" not in fallback[1]["content"]
    assert 'evo_studio_cloud_train({"action":"backend_probe"})' in fallback[1]["content"]


def test_text_tool_protocol_recovers_plain_consult_request() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    content = (
        "我现在通过 evo_studio_agent_consult 委托后端代理。\n"
        "Previous assistant tool requests:\n"
        'evo_studio_agent_consult({"task":"在 SSH 后端复现 OpenVLA-OFT baseline","mode":"execute","context":{"backend":"ssh"}})'
    )

    message, calls = provider._parse_text_tool_protocol(content)

    assert message is None
    assert len(calls) == 1
    assert calls[0].name == "evo_studio_agent_consult"
    assert calls[0].arguments == {
        "task": "在 SSH 后端复现 OpenVLA-OFT baseline",
        "mode": "execute",
        "context": {"backend": "ssh"},
    }


def test_text_tool_protocol_recovers_plain_safe_cloud_train_request() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    content = (
        "我立即查询当前任务状态，确认是否还在运行。\n"
        "Previous assistant tool requests:\n"
        'evo_studio_cloud_train({"action":"current","username":"pearl"})'
    )

    message, calls = provider._parse_text_tool_protocol(content)

    assert message is None
    assert len(calls) == 1
    assert calls[0].name == "evo_studio_cloud_train"
    assert calls[0].arguments == {"action": "current", "username": "pearl"}


def test_text_tool_protocol_does_not_recover_plain_mutating_cloud_train_request() -> None:
    provider = CustomProvider(
        api_key="sk-test",
        api_base="https://gateway.example/v1",
        default_model="claude-test",
    )
    content = (
        "Previous assistant tool requests:\n"
        'evo_studio_cloud_train({"action":"start","username":"pearl","confirmed":true})'
    )

    message, calls = provider._parse_text_tool_protocol(content)

    assert message == content
    assert calls == []
