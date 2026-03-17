from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.base import Tool
from roboclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage
from roboclaw.bus.queue import MessageBus
from roboclaw.embodied.onboarding import SETUP_STATE_KEY, OnboardingController
from roboclaw.providers.base import LLMResponse
from roboclaw.session.manager import Session


class FakeExecTool(Tool):
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Fake exec tool for onboarding tests."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {"type": "string"},
            },
            "required": ["command"],
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs) -> str:
        self.calls.append(command)
        for marker, result in self.responses.items():
            if marker in command:
                return result
        return "(no output)"


class DummyProvider:
    def __init__(self) -> None:
        self.chat_calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(content="provider should not be called")

    def get_default_model(self) -> str:
        return "openai-codex/gpt-5.4"


def _prepare_workspace(root: Path) -> None:
    for rel in (
        "embodied/intake",
        "embodied/assemblies",
        "embodied/deployments",
        "embodied/adapters",
        "embodied/guides",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "embodied" / "guides" / "ROS2_INSTALL.md").write_text(
        "# ROS2 Install\n\n## Ubuntu 24.04\nUse Jazzy.\n",
        encoding="utf-8",
    )


def _build_tools(workspace: Path, exec_responses: dict[str, str]) -> tuple[ToolRegistry, FakeExecTool]:
    registry = ToolRegistry()
    for cls in (ReadFileTool, WriteFileTool, ListDirTool):
        registry.register(cls(workspace=workspace))
    fake_exec = FakeExecTool(exec_responses)
    registry.register(fake_exec)
    return registry, fake_exec


@pytest.mark.asyncio
async def test_onboarding_generates_ready_setup_for_so101_with_camera(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/serial/by-id/usb-so101\n/dev/ttyACM0\n",
            "command -v ros2": "ROS2_OK\nros2 0.0.0\nROS_DISTRO=jazzy\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="I want to connect a real robot"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101 and a wrist camera"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assembly_path = tmp_path / "embodied" / "assemblies" / "so101_setup.py"
    deployment_path = tmp_path / "embodied" / "deployments" / "so101_setup_real_local.py"
    adapter_path = tmp_path / "embodied" / "adapters" / "so101_setup_ros2_local.py"

    assert "ready" in response.content
    assert state["stage"] == "handoff_ready"
    assert assembly_path.exists()
    assert deployment_path.exists()
    assert adapter_path.exists()
    assert "wrist_camera" in assembly_path.read_text(encoding="utf-8")
    assert "/wrist_camera/image_raw" in deployment_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_onboarding_stops_at_ros2_prerequisite_gate(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert "ROS2" in response.content
    assert state["stage"] == "resolve_prerequisites"
    assert not (tmp_path / "embodied" / "assemblies" / "so101_setup.py").exists()


@pytest.mark.asyncio
async def test_onboarding_refinement_updates_existing_setup(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/serial/by-id/usb-so101\n",
            "command -v ros2": "ROS2_OK\nROS_DISTRO=jazzy\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connect so101 and a wrist camera"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="add an overhead camera"),
        session,
    )

    assembly_text = (tmp_path / "embodied" / "assemblies" / "so101_setup.py").read_text(encoding="utf-8")
    deployment_text = (tmp_path / "embodied" / "deployments" / "so101_setup_real_local.py").read_text(encoding="utf-8")

    assert "ready" in response.content
    assert "wrist_camera" in assembly_text
    assert "overhead_camera" in assembly_text
    assert "/wrist_camera/image_raw" in deployment_text
    assert "/overhead_camera/image_raw" in deployment_text


@pytest.mark.asyncio
async def test_agent_loop_routes_first_run_setup_without_calling_provider(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    provider = DummyProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )
    loop.tools.unregister("exec")
    loop.tools.register(
        FakeExecTool(
            {
                "ls -1 /dev/serial/by-id": "/dev/serial/by-id/usb-so101\n",
                "command -v ros2": "ROS2_OK\nROS_DISTRO=jazzy\n",
            }
        )
    )

    response = await loop.process_direct("connect so101")

    assert "connected" in response
    assert provider.chat_calls == 0
