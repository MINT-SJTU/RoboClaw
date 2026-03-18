from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.base import Tool
from roboclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage
from roboclaw.bus.queue import MessageBus
from roboclaw.embodied.onboarding import SETUP_STATE_KEY, OnboardingController, SetupOnboardingState, SetupStage, SetupStatus
from roboclaw.providers.base import LLMResponse
from roboclaw.session.manager import Session


class FakeExecTool(Tool):
    def __init__(self, responses: dict[str, str | list[str]]):
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
                if isinstance(result, list):
                    if len(result) > 1:
                        return result.pop(0)
                    return result[0]
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


def _build_tools(workspace: Path, exec_responses: dict[str, str | list[str]]) -> tuple[ToolRegistry, FakeExecTool]:
    registry = ToolRegistry()
    for cls in (ReadFileTool, WriteFileTool, ListDirTool):
        registry.register(cls(workspace=workspace))
    fake_exec = FakeExecTool(exec_responses)
    registry.register(fake_exec)
    return registry, fake_exec


def test_onboarding_routes_chinese_real_robot_request(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    controller = OnboardingController(tmp_path, ToolRegistry())
    session = Session(key="cli:direct")

    assert controller.should_handle(session, "我想用一个真实的机器人")


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
    deployment_text = deployment_path.read_text(encoding="utf-8")
    assert "/wrist_camera/image_raw" in deployment_text
    assert "stage1_server" in deployment_text
    assert "--profile-id so101_ros2_standard" in deployment_text
    assert "/usr/bin/python3 -m roboclaw.embodied.execution.integration.bridges.ros2.stage1_server" in deployment_text


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
async def test_onboarding_blocks_unknown_stage1_profile_before_asset_generation(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(tmp_path, {})
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")
    session.metadata[SETUP_STATE_KEY] = SetupOnboardingState(
        setup_id="custom_setup",
        intake_slug="custom_setup",
        assembly_id="custom_setup",
        deployment_id="custom_setup_real_local",
        adapter_id="custom_setup_ros2_local",
        stage=SetupStage.IDENTIFY_SETUP_SCOPE,
        status=SetupStatus.BOOTSTRAPPING,
        robot_attachments=[{"attachment_id": "primary", "robot_id": "custom_arm", "role": "primary"}],
        detected_facts={"connected": True},
    ).to_dict()

    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="continue"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "identify_setup_scope"
    assert state["missing_facts"] == ["stage1_execution_profile"]
    assert "does not have a framework ROS2 stage-1 execution profile" in response.content
    assert not (tmp_path / "embodied" / "assemblies" / "custom_setup.py").exists()


@pytest.mark.asyncio
async def test_onboarding_accepts_chinese_connected_confirmation(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\nPRETTY_NAME=Ubuntu 22.04 LTS\nSHELL_NAME=bash\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="SO101"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="都连好了"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "resolve_prerequisites"
    assert state["detected_facts"]["connected"] is True
    assert "ROS2" in response.content


@pytest.mark.asyncio
async def test_onboarding_starts_guided_ros2_install_flow(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=noble\nPRETTY_NAME=Ubuntu 24.04 LTS\nSHELL_NAME=zsh\nCONDA_PREFIX=\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="start ROS2 install"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "install_prerequisites"
    assert state["detected_facts"]["ros2_install_recipe"] == "jazzy"
    assert state["detected_facts"]["ros2_install_step_index"] == 0
    assert "Current step: `1` of `4`." in response.content
    assert "sudo add-apt-repository universe" in response.content
    assert "source /opt/ros/jazzy/setup.zsh" not in response.content
    assert "Reply with `done`" in response.content


@pytest.mark.asyncio
async def test_onboarding_advances_guided_ros2_install_steps(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\nPRETTY_NAME=Ubuntu 22.04 LTS\nSHELL_NAME=bash\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="need desktop tools and start ROS2 install"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="done"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "install_prerequisites"
    assert state["detected_facts"]["ros2_install_step_index"] == 1
    assert "Current step: `2` of `4`." in response.content
    assert "ros-humble-desktop" in response.content
    assert "sudo apt upgrade -y" in response.content


@pytest.mark.asyncio
async def test_onboarding_does_not_advance_ros2_install_on_continue_request(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\nPRETTY_NAME=Ubuntu 22.04 LTS\nSHELL_NAME=bash\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="start ROS2 install"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="continue ros2 install"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "install_prerequisites"
    assert state["detected_facts"]["ros2_install_step_index"] == 0
    assert "Current step: `1` of `4`." in response.content
    assert "sudo add-apt-repository universe" in response.content


@pytest.mark.asyncio
async def test_onboarding_resumes_after_manual_ros2_install_report(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": [
                "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
                "ROS2_OK\nros2 0.0.0\nROS_DISTRO=humble\n",
            ],
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\nPRETTY_NAME=Ubuntu 22.04 LTS\nSHELL_NAME=bash\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
        },
    )
    controller = OnboardingController(tmp_path, tools)
    session = Session(key="cli:direct")

    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="so101"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="connected"),
        session,
    )
    await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="start ROS2 install"),
        session,
    )
    response = await controller.handle_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="ROS2 installed"),
        session,
    )

    state = session.metadata[SETUP_STATE_KEY]
    assert state["stage"] == "handoff_ready"
    assert "ready" in response.content
    assert (tmp_path / "embodied" / "assemblies" / "so101_setup.py").exists()


@pytest.mark.asyncio
async def test_onboarding_keeps_partial_opt_ros_install_in_prerequisite_flow(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_PRESENT\nINSTALLED_DISTROS=jazzy\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=noble\nPRETTY_NAME=Ubuntu 24.04 LTS\nSHELL_NAME=bash\nCONDA_PREFIX=\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
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
    assert state["stage"] == "resolve_prerequisites"
    assert state["detected_facts"]["ros2_available"] is False
    assert state["detected_facts"]["ros2_shell_initialized"] is False
    assert "partial install" in response.content
    assert "source /opt/ros/jazzy/setup.bash" in response.content


@pytest.mark.asyncio
async def test_onboarding_does_not_treat_ros1_install_as_ros2_available(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_MISSING\nROS2_SHELL_INIT=0\n",
            "printf \"ID=%s\\n\"": "ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=noble\nPRETTY_NAME=Ubuntu 24.04 LTS\nSHELL_NAME=bash\nWSL=0\nSUDO=1\nSUDO_PASSWORDLESS=0\n",
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
    assert state["stage"] == "resolve_prerequisites"
    assert state["detected_facts"]["ros2_available"] is False
    assert "start ROS2 install" in response.content


@pytest.mark.asyncio
async def test_onboarding_accepts_installed_ros2_when_shell_init_is_configured(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    tools, _ = _build_tools(
        tmp_path,
        {
            "ls -1 /dev/serial/by-id": "/dev/ttyACM0\n",
            "command -v ros2": "ROS2_PRESENT\nINSTALLED_DISTROS=humble\nROS2_SHELL_INIT=1\n",
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
    assert state["stage"] == "handoff_ready"
    assert state["detected_facts"]["ros2_available"] is False
    assert state["detected_facts"]["ros2_shell_initialized"] is True
    assert "ready" in response.content
    assert "start ROS2 install" not in response.content


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
