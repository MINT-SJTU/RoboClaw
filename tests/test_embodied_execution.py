from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.base import Tool
from roboclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.bus.queue import MessageBus
from roboclaw.embodied.execution.integration.transports.ros2 import canonical_ros2_namespace
from roboclaw.embodied.execution.orchestration.runtime.executor import ProcedureExecutor
from roboclaw.embodied.onboarding import (
    SETUP_STATE_KEY,
    SetupOnboardingState,
    SetupStage,
    SetupStatus,
)
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
        return "Fake exec tool for embodied execution tests."

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


class RecordingProvider:
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


def _write_setup_assets(root: Path, setup_id: str, *, launch_command: str | None = None) -> None:
    namespace = canonical_ros2_namespace(setup_id, "real")
    assembly_path = root / "embodied" / "assemblies" / f"{setup_id}.py"
    deployment_path = root / "embodied" / "deployments" / f"{setup_id}_real_local.py"
    adapter_path = root / "embodied" / "adapters" / f"{setup_id}_ros2_local.py"

    assembly_path.write_text(
        "\n".join(
            [
                '"""Workspace-generated embodied assembly for tests."""',
                "",
                "from roboclaw.embodied.definition.systems.assemblies import AssemblyBlueprint, RobotAttachment",
                "from roboclaw.embodied.execution.integration.carriers.real import build_real_ros2_target",
                "from roboclaw.embodied.execution.integration.transports.ros2 import build_standard_ros2_contract",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ASSEMBLY,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ASSEMBLY,",
                ")",
                "",
                "ASSEMBLY = AssemblyBlueprint(",
                f"    id={setup_id!r},",
                f"    name={setup_id!r},",
                '    description="Test assembly",',
                "    robots=(",
                "        RobotAttachment(",
                '            attachment_id="primary",',
                '            robot_id="so101",',
                "        ),",
                "    ),",
                "    sensors=(),",
                "    execution_targets=(",
                "        build_real_ros2_target(",
                '            target_id="real",',
                '            description="Real target",',
                f"            ros2=build_standard_ros2_contract({setup_id!r}, 'real'),",
                "        ),",
                "    ),",
                '    default_execution_target_id="real",',
                "    frame_transforms=(),",
                "    tools=(),",
                "    control_groups=(),",
                "    safety_zones=(),",
                "    safety_boundaries=(),",
                "    failure_domains=(),",
                "    resource_ownerships=(),",
                ").build()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    connection_lines = [
        "        'transport': 'ros2',",
        "        'ros_distro': 'jazzy',",
        "        'profile_id': 'so101_ros2_standard',",
        f"        'namespace': {namespace!r},",
    ]
    if launch_command is not None:
        connection_lines.append(f"        'launch_command': {launch_command!r},")

    deployment_path.write_text(
        "\n".join(
            [
                '"""Workspace-generated deployment profile for tests."""',
                "",
                "from roboclaw.embodied.definition.systems.deployments import DeploymentProfile",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.DEPLOYMENT,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.DEPLOYMENT,",
                ")",
                "",
                "DEPLOYMENT = DeploymentProfile(",
                f"    id={setup_id + '_real_local'!r},",
                f"    assembly_id={setup_id!r},",
                '    target_id="real",',
                "    connection={",
                *connection_lines,
                "    },",
                "    robots={",
                "        'primary': {'serial_device': '/dev/ttyACM0'},",
                "    },",
                "    sensors={},",
                "    safety_overrides={},",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
    adapter_path.write_text(
        "\n".join(
            [
                '"""Workspace-generated adapter binding for tests."""',
                "",
                "from roboclaw.embodied.definition.foundation.schema import TransportKind",
                "from roboclaw.embodied.execution.integration.adapters import (",
                "    AdapterBinding,",
                "    AdapterCompatibilitySpec,",
                "    CompatibilityComponent,",
                "    VersionConstraint,",
                ")",
                "from roboclaw.embodied.execution.integration.bridges import ARM_HAND_BRIDGE",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ADAPTER,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ADAPTER,",
                ")",
                "",
                "COMPATIBILITY = AdapterCompatibilitySpec(",
                "    constraints=(",
                "        VersionConstraint(",
                "            component=CompatibilityComponent.TRANSPORT,",
                "            target='ros2',",
                "            requirement='>=1.0,<2.0',",
                "        ),",
                "        VersionConstraint(",
                "            component=CompatibilityComponent.BRIDGE,",
                "            target=ARM_HAND_BRIDGE.id,",
                "            requirement='>=1.0,<2.0',",
                "        ),",
                "    ),",
                ")",
                "",
                "ADAPTER = AdapterBinding(",
                f"    id={setup_id + '_ros2_local'!r},",
                f"    assembly_id={setup_id!r},",
                "    transport=TransportKind.ROS2,",
                "    implementation=",
                "        'roboclaw.embodied.execution.integration.adapters.ros2.standard:Ros2ActionServiceAdapter',",
                "    supported_targets=('real',),",
                "    bridge_id=ARM_HAND_BRIDGE.id,",
                "    compatibility=COMPATIBILITY,",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _seed_ready_setup(session: Session, setup_id: str = "so101_setup") -> None:
    state = SetupOnboardingState(
        setup_id=setup_id,
        intake_slug=setup_id,
        assembly_id=setup_id,
        deployment_id=f"{setup_id}_real_local",
        adapter_id=f"{setup_id}_ros2_local",
        stage=SetupStage.HANDOFF_READY,
        status=SetupStatus.READY,
        robot_attachments=[{"attachment_id": "primary", "robot_id": "so101", "role": "primary"}],
        execution_targets=[{"id": "real", "carrier": "real"}],
        detected_facts={"connected": True, "ros2_available": True, "ros2_distro": "jazzy"},
    )
    session.metadata[SETUP_STATE_KEY] = state.to_dict()


def _build_loop(tmp_path: Path, exec_responses: dict[str, str | list[str]]) -> tuple[AgentLoop, RecordingProvider, FakeExecTool]:
    provider = RecordingProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
    )
    loop.tools.unregister("exec")
    fake_exec = FakeExecTool(exec_responses)
    loop.tools.register(fake_exec)
    for cls in (ReadFileTool, WriteFileTool, ListDirTool):
        if not loop.tools.has(cls(workspace=tmp_path).name):
            loop.tools.register(cls(workspace=tmp_path))
    return loop, provider, fake_exec


def _seed_session(loop: AgentLoop, setup_id: str = "so101_setup", session_key: str = "cli:embodied") -> Session:
    session = Session(key=session_key)
    _seed_ready_setup(session, setup_id=setup_id)
    loop.sessions.save(session)
    return session


def _standard_ros2_responses(setup_id: str) -> dict[str, str]:
    namespace = canonical_ros2_namespace(setup_id, "real")
    return {
        "ros2 service list": "\n".join(
            [
                f"{namespace}/connect",
                f"{namespace}/disconnect",
                f"{namespace}/stop",
                f"{namespace}/reset",
                f"{namespace}/recover",
                f"{namespace}/debug_snapshot",
            ]
        ),
        "ros2 action list": "\n".join(
            [
                f"{namespace}/execute_primitive",
                f"{namespace}/start_calibration",
            ]
        ),
        "ros2 topic list": "\n".join(
            [
                f"{namespace}/state",
                f"{namespace}/health",
                f"{namespace}/events",
            ]
        ),
        f"ros2 service call {namespace}/connect": "success: true\nmessage: connected\n",
        f"ros2 service call {namespace}/reset": "success: true\nmessage: reset\n",
        f"ros2 service call {namespace}/stop": "success: true\nmessage: stopped\n",
        f"ros2 service call {namespace}/recover": "success: true\nmessage: recovered\n",
        f"ros2 service call {namespace}/debug_snapshot": "success: true\nartifact: /tmp/debug.json\n",
        f"ros2 topic echo --once {namespace}/state": "status: ready\n",
        f"ros2 action send_goal {namespace}/execute_primitive": "Goal accepted.\nresult:\n  status: succeeded\n",
    }


@pytest.mark.asyncio
async def test_ready_session_routes_embodied_commands_without_provider(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"))
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 0
    assert "provider should not be called" not in response
    assert any("execute_primitive" in call for call in fake_exec.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_fragment"),
    [
        ("打开夹爪", "gripper_open"),
        ("闭合夹爪", "gripper_close"),
        ("回到 home", "go_named_pose"),
    ],
)
async def test_chinese_aliases_normalize_to_expected_primitives(
    tmp_path: Path,
    message: str,
    expected_fragment: str,
) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"))
    session = _seed_session(loop)

    await loop.process_direct(message, session_key=session.key)

    assert provider.chat_calls == 0
    assert any(expected_fragment in call for call in fake_exec.calls)


@pytest.mark.asyncio
async def test_runtime_session_is_reused_across_multiple_commands(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"))
    session = _seed_session(loop)

    await loop.process_direct("打开夹爪", session_key=session.key)
    await loop.process_direct("闭合夹爪", session_key=session.key)

    connect_calls = [call for call in fake_exec.calls if "ros2 service call" in call and "/connect" in call]

    assert provider.chat_calls == 0
    assert len(connect_calls) == 1


@pytest.mark.asyncio
async def test_setup_ambiguity_prompts_for_clarification(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    _write_setup_assets(tmp_path, "lab_arm_setup")
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"))

    response = await loop.process_direct("打开夹爪", session_key="cli:ambiguous")

    assert provider.chat_calls == 0
    assert "so101_setup" in response
    assert "lab_arm_setup" in response
    assert any(token in response.lower() for token in ("which", "clarify", "哪个", "哪一个", "哪台"))


@pytest.mark.asyncio
async def test_adapter_ignores_unrelated_ros_nodes_when_required_interfaces_are_missing(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    responses = _standard_ros2_responses("so101_setup")
    responses["ros2 node list"] = "/handretarget_node\n/linker_hand_sdk\n/some_other_robot_node\n"
    responses["ros2 service list"] = "/some_unrelated_service\n"
    responses["ros2 action list"] = "/some_other_robot_action\n"
    responses["ros2 topic list"] = "/cb_left_hand_state\n/cb_left_hand_control_cmd\n"
    loop, provider, _ = _build_loop(tmp_path, responses)
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 0
    assert "handretarget" not in response.lower()
    assert "linker_hand_sdk" not in response.lower()
    assert any(token in response.lower() for token in ("missing", "required", "unavailable", "not ready"))


@pytest.mark.asyncio
async def test_stage1_profile_can_fallback_to_primitive_services_without_action(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    namespace = canonical_ros2_namespace("so101_setup", "real")
    responses = _standard_ros2_responses("so101_setup")
    responses["ros2 action list"] = ""
    responses["ros2 service list"] = "\n".join(
        [
            f"{namespace}/connect",
            f"{namespace}/disconnect",
            f"{namespace}/stop",
            f"{namespace}/reset",
            f"{namespace}/recover",
            f"{namespace}/debug_snapshot",
            f"{namespace}/primitive_gripper_open",
            f"{namespace}/primitive_gripper_close",
            f"{namespace}/primitive_go_home",
        ]
    )
    responses[f"ros2 service type {namespace}/connect"] = "std_srvs/srv/Trigger\n"
    responses[f"ros2 service type {namespace}/primitive_gripper_open"] = "std_srvs/srv/Trigger\n"
    responses[f"ros2 service call {namespace}/connect"] = "response:\nstd_srvs.srv.Trigger_Response(success=True, message='connected')\n"
    responses[f"ros2 service call {namespace}/primitive_gripper_open"] = (
        "response:\nstd_srvs.srv.Trigger_Response(success=True, message='gripper opened')\n"
    )
    responses[f"ros2 topic echo --once --field data {namespace}/state"] = (
        '{"connected": true, "gripper_percent": 100.0}\n---\n'
    )
    responses[f"ros2 topic echo --once {namespace}/state"] = (
        "data: '{\"connected\": true, \"gripper_percent\": 100.0}'\n---\n"
    )
    loop, provider, fake_exec = _build_loop(tmp_path, responses)
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 0
    assert "Current gripper state" in response
    assert any("--field data" in call for call in fake_exec.calls)
    assert any("primitive_gripper_open" in call for call in fake_exec.calls)


@pytest.mark.asyncio
async def test_adapter_does_not_launch_duplicate_runtime_when_connect_service_exists(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup", launch_command="python -m fake_bridge")
    responses = _standard_ros2_responses("so101_setup")
    loop, provider, fake_exec = _build_loop(tmp_path, responses)
    session = _seed_session(loop)

    await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 0
    assert not any("nohup bash -lc" in call for call in fake_exec.calls)


def test_state_confirmation_treats_real_so101_open_position_as_open() -> None:
    message = ProcedureExecutor._state_confirmation(
        "gripper_open",
        {"gripper_percent": 76.6},
        {"gripper_percent": 76.6},
    )

    assert "Current gripper state: open" in message
