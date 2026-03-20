from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.base import Tool
from roboclaw.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.queue import MessageBus
from roboclaw.config.loader import CONFIG_PATH_ENV, set_config_path
from roboclaw.embodied.execution.integration.transports.ros2 import canonical_ros2_namespace
from roboclaw.embodied.execution.integration.control_surfaces.ros2.so101_feetech import (
    ADDR_HOMING_OFFSET,
    ADDR_LOCK,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    ADDR_OPERATING_MODE,
    ADDR_TORQUE_ENABLE,
    POSITION_MODE,
    SERVO_RESOLUTION_MAX,
    So101CalibrationMonitor,
)
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind
from roboclaw.embodied.execution.orchestration.runtime.executor import ProcedureExecutionResult, ProcedureExecutor
from roboclaw.embodied.execution.orchestration.runtime.manager import RuntimeManager
from roboclaw.embodied.execution.orchestration.runtime.model import RuntimeStatus
from roboclaw.embodied.onboarding import (
    SETUP_STATE_KEY,
    SetupOnboardingState,
    SetupStage,
    SetupStatus,
)
from roboclaw.providers.base import LLMResponse, ToolCallRequest
from roboclaw.session.manager import Session


@pytest.fixture(autouse=True)
def _framework_calibration_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    calibration_path = config_path.parent / "calibration" / "so101" / "so101_real.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config_path))
    set_config_path(config_path)
    try:
        yield
    finally:
        set_config_path(None)


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
    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.chat_calls = 0
        self.messages: list[list[dict[str, object]]] = []
        self.tools: list[list[dict[str, object]]] = []
        self.responses = list(responses or [LLMResponse(content="done")])

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.chat_calls += 1
        self.messages.append(list(kwargs.get("messages") or (args[0] if args else [])))
        self.tools.append(list(kwargs.get("tools") or []))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "openai-codex/gpt-5.4"


def _tool_call_response(name: str, arguments: dict[str, object], *, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def _last_tool_payload(provider: RecordingProvider, *, call_index: int = -1) -> dict[str, object]:
    tool_messages = [msg for msg in provider.messages[call_index] if msg.get("role") == "tool"]
    assert tool_messages
    return json.loads(str(tool_messages[-1]["content"]))


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
        "        'serial_device_by_id': '/dev/serial/by-id/usb-so101',",
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
                "        'primary': {'serial_device_by_id': '/dev/serial/by-id/usb-so101'},",
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
                "from roboclaw.embodied.execution.integration.control_surfaces import ARM_HAND_CONTROL_SURFACE_PROFILE",
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
                "            component=CompatibilityComponent.CONTROL_SURFACE_PROFILE,",
                "            target=ARM_HAND_CONTROL_SURFACE_PROFILE.id,",
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
                "    control_surface_profile_id=ARM_HAND_CONTROL_SURFACE_PROFILE.id,",
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


def _build_loop(
    tmp_path: Path,
    exec_responses: dict[str, str | list[str]],
    *,
    provider: RecordingProvider | None = None,
) -> tuple[AgentLoop, RecordingProvider, FakeExecTool]:
    provider = provider or RecordingProvider()
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
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Opened."),
        ]
    )
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 2
    assert response == "Opened."
    assert any(tool["function"]["name"] == "embodied_status" for tool in provider.tools[0])
    assert any(tool["function"]["name"] == "embodied_control" for tool in provider.tools[0])
    assert "[Embodied Context]" in str(provider.messages[0][-1]["content"])
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
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": expected_fragment if expected_fragment != "go_named_pose" else "go_named_pose", "primitive_args": {"name": "home"} if expected_fragment == "go_named_pose" else {}},
            ),
            LLMResponse(content="Done."),
        ]
    )
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    await loop.process_direct(message, session_key=session.key)

    assert provider.chat_calls == 2
    assert any(expected_fragment in call for call in fake_exec.calls)


@pytest.mark.asyncio
async def test_runtime_session_is_reused_across_multiple_commands(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response("embodied_control", {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}}),
            LLMResponse(content="Opened."),
            _tool_call_response("embodied_control", {"action": "run_primitive", "primitive_name": "gripper_close", "primitive_args": {}}),
            LLMResponse(content="Closed."),
        ]
    )
    loop, provider, fake_exec = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    await loop.process_direct("打开夹爪", session_key=session.key)
    await loop.process_direct("闭合夹爪", session_key=session.key)

    connect_calls = [call for call in fake_exec.calls if "ros2 service call" in call and "/connect" in call]

    assert provider.chat_calls == 4
    assert len(connect_calls) == 1


@pytest.mark.asyncio
async def test_session_active_setup_is_bound_and_visible_to_agent(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Opened."),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    await loop.process_direct("打开夹爪", session_key=session.key)

    assert session.metadata["embodied_active_setup"] == "so101_setup"
    snapshot = loop.embodied_execution.build_agent_snapshot(session)
    assert snapshot.active_setup_id == "so101_setup"
    assert '"active_setup_id": "so101_setup"' in str(provider.messages[0][-1]["content"])


@pytest.mark.asyncio
async def test_chinese_calibration_phrase_routes_to_calibration_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response("embodied_control", {"action": "calibrate"}),
            LLMResponse(content="calibration prompt"),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    async def fake_execute_calibrate(context, on_progress=None):
        assert context.setup_id == "so101_setup"
        return ProcedureExecutionResult(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message="calibration prompt",
            details={"calibration_phase": "await_mid_pose_ack"},
        )

    monkeypatch.setattr(loop.embodied_execution.executor, "execute_calibrate", fake_execute_calibrate)

    response = await loop.process_direct("我要标定", session_key=session.key)

    assert provider.chat_calls == 2
    assert response == "calibration prompt"
    assert session.metadata["embodied_calibration"]["phase"] == "await_mid_pose_ack"


@pytest.mark.asyncio
async def test_setup_ambiguity_prompts_for_clarification(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    _write_setup_assets(tmp_path, "lab_arm_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response("embodied_status", {}),
            LLMResponse(content="I found multiple embodied setups: so101_setup, lab_arm_setup. Tell me which setup id to use."),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)

    response = await loop.process_direct("打开夹爪", session_key="cli:ambiguous")

    assert provider.chat_calls == 2
    assert "so101_setup" in response
    assert "lab_arm_setup" in response
    assert any(token in response.lower() for token in ("which", "clarify", "哪个", "哪一个", "哪台"))


@pytest.mark.asyncio
async def test_direct_control_without_ready_setup_routes_into_onboarding(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    loop, provider, _ = _build_loop(tmp_path, {})

    response = await loop.process_direct("打开夹爪", session_key="cli:first_time")

    assert provider.chat_calls == 0
    session = loop.sessions.get_or_create("cli:first_time")
    assert SETUP_STATE_KEY in session.metadata
    assert any(token in response.lower() for token in ("robot", "setup", "so101", "connect", "机器人"))


@pytest.mark.asyncio
async def test_embodied_control_reports_needs_calibration_for_first_motion_failure(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_close", "primitive_args": {}},
            ),
            LLMResponse(content="Need calibration first."),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    config_path = tmp_path / "config.json"
    calibration_path = config_path.parent / "calibration" / "so101" / "so101_real.json"
    calibration_path.unlink()

    await loop.process_direct("闭合夹爪", session_key=session.key)

    payload = _last_tool_payload(provider)
    assert payload["ok"] is False
    assert payload["needs_calibration"] is True
    assert "calibrate" in payload["suggested_next_actions"]


@pytest.mark.asyncio
async def test_embodied_control_suggests_debug_after_non_calibration_failure(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Try debug."),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"), provider=provider)
    session = _seed_session(loop)

    async def fake_move(context, primitive_name: str, primitive_args=None, on_progress=None):
        assert primitive_name == "gripper_open"
        return ProcedureExecutionResult(
            procedure=ProcedureKind.MOVE,
            ok=False,
            message="Primitive rejected by adapter.",
            details={"error_code": "rejected"},
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(loop.embodied_execution.executor, "execute_move", fake_move)
    try:
        await loop.process_direct("打开夹爪", session_key=session.key)
    finally:
        monkeypatch.undo()

    payload = _last_tool_payload(provider)
    assert payload["ok"] is False
    assert payload["needs_calibration"] is False
    assert "debug" in payload["suggested_next_actions"]


@pytest.mark.asyncio
async def test_adapter_ignores_unrelated_ros_nodes_when_required_interfaces_are_missing(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    responses = _standard_ros2_responses("so101_setup")
    responses["ros2 node list"] = "/handretarget_node\n/linker_hand_sdk\n/some_other_robot_node\n"
    responses["ros2 service list"] = "/some_unrelated_service\n"
    responses["ros2 action list"] = "/some_other_robot_action\n"
    responses["ros2 topic list"] = "/cb_left_hand_state\n/cb_left_hand_control_cmd\n"
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Setup `so101_setup` is not ready yet: missing required ROS2 interfaces."),
        ]
    )
    loop, provider, _ = _build_loop(tmp_path, responses, provider=provider)
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 2
    assert "handretarget" not in response.lower()
    assert "linker_hand_sdk" not in response.lower()
    assert any(token in response.lower() for token in ("missing", "required", "unavailable", "not ready"))


@pytest.mark.asyncio
async def test_control_surface_profile_can_fallback_to_primitive_services_without_action(tmp_path: Path) -> None:
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
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Primitive `gripper_open` completed on setup `so101_setup`. Current gripper state: open."),
        ]
    )
    loop, provider, fake_exec = _build_loop(tmp_path, responses, provider=provider)
    session = _seed_session(loop)

    response = await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 2
    assert "Current gripper state" in response
    assert any("--field data" in call for call in fake_exec.calls)
    assert any("primitive_gripper_open" in call for call in fake_exec.calls)


@pytest.mark.asyncio
async def test_adapter_does_not_launch_duplicate_runtime_when_connect_service_exists(tmp_path: Path) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup", launch_command="python -m fake_control_surface")
    responses = _standard_ros2_responses("so101_setup")
    provider = RecordingProvider(
        responses=[
            _tool_call_response(
                "embodied_control",
                {"action": "run_primitive", "primitive_name": "gripper_open", "primitive_args": {}},
            ),
            LLMResponse(content="Opened."),
        ]
    )
    loop, provider, fake_exec = _build_loop(tmp_path, responses, provider=provider)
    session = _seed_session(loop)

    await loop.process_direct("打开夹爪", session_key=session.key)

    assert provider.chat_calls == 2
    assert not any("nohup bash -lc" in call for call in fake_exec.calls)


def test_state_confirmation_treats_real_so101_open_position_as_open() -> None:
    message = ProcedureExecutor._state_confirmation(
        "gripper_open",
        {"gripper_percent": 76.6},
        {"gripper_percent": 76.6},
    )

    assert "Current gripper state: open" in message


def _execution_context(
    tmp_path: Path,
    *,
    calibration_exists: bool,
    runtime_status: RuntimeStatus = RuntimeStatus.DISCONNECTED,
):
    calibration_path = tmp_path / "scenario-calibration" / "so101" / "so101_real.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    if calibration_exists:
        calibration_path.write_text("{}", encoding="utf-8")

    runtime_manager = RuntimeManager()
    executor = ProcedureExecutor(ToolRegistry(), runtime_manager)
    runtime = runtime_manager.create(
        session_id="cli:test:so101_setup",
        assembly_id="so101_setup",
        deployment_id="so101_setup_real_local",
        target_id="real",
        adapter_id="so101_setup_ros2_local",
    )
    runtime.status = runtime_status
    context = SimpleNamespace(
        setup_id="so101_setup",
        assembly=SimpleNamespace(id="so101_setup"),
        deployment=SimpleNamespace(
            id="so101_setup_real_local",
            target_id="real",
            connection={"serial_device_by_id": "/dev/serial/by-id/usb-so101"},
            robots={"primary": {"serial_device_by_id": "/dev/serial/by-id/usb-so101"}},
        ),
        target=SimpleNamespace(id="real"),
        robot=SimpleNamespace(id="so101"),
        adapter_binding=SimpleNamespace(id="so101_setup_ros2_local"),
        profile=SimpleNamespace(
            robot_id="so101",
            requires_calibration=True,
            canonical_calibration_path=lambda: calibration_path,
        ),
        runtime=runtime,
    )
    return executor, context, calibration_path


@pytest.mark.asyncio
async def test_first_move_requires_calibration_before_connect(tmp_path: Path) -> None:
    executor, context, calibration_path = _execution_context(tmp_path, calibration_exists=False)

    result = await executor.execute_move(context, primitive_name="gripper_close")

    assert result.ok is False
    assert "needs calibration before `connect` or motion" in result.message
    assert str(calibration_path) in result.message
    assert "Reply with `calibrate`" in result.message
    assert context.runtime.status == RuntimeStatus.ERROR


@pytest.mark.asyncio
async def test_follow_on_move_skips_calibration_recheck_once_runtime_is_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor, context, _ = _execution_context(
        tmp_path,
        calibration_exists=False,
        runtime_status=RuntimeStatus.READY,
    )

    class FakeAdapter:
        async def get_state(self):
            return SimpleNamespace(values={"gripper_percent": 20.0})

        async def execute_primitive(self, primitive_name: str, primitive_args: dict[str, object]):
            assert primitive_name == "gripper_close"
            assert primitive_args == {}
            return SimpleNamespace(
                accepted=True,
                completed=True,
                message="done",
                error_code=None,
                output={},
            )

    monkeypatch.setattr(executor, "_adapter", lambda _: FakeAdapter())

    result = await executor.execute_move(context, primitive_name="gripper_close")

    assert result.ok is True
    assert "gripper_close" in result.message


@pytest.mark.asyncio
async def test_calibrate_prompts_for_mid_pose_then_saves_on_second_enter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor, context, calibration_path = _execution_context(tmp_path, calibration_exists=False)
    progress: list[str] = []

    class FakeMonitor:
        def __init__(self) -> None:
            self.snapshot_calls = 0

        def connect(self) -> None:
            return None

        def prepare_manual_calibration(self) -> None:
            return None

        def capture_mid_pose(self) -> dict[str, int]:
            return {
                "shoulder_pan": 2050,
                "shoulder_lift": 2051,
                "elbow_flex": 2052,
                "wrist_flex": 2053,
                "wrist_roll": 2054,
                "gripper": 2055,
            }

        def apply_half_turn_homings(self, mid_pose_raw: dict[str, int]) -> dict[str, int]:
            return {joint_name: raw - 2047 for joint_name, raw in mid_pose_raw.items()}

        def start_observation(self):
            rows = (
                SimpleNamespace(joint_name="shoulder_pan", servo_id=1, range_min_raw=512, position_raw=512, range_max_raw=512),
                SimpleNamespace(joint_name="shoulder_lift", servo_id=2, range_min_raw=612, position_raw=612, range_max_raw=612),
                SimpleNamespace(joint_name="elbow_flex", servo_id=3, range_min_raw=712, position_raw=712, range_max_raw=712),
                SimpleNamespace(joint_name="wrist_flex", servo_id=4, range_min_raw=812, position_raw=812, range_max_raw=812),
                SimpleNamespace(joint_name="wrist_roll", servo_id=5, range_min_raw=912, position_raw=912, range_max_raw=912),
                SimpleNamespace(joint_name="gripper", servo_id=6, range_min_raw=2100, position_raw=2100, range_max_raw=2100),
            )
            return SimpleNamespace(
                device_by_id="/dev/serial/by-id/usb-so101",
                resolved_device="/dev/ttyACM0",
                rows=rows,
            )

        def snapshot_observed(self):
            self.snapshot_calls += 1
            rows = (
                SimpleNamespace(joint_name="shoulder_pan", servo_id=1, range_min_raw=120, position_raw=512, range_max_raw=980),
                SimpleNamespace(joint_name="shoulder_lift", servo_id=2, range_min_raw=220, position_raw=612, range_max_raw=1080),
                SimpleNamespace(joint_name="elbow_flex", servo_id=3, range_min_raw=320, position_raw=712, range_max_raw=1180),
                SimpleNamespace(joint_name="wrist_flex", servo_id=4, range_min_raw=420, position_raw=812, range_max_raw=1280),
                SimpleNamespace(joint_name="wrist_roll", servo_id=5, range_min_raw=520, position_raw=912, range_max_raw=1380),
                SimpleNamespace(joint_name="gripper", servo_id=6, range_min_raw=1900, position_raw=2100, range_max_raw=3600),
            )
            return SimpleNamespace(
                device_by_id="/dev/serial/by-id/usb-so101",
                resolved_device="/dev/ttyACM0",
                rows=rows,
            )

        def export_calibration_payload(self) -> dict[str, dict[str, int]]:
            return {
                "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 3, "range_min": 120, "range_max": 980},
                "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 4, "range_min": 220, "range_max": 1080},
                "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 5, "range_min": 320, "range_max": 1180},
                "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 6, "range_min": 420, "range_max": 1280},
                "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 7, "range_min": 520, "range_max": 1380},
                "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 8, "range_min": 1900, "range_max": 3600},
            }

        def disconnect(self) -> None:
            return None

    fake_monitor = FakeMonitor()
    monkeypatch.setattr(executor, "_build_so101_calibration_monitor", lambda _: fake_monitor)
    monkeypatch.setattr(executor, "_so101_calibration_stream_settings", lambda: (0.0, 0.0, None))

    async def on_progress(content: str) -> None:
        progress.append(content)

    prompt = await executor.execute_calibrate(context, on_progress=on_progress)
    started = await executor.advance_calibration(context, on_progress=on_progress)
    await asyncio.sleep(0.01)
    saved = await executor.advance_calibration(context, on_progress=on_progress)

    assert prompt.ok is False
    assert "middle pose" in prompt.message
    assert started.ok is False
    assert "press Enter again to stop and save" in started.message
    assert saved.ok is True
    assert str(calibration_path) in saved.message
    assert calibration_path.exists()
    assert progress
    assert "SO101 calibration live view on" in progress[0]
    assert "gripper           6    2100   2100   2100" in progress[0]
    assert "shoulder_pan" in progress[0]
    assert fake_monitor.snapshot_calls >= 1
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert payload["gripper"]["range_min"] == 1900
    assert executor.calibration_phase(context.runtime.id) is None


@pytest.mark.asyncio
async def test_calibrate_allows_overwriting_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor, context, calibration_path = _execution_context(tmp_path, calibration_exists=True)

    class FakeMonitor:
        def connect(self) -> None:
            return None

        def prepare_manual_calibration(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(executor, "_build_so101_calibration_monitor", lambda _: FakeMonitor())

    prompt = await executor.execute_calibrate(context)

    assert prompt.ok is False
    assert "overwrite the existing calibration file" in prompt.message
    assert str(calibration_path) in prompt.message
    assert executor.calibration_phase(context.runtime.id) == "await_mid_pose_ack"


@pytest.mark.asyncio
async def test_calibrate_disconnects_active_runtime_before_monitor_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, context, _ = _execution_context(
        tmp_path,
        calibration_exists=True,
        runtime_status=RuntimeStatus.READY,
    )
    disconnect_calls: list[str] = []

    class FakeAdapter:
        async def disconnect(self):
            disconnect_calls.append("disconnect")
            return SimpleNamespace(ok=True)

    class FakeMonitor:
        def connect(self) -> None:
            return None

        def prepare_manual_calibration(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    executor._adapters[context.runtime.id] = FakeAdapter()
    monkeypatch.setattr(executor, "_build_so101_calibration_monitor", lambda _: FakeMonitor())

    prompt = await executor.execute_calibrate(context)

    assert prompt.ok is False
    assert disconnect_calls == ["disconnect"]
    assert executor.calibration_phase(context.runtime.id) == "await_mid_pose_ack"


@pytest.mark.asyncio
async def test_calibration_start_failure_returns_friendly_retry_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, context, _ = _execution_context(tmp_path, calibration_exists=True)

    class FakeMonitor:
        def connect(self) -> None:
            return None

        def prepare_manual_calibration(self) -> None:
            return None

        def capture_mid_pose(self) -> dict[str, int]:
            raise RuntimeError("read 0x38 for servo 1 failed: [TxRxResult] There is no status packet!")

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(executor, "_build_so101_calibration_monitor", lambda _: FakeMonitor())

    prompt = await executor.execute_calibrate(context)
    started = await executor.advance_calibration(context)

    assert prompt.ok is False
    assert started.ok is False
    assert "the arm did not answer" in started.message
    assert "reply `calibrate` again" in started.message
    assert "reply `connect`" in started.message
    assert "There is no status packet" not in started.message
    assert started.details["raw_error"] == "read 0x38 for servo 1 failed: [TxRxResult] There is no status packet!"
    assert executor.calibration_phase(context.runtime.id) is None


def test_prepare_manual_calibration_resets_homing_and_limits() -> None:
    calls: list[tuple[str, int, int, int]] = []

    class FakeBus:
        def write_byte(self, servo_id: int, address: int, value: int) -> None:
            calls.append(("byte", servo_id, address, value))

        def write_word(self, servo_id: int, address: int, value: int) -> None:
            calls.append(("word", servo_id, address, value))

    monitor = So101CalibrationMonitor(device_by_id="/dev/serial/by-id/fake")
    monitor._bus = FakeBus()  # type: ignore[assignment]
    monitor._mid_pose_raw = {"shoulder_pan": 1}
    monitor._homing_offsets = {"shoulder_pan": 2}
    monitor._observed_mins = {"shoulder_pan": 3}
    monitor._observed_maxs = {"shoulder_pan": 4}

    monitor.prepare_manual_calibration()

    assert ("byte", 1, ADDR_LOCK, 0) in calls
    assert ("byte", 1, ADDR_TORQUE_ENABLE, 0) in calls
    assert ("byte", 1, ADDR_OPERATING_MODE, POSITION_MODE) in calls
    assert ("word", 1, ADDR_HOMING_OFFSET, 0) in calls
    assert ("word", 1, ADDR_MIN_POSITION_LIMIT, 0) in calls
    assert ("word", 1, ADDR_MAX_POSITION_LIMIT, SERVO_RESOLUTION_MAX) in calls
    assert monitor._mid_pose_raw == {}
    assert monitor._homing_offsets == {}
    assert monitor._observed_mins == {}
    assert monitor._observed_maxs == {}


@pytest.mark.asyncio
async def test_pending_calibration_blank_message_advances_without_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_workspace(tmp_path)
    _write_setup_assets(tmp_path, "so101_setup")
    loop, provider, _ = _build_loop(tmp_path, _standard_ros2_responses("so101_setup"))
    session = _seed_session(loop)
    runtime_id = f"{session.key}:so101_setup"
    session.metadata["embodied_calibration"] = {
        "setup_id": "so101_setup",
        "runtime_id": runtime_id,
        "phase": "await_mid_pose_ack",
    }
    loop.sessions.save(session)

    async def fake_advance(context, on_progress=None):
        assert context.runtime.id == runtime_id
        return ProcedureExecutionResult(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message="started live calibration",
            details={"calibration_phase": "streaming"},
        )

    monkeypatch.setattr(loop.embodied_execution.executor, "advance_calibration", fake_advance)

    response = await loop.process_direct("", session_key=session.key)

    assert provider.chat_calls == 0
    assert response == "started live calibration"
