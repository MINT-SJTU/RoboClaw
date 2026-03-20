"""Embodied execution controller and agent-facing session service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.catalog import build_catalog
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind
from roboclaw.embodied.execution.orchestration.runtime.executor import (
    ExecutionContext,
    ProcedureExecutionResult,
    ProcedureExecutor,
)
from roboclaw.embodied.execution.orchestration.runtime.manager import RuntimeManager
from roboclaw.embodied.onboarding import SETUP_STATE_KEY, SetupOnboardingState
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]

EMBODIED_RUNTIME_STATE_KEY = "embodied_runtime"
EMBODIED_CALIBRATION_STATE_KEY = "embodied_calibration"
EMBODIED_ACTIVE_SETUP_KEY = "embodied_active_setup"


@dataclass(frozen=True)
class ResolvedSetup:
    """Runtime-ready setup selection."""

    setup_id: str
    assembly_id: str
    deployment_id: str
    adapter_id: str
    source: str


@dataclass(frozen=True)
class EmbodiedAgentSnapshot:
    """Embodied session/workspace snapshot exposed to the agent."""

    active_setup_id: str | None
    session_setup_id: str | None
    selected_setup_id: str | None
    needs_user_choice: bool
    candidates: tuple[dict[str, Any], ...] = ()
    pending_calibration_phase: str | None = None
    runtime_status: str | None = None
    last_error: str | None = None
    robot_id: str | None = None
    robot_name: str | None = None
    target_id: str | None = None
    transport: str | None = None
    profile_id: str | None = None
    capability_families: tuple[str, ...] = ()
    supported_primitives: tuple[str, ...] = ()
    primitive_alias_examples: dict[str, tuple[str, ...]] = field(default_factory=dict)
    calibration_required: bool | None = None
    calibration_present: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_setup_id": self.active_setup_id,
            "session_setup_id": self.session_setup_id,
            "selected_setup_id": self.selected_setup_id,
            "needs_user_choice": self.needs_user_choice,
            "candidates": list(self.candidates),
            "pending_calibration_phase": self.pending_calibration_phase,
            "runtime_status": self.runtime_status,
            "last_error": self.last_error,
            "robot_id": self.robot_id,
            "robot_name": self.robot_name,
            "target_id": self.target_id,
            "transport": self.transport,
            "profile_id": self.profile_id,
            "capability_families": list(self.capability_families),
            "supported_primitives": list(self.supported_primitives),
            "primitive_alias_examples": {
                key: list(values) for key, values in self.primitive_alias_examples.items()
            },
            "calibration_required": self.calibration_required,
            "calibration_present": self.calibration_present,
        }


@dataclass(frozen=True)
class EmbodiedToolResult:
    """Normalized tool result returned to the main agent."""

    ok: bool
    action: str
    setup_id: str | None
    runtime_status: str | None
    message: str
    needs_user_choice: bool = False
    needs_calibration: bool = False
    external_intervention_required: bool = False
    suggested_next_actions: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "setup_id": self.setup_id,
            "runtime_status": self.runtime_status,
            "message": self.message,
            "needs_user_choice": self.needs_user_choice,
            "needs_calibration": self.needs_calibration,
            "external_intervention_required": self.external_intervention_required,
            "suggested_next_actions": list(self.suggested_next_actions),
            "details": self.details,
        }


class EmbodiedExecutionController:
    """Agent-facing embodied execution service with strong procedure execution."""

    _GENERIC_HINTS = (
        "robot",
        "arm",
        "gripper",
        "calibrate",
        "calibration",
        "reset",
        "debug",
        "diagnose",
        "status",
        "connect",
        "disconnect",
        "home",
        "pose",
        "move",
        "joint",
        "servo",
        "夹爪",
        "机械臂",
        "机器人",
        "连接",
        "校准",
        "标定",
        "复位",
        "回零",
        "回到",
        "状态",
        "诊断",
    )

    def __init__(self, workspace: Path, tools: ToolRegistry, runtime_manager: RuntimeManager):
        self.workspace = workspace
        self.tools = tools
        self.runtime_manager = runtime_manager
        self.executor = ProcedureExecutor(tools, runtime_manager)

    def has_pending_calibration(self, session: Session) -> bool:
        """Return whether the session has an interactive calibration in progress."""
        return self._load_calibration_state(session) is not None

    def looks_like_embodied_request(self, content: str) -> bool:
        """Best-effort embodied intent detector for onboarding interception only."""
        normalized = " ".join(content.strip().lower().split())
        if not normalized:
            return False

        if any(token in normalized for token in self._GENERIC_HINTS):
            return True

        try:
            catalog = build_catalog(self.workspace)
        except Exception:
            catalog = build_catalog()

        for robot in catalog.robots.list():
            if robot.id.lower() in normalized or robot.name.lower() in normalized:
                return True
            if any(primitive.name.lower() in normalized for primitive in robot.primitives):
                return True

        from roboclaw.embodied.execution.integration.adapters.ros2.profiles import DEFAULT_ROS2_PROFILES

        for profile in DEFAULT_ROS2_PROFILES:
            for alias_spec in profile.primitive_aliases:
                if alias_spec.primitive_name.lower() in normalized:
                    return True
                if any(alias.lower() in normalized for alias in alias_spec.aliases):
                    return True
        return False

    async def handle_pending_calibration_message(
        self,
        msg: InboundMessage,
        session: Session,
        on_progress: ProgressCallback | None = None,
    ) -> OutboundMessage:
        """Handle the hard-intercept calibration continuation flow."""
        calibration_state = self._load_calibration_state(session) or {}
        catalog = build_catalog(self.workspace)
        setup, ambiguity, _ = self._resolve_setup(
            session,
            catalog,
            explicit_setup_id=str(calibration_state.get("setup_id") or "").strip() or None,
        )
        if setup is None:
            session.metadata.pop(EMBODIED_CALIBRATION_STATE_KEY, None)
            content = ambiguity or "The pending calibration interaction no longer has a resolvable setup. Reply with `calibrate` again."
            session.add_message("user", msg.content)
            session.add_message("assistant", content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=msg.metadata or {})

        context = self._build_context(session, setup)
        self._bind_active_setup(session, setup.setup_id)
        content = msg.content.strip()
        if not content:
            result = await self.executor.advance_calibration(context, on_progress=on_progress)
            self._sync_calibration_state(session, setup_id=setup.setup_id, runtime_id=context.runtime.id, result=result)
        elif content.lower() == "calibrate":
            phase = self.executor.calibration_phase(context.runtime.id) or str(calibration_state.get("phase") or "")
            calibration_path = context.profile.canonical_calibration_path()
            result = self.executor._so101_calibration_phase_message(
                context,
                phase=phase,
                calibration_path=calibration_path,
            )
        else:
            phase = self.executor.calibration_phase(context.runtime.id) or str(calibration_state.get("phase") or "")
            if phase == "await_mid_pose_ack":
                message = (
                    f"Calibration is pending for setup `{setup.setup_id}`."
                    " Move the arm to a middle pose, then press Enter to start live calibration."
                )
            else:
                message = (
                    f"Calibration is already streaming for setup `{setup.setup_id}`."
                    " Keep moving every joint through its full range of motion, then press Enter to stop and save."
                )
            result = ProcedureExecutionResult(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=message,
                details={"calibration_phase": phase},
            )

        self._sync_runtime_state(session, setup_id=setup.setup_id, runtime=context.runtime)
        session.add_message("user", msg.content)
        session.add_message("assistant", result.message)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=result.message,
            metadata=msg.metadata or {},
        )

    def build_agent_snapshot(
        self,
        session: Session,
        *,
        setup_id: str | None = None,
    ) -> EmbodiedAgentSnapshot:
        """Build the current embodied snapshot for LLM context/tool use."""
        catalog = build_catalog(self.workspace)
        setup, _, candidates = self._resolve_setup(session, catalog, explicit_setup_id=setup_id)
        candidate_summaries = tuple(self._candidate_summary(catalog, item) for item in candidates)
        selected = self._selected_setup_payload(session, catalog, setup)
        calibration_state = self._load_calibration_state(session)
        active_setup_id = self._session_setup_id(session)

        return EmbodiedAgentSnapshot(
            active_setup_id=active_setup_id,
            session_setup_id=active_setup_id,
            selected_setup_id=setup.setup_id if setup is not None else None,
            needs_user_choice=setup is None and len(candidates) > 1,
            candidates=candidate_summaries,
            pending_calibration_phase=str(calibration_state.get("phase") or "").strip() or None if calibration_state else None,
            runtime_status=selected.get("runtime_status"),
            last_error=selected.get("last_error"),
            robot_id=selected.get("robot_id"),
            robot_name=selected.get("robot_name"),
            target_id=selected.get("target_id"),
            transport=selected.get("transport"),
            profile_id=selected.get("profile_id"),
            capability_families=tuple(selected.get("capability_families", ())),
            supported_primitives=tuple(selected.get("supported_primitives", ())),
            primitive_alias_examples=selected.get("primitive_alias_examples", {}),
            calibration_required=selected.get("calibration_required"),
            calibration_present=selected.get("calibration_present"),
        )

    async def execute_action(
        self,
        session: Session,
        *,
        action: str,
        setup_id: str | None = None,
        primitive_name: str | None = None,
        primitive_args: dict[str, Any] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> EmbodiedToolResult:
        """Execute one strong-constrained embodied action for the agent."""
        catalog = build_catalog(self.workspace)
        setup, ambiguity, candidates = self._resolve_setup(session, catalog, explicit_setup_id=setup_id)
        if setup is None:
            if candidates:
                candidate_summaries = [self._candidate_summary(catalog, item) for item in candidates]
                return EmbodiedToolResult(
                    ok=False,
                    action=action,
                    setup_id=None,
                    runtime_status=None,
                    message=ambiguity or "I found multiple embodied setups. Choose a setup id first.",
                    needs_user_choice=True,
                    suggested_next_actions=("ask_user_to_select_setup",),
                    details={"candidates": candidate_summaries},
                )
            return EmbodiedToolResult(
                ok=False,
                action=action,
                setup_id=None,
                runtime_status=None,
                message=(
                    "I could not find a ready embodied setup in this workspace yet. "
                    "Start with onboarding, for example: `I want to connect a real robot`."
                ),
                suggested_next_actions=("start_onboarding",),
                details={},
            )

        context = self._build_context(session, setup)
        self._bind_active_setup(session, setup.setup_id)
        if on_progress:
            await on_progress(f"Embodied action routed to setup `{setup.setup_id}`.")

        if action == "connect":
            result = await self.executor.execute_connect(context, on_progress=on_progress)
        elif action == "calibrate":
            result = await self.executor.execute_calibrate(context, on_progress=on_progress)
            self._sync_calibration_state(session, setup_id=setup.setup_id, runtime_id=context.runtime.id, result=result)
        elif action == "debug":
            result = await self.executor.execute_debug(context)
        elif action == "reset":
            result = await self.executor.execute_reset(context)
        elif action == "run_primitive":
            if not primitive_name or not primitive_name.strip():
                return EmbodiedToolResult(
                    ok=False,
                    action=action,
                    setup_id=setup.setup_id,
                    runtime_status=context.runtime.status.value,
                    message="`primitive_name` is required when action is `run_primitive`.",
                    details={},
                )
            result = await self.executor.execute_move(
                context,
                primitive_name=primitive_name,
                primitive_args=primitive_args,
                on_progress=on_progress,
            )
        else:
            return EmbodiedToolResult(
                ok=False,
                action=action,
                setup_id=setup.setup_id,
                runtime_status=context.runtime.status.value,
                message=f"Unsupported embodied action `{action}`.",
                details={},
            )

        self._sync_runtime_state(session, setup_id=setup.setup_id, runtime=context.runtime)
        return self._tool_result_from_execution(
            session,
            action=action,
            setup=setup,
            context=context,
            result=result,
        )

    def _tool_result_from_execution(
        self,
        session: Session,
        *,
        action: str,
        setup: ResolvedSetup,
        context: ExecutionContext,
        result: ProcedureExecutionResult,
    ) -> EmbodiedToolResult:
        needs_calibration = action != "calibrate" and result.procedure == ProcedureKind.CALIBRATE
        external_intervention_required = self._requires_external_intervention(result.message)
        suggested_next_actions: list[str] = []
        if needs_calibration:
            suggested_next_actions.append("calibrate")
        elif not result.ok and action not in {"debug", "calibrate"}:
            suggested_next_actions.append("debug")
        if action == "calibrate" and not result.ok:
            suggested_next_actions.append("wait_for_user_enter")
        if external_intervention_required:
            suggested_next_actions.append("ask_user_to_check_power_usb")

        if result.procedure == ProcedureKind.CALIBRATE and action == "calibrate":
            self._sync_calibration_state(session, setup_id=setup.setup_id, runtime_id=context.runtime.id, result=result)

        return EmbodiedToolResult(
            ok=result.ok,
            action=action,
            setup_id=setup.setup_id,
            runtime_status=context.runtime.status.value,
            message=result.message,
            needs_calibration=needs_calibration,
            external_intervention_required=external_intervention_required,
            suggested_next_actions=tuple(dict.fromkeys(suggested_next_actions)),
            details=dict(result.details),
        )

    def _selected_setup_payload(
        self,
        session: Session,
        catalog: Any,
        setup: ResolvedSetup | None,
    ) -> dict[str, Any]:
        if setup is None:
            runtime_state = self._load_runtime_state(session)
            return {
                "runtime_status": runtime_state.get("status"),
                "last_error": runtime_state.get("last_error"),
            }

        assembly = catalog.assemblies.get(setup.assembly_id)
        deployment = catalog.deployments.get(setup.deployment_id)
        robot_attachment = assembly.robots[0] if assembly.robots else None
        robot = catalog.robots.get(robot_attachment.robot_id) if robot_attachment is not None else None
        profile = self._resolve_profile(robot.id) if robot is not None else None
        runtime_state = self._runtime_state_for_setup(session, setup)
        calibration_path = profile.canonical_calibration_path() if profile is not None and getattr(profile, "requires_calibration", False) else None

        primitive_alias_examples: dict[str, tuple[str, ...]] = {}
        if profile is not None:
            primitive_alias_examples = {
                item.primitive_name: tuple(item.aliases)
                for item in getattr(profile, "primitive_aliases", ())
            }

        return {
            "runtime_status": runtime_state.get("status"),
            "last_error": runtime_state.get("last_error"),
            "robot_id": getattr(robot, "id", None),
            "robot_name": getattr(robot, "name", None),
            "target_id": deployment.target_id,
            "transport": str(deployment.connection.get("transport") or "").strip() or None,
            "profile_id": getattr(profile, "id", None),
            "capability_families": tuple(item.value for item in getattr(robot, "capability_families", ())),
            "supported_primitives": tuple(primitive.name for primitive in getattr(robot, "primitives", ())),
            "primitive_alias_examples": primitive_alias_examples,
            "calibration_required": bool(getattr(profile, "requires_calibration", False)) if profile is not None else None,
            "calibration_present": calibration_path.exists() if calibration_path is not None else None,
        }

    def _build_context(self, session: Session, setup: ResolvedSetup) -> ExecutionContext:
        catalog = build_catalog(self.workspace)
        assembly = catalog.assemblies.get(setup.assembly_id)
        deployment = catalog.deployments.get(setup.deployment_id)
        adapter_binding = catalog.adapters.get(setup.adapter_id)
        target = assembly.execution_target(deployment.target_id)
        robot_attachment = assembly.robots[0]
        robot = catalog.robots.get(robot_attachment.robot_id)
        profile = self._resolve_profile(robot.id)
        runtime_id = f"{session.key}:{setup.setup_id}"
        runtime = self.executor.runtime_for(
            runtime_id=runtime_id,
            setup_id=setup.setup_id,
            assembly_id=setup.assembly_id,
            deployment_id=setup.deployment_id,
            target_id=deployment.target_id,
            adapter_id=setup.adapter_id,
        )
        return ExecutionContext(
            setup_id=setup.setup_id,
            assembly=assembly,
            deployment=deployment,
            target=target,
            robot=robot,
            adapter_binding=adapter_binding,
            profile=profile,
            runtime=runtime,
        )

    def _resolve_setup(
        self,
        session: Session,
        catalog: Any,
        *,
        explicit_setup_id: str | None = None,
    ) -> tuple[ResolvedSetup | None, str | None, list[ResolvedSetup]]:
        candidates = self._workspace_candidates(catalog)
        if explicit_setup_id:
            selected = next((item for item in candidates if item.setup_id == explicit_setup_id), None)
            if selected is None:
                return None, f"I could not find embodied setup `{explicit_setup_id}` in this workspace.", candidates
            return selected, None, candidates

        session_setup_id = self._session_setup_id(session)
        if session_setup_id:
            selected = next((item for item in candidates if item.setup_id == session_setup_id), None)
            if selected is not None:
                return selected, None, candidates

        if not candidates:
            return None, None, candidates
        if len(candidates) > 1:
            ids = ", ".join(item.setup_id for item in candidates)
            return None, f"I found multiple embodied setups in this workspace: {ids}. Tell me which setup id to use.", candidates
        return candidates[0], None, candidates

    def _workspace_candidates(self, catalog: Any) -> list[ResolvedSetup]:
        candidates: list[ResolvedSetup] = []
        for assembly in catalog.assemblies.list():
            deployments = catalog.deployments.for_assembly(assembly.id)
            adapters = catalog.adapters.for_assembly(assembly.id)
            if len(deployments) == 1 and len(adapters) == 1:
                deployment = deployments[0]
                adapter = adapters[0]
                candidates.append(
                    ResolvedSetup(
                        setup_id=assembly.id,
                        assembly_id=assembly.id,
                        deployment_id=deployment.id,
                        adapter_id=adapter.id,
                        source="workspace",
                    )
                )
        return candidates

    def _candidate_summary(self, catalog: Any, setup: ResolvedSetup) -> dict[str, Any]:
        assembly = catalog.assemblies.get(setup.assembly_id)
        deployment = catalog.deployments.get(setup.deployment_id)
        robot_attachment = assembly.robots[0] if assembly.robots else None
        robot = catalog.robots.get(robot_attachment.robot_id) if robot_attachment is not None else None
        return {
            "setup_id": setup.setup_id,
            "assembly_id": setup.assembly_id,
            "deployment_id": setup.deployment_id,
            "adapter_id": setup.adapter_id,
            "robot_id": getattr(robot, "id", None),
            "robot_name": getattr(robot, "name", None),
            "target_id": deployment.target_id,
            "source": setup.source,
        }

    def _session_setup_id(self, session: Session) -> str | None:
        active_setup_id = str(session.metadata.get(EMBODIED_ACTIVE_SETUP_KEY) or "").strip()
        if active_setup_id:
            return active_setup_id
        state = self._load_onboarding_state(session)
        if state is not None and state.is_ready:
            return state.setup_id
        runtime = self._load_runtime_state(session)
        setup_id = str(runtime.get("setup_id") or "").strip()
        return setup_id or None

    @staticmethod
    def _bind_active_setup(session: Session, setup_id: str) -> None:
        session.metadata[EMBODIED_ACTIVE_SETUP_KEY] = setup_id

    def _runtime_state_for_setup(self, session: Session, setup: ResolvedSetup) -> dict[str, Any]:
        runtime_id = f"{session.key}:{setup.setup_id}"
        try:
            runtime = self.runtime_manager.get(runtime_id)
        except KeyError:
            runtime = None

        if runtime is not None:
            return {
                "runtime_id": runtime.id,
                "status": runtime.status.value,
                "last_error": runtime.last_error,
            }

        stored = self._load_runtime_state(session)
        if str(stored.get("setup_id") or "").strip() == setup.setup_id:
            return stored
        return {}

    @staticmethod
    def _load_runtime_state(session: Session) -> dict[str, Any]:
        raw = session.metadata.get(EMBODIED_RUNTIME_STATE_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _sync_runtime_state(session: Session, *, setup_id: str, runtime: Any) -> None:
        session.metadata[EMBODIED_ACTIVE_SETUP_KEY] = setup_id
        session.metadata[EMBODIED_RUNTIME_STATE_KEY] = {
            "runtime_id": runtime.id,
            "setup_id": setup_id,
            "status": runtime.status.value,
            "last_error": runtime.last_error,
        }

    def _load_onboarding_state(self, session: Session) -> SetupOnboardingState | None:
        raw = session.metadata.get(SETUP_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            return SetupOnboardingState.from_dict(raw)
        except Exception:
            return None

    @staticmethod
    def _load_calibration_state(session: Session) -> dict[str, Any] | None:
        raw = session.metadata.get(EMBODIED_CALIBRATION_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        return dict(raw)

    @staticmethod
    def _sync_calibration_state(
        session: Session,
        *,
        setup_id: str,
        runtime_id: str,
        result: ProcedureExecutionResult,
    ) -> None:
        phase = str(result.details.get("calibration_phase") or "").strip()
        if phase in {"await_mid_pose_ack", "streaming"}:
            session.metadata[EMBODIED_CALIBRATION_STATE_KEY] = {
                "setup_id": setup_id,
                "runtime_id": runtime_id,
                "phase": phase,
            }
            return
        session.metadata.pop(EMBODIED_CALIBRATION_STATE_KEY, None)

    @staticmethod
    def _requires_external_intervention(message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in (
                "power is on",
                "usb/serial cable",
                "usb device",
                "plug",
                "replug",
                "powered",
                "power-cycle",
                "manual",
                "check_power_usb",
            )
        )

    @staticmethod
    def _resolve_profile(robot_id: str) -> Any:
        from roboclaw.embodied.execution.integration.adapters.ros2.profiles import get_ros2_profile

        return get_ros2_profile(robot_id)
