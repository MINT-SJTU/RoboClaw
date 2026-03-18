"""Embodied execution controller for ready setups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.catalog import build_catalog
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind
from roboclaw.embodied.execution.orchestration.runtime.executor import (
    ExecutionContext,
    ProcedureExecutor,
)
from roboclaw.embodied.execution.orchestration.runtime.manager import RuntimeManager
from roboclaw.embodied.onboarding import SETUP_STATE_KEY, SetupOnboardingState
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]

EMBODIED_RUNTIME_STATE_KEY = "embodied_runtime"


@dataclass(frozen=True)
class EmbodiedIntent:
    """One normalized stage-1 execution intent."""

    kind: ProcedureKind
    primitive_name: str | None = None
    primitive_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedSetup:
    """Runtime-ready setup selection."""

    setup_id: str
    assembly_id: str
    deployment_id: str
    adapter_id: str
    source: str


class EmbodiedExecutionController:
    """Handle embodied commands after onboarding hands off a ready setup."""

    _CONNECT_TOKENS = ("connect", "连接", "接入", "开始连接")
    _CALIBRATE_TOKENS = ("calibrate", "校准")
    _DEBUG_TOKENS = ("debug", "诊断", "检查机器人", "检查状态")
    _RESET_TOKENS = ("reset", "复位", "回零", "恢复默认", "回到初始")

    def __init__(self, workspace: Path, tools: ToolRegistry, runtime_manager: RuntimeManager):
        self.workspace = workspace
        self.tools = tools
        self.runtime_manager = runtime_manager
        self.executor = ProcedureExecutor(tools, runtime_manager)

    def should_handle(self, session: Session, content: str) -> bool:
        if self._parse_intent(content) is None:
            return False
        state = self._load_onboarding_state(session)
        if state is not None and state.is_ready:
            return True
        try:
            catalog = build_catalog(self.workspace)
        except Exception:
            return False
        return bool(catalog.assemblies.list())

    async def handle_message(
        self,
        msg: InboundMessage,
        session: Session,
        on_progress: ProgressCallback | None = None,
    ) -> OutboundMessage:
        intent = self._parse_intent(msg.content)
        if intent is None:
            content = (
                "I can currently help with embodied `connect`, `calibrate`, `move`, `debug`, and `reset` commands."
            )
            session.add_message("user", msg.content)
            session.add_message("assistant", content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=msg.metadata or {})

        catalog = build_catalog(self.workspace)
        setup, ambiguity = self._resolve_setup(session, catalog)
        if setup is None:
            content = ambiguity or (
                "I could not find a ready embodied setup in this workspace yet. "
                "Start with onboarding, for example: `I want to use a real robot`."
            )
            session.add_message("user", msg.content)
            session.add_message("assistant", content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=msg.metadata or {})

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
        context = ExecutionContext(
            setup_id=setup.setup_id,
            assembly=assembly,
            deployment=deployment,
            target=target,
            robot=robot,
            adapter_binding=adapter_binding,
            profile=profile,
            runtime=runtime,
        )

        if on_progress:
            await on_progress(f"Embodied command routed to setup `{setup.setup_id}`.")

        if intent.kind == ProcedureKind.CONNECT:
            result = await self.executor.execute_connect(context)
        elif intent.kind == ProcedureKind.MOVE and intent.primitive_name is not None:
            result = await self.executor.execute_move(
                context,
                primitive_name=intent.primitive_name,
                primitive_args=intent.primitive_args,
            )
        elif intent.kind == ProcedureKind.DEBUG:
            result = await self.executor.execute_debug(context)
        elif intent.kind == ProcedureKind.RESET:
            result = await self.executor.execute_reset(context)
        elif intent.kind == ProcedureKind.CALIBRATE:
            result = await self.executor.execute_calibrate(context)
        else:
            result = await self.executor.execute_move(
                context,
                primitive_name=intent.primitive_name or "unknown",
                primitive_args=intent.primitive_args,
            )

        session.metadata[EMBODIED_RUNTIME_STATE_KEY] = {
            "runtime_id": runtime.id,
            "setup_id": setup.setup_id,
            "status": runtime.status.value,
            "last_error": runtime.last_error,
        }
        session.add_message("user", msg.content)
        session.add_message("assistant", result.message)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=result.message,
            metadata=msg.metadata or {},
        )

    def _load_onboarding_state(self, session: Session) -> SetupOnboardingState | None:
        raw = session.metadata.get(SETUP_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            return SetupOnboardingState.from_dict(raw)
        except Exception:
            return None

    def _resolve_setup(self, session: Session, catalog: Any) -> tuple[ResolvedSetup | None, str | None]:
        state = self._load_onboarding_state(session)
        if state is not None and state.is_ready:
            return ResolvedSetup(
                setup_id=state.setup_id,
                assembly_id=state.assembly_id,
                deployment_id=state.deployment_id,
                adapter_id=state.adapter_id,
                source="session",
            ), None

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

        if not candidates:
            return None, None
        if len(candidates) > 1:
            ids = ", ".join(item.setup_id for item in candidates)
            return None, f"I found multiple embodied setups in this workspace: {ids}. Tell me which setup id to use."
        return candidates[0], None

    def _parse_intent(self, content: str) -> EmbodiedIntent | None:
        normalized = " ".join(content.strip().lower().split())
        if not normalized:
            return None

        if any(token in normalized for token in self._CONNECT_TOKENS):
            return EmbodiedIntent(kind=ProcedureKind.CONNECT)
        if any(token in normalized for token in self._CALIBRATE_TOKENS):
            return EmbodiedIntent(kind=ProcedureKind.CALIBRATE)
        if any(token in normalized for token in self._DEBUG_TOKENS):
            return EmbodiedIntent(kind=ProcedureKind.DEBUG)
        if any(token in normalized for token in self._RESET_TOKENS):
            return EmbodiedIntent(kind=ProcedureKind.RESET)

        from roboclaw.embodied.execution.integration.adapters.ros2.profiles import DEFAULT_ROS2_PROFILES

        for profile in DEFAULT_ROS2_PROFILES:
            primitive = profile.resolve_primitive_alias(normalized)
            if primitive is not None:
                return EmbodiedIntent(
                    kind=ProcedureKind.MOVE,
                    primitive_name=primitive.primitive_name,
                    primitive_args=primitive.args,
                )
        return None

    @staticmethod
    def _resolve_profile(robot_id: str) -> Any:
        from roboclaw.embodied.execution.integration.adapters.ros2.profiles import get_ros2_profile

        return get_ros2_profile(robot_id)
