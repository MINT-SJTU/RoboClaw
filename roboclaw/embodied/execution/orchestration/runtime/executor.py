"""Runtime execution helpers for embodied procedures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.embodied.definition.components.robots.model import RobotManifest
from roboclaw.embodied.definition.systems.assemblies.model import AssemblyManifest
from roboclaw.embodied.definition.systems.deployments.model import DeploymentProfile
from roboclaw.embodied.execution.integration.adapters.loader import AdapterLoader
from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind
from roboclaw.embodied.execution.orchestration.runtime.manager import RuntimeManager
from roboclaw.embodied.execution.orchestration.runtime.model import RuntimeSession, RuntimeStatus


@dataclass(frozen=True)
class ExecutionContext:
    """Resolved embodied execution state for one setup and runtime session."""

    setup_id: str
    assembly: AssemblyManifest
    deployment: DeploymentProfile
    target: ExecutionTarget
    robot: RobotManifest
    adapter_binding: Any
    profile: Any
    runtime: RuntimeSession


@dataclass(frozen=True)
class ProcedureExecutionResult:
    """Normalized result returned to the controller."""

    procedure: ProcedureKind
    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class ProcedureExecutor:
    """Execute a small stage-1 subset of embodied procedures."""

    def __init__(self, tools: ToolRegistry, runtime_manager: RuntimeManager):
        self._loader = AdapterLoader(tools)
        self._runtime_manager = runtime_manager
        self._adapters: dict[str, Any] = {}

    def runtime_for(
        self,
        *,
        runtime_id: str,
        setup_id: str,
        assembly_id: str,
        deployment_id: str,
        target_id: str,
        adapter_id: str,
    ) -> RuntimeSession:
        try:
            return self._runtime_manager.get(runtime_id)
        except KeyError:
            return self._runtime_manager.create(
                session_id=runtime_id,
                assembly_id=assembly_id,
                deployment_id=deployment_id,
                target_id=target_id,
                adapter_id=adapter_id,
            )

    def _adapter(self, context: ExecutionContext) -> Any:
        adapter = self._adapters.get(context.runtime.id)
        if adapter is None:
            adapter = self._loader.load(
                binding=context.adapter_binding,
                assembly=context.assembly,
                deployment=context.deployment,
                target=context.target,
                robot=context.robot,
                profile=context.profile,
            )
            self._adapters[context.runtime.id] = adapter
        return adapter

    async def execute_connect(self, context: ExecutionContext) -> ProcedureExecutionResult:
        adapter = self._adapter(context)
        context.runtime.status = RuntimeStatus.CONNECTING

        probe = adapter.probe_env()
        deps = adapter.check_dependencies()
        if not deps.ok:
            missing = ", ".join(deps.missing_required) or "required ROS2 interfaces"
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = missing
            return ProcedureExecutionResult(
                procedure=ProcedureKind.CONNECT,
                ok=False,
                message=f"Setup `{context.setup_id}` is not ready yet: missing {missing}.",
                details={"probe": probe.details, "missing_required": deps.missing_required},
            )

        connected = await adapter.connect(
            target_id=context.target.id,
            config=dict(context.deployment.connection),
        )
        if not connected.ok:
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = connected.message or connected.error_code
            return ProcedureExecutionResult(
                procedure=ProcedureKind.CONNECT,
                ok=False,
                message=connected.message or f"Failed to connect setup `{context.setup_id}`.",
                details={"error_code": connected.error_code, **connected.details},
            )

        ready = await adapter.ready()
        if not ready.ready:
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = ready.message
            return ProcedureExecutionResult(
                procedure=ProcedureKind.CONNECT,
                ok=False,
                message=ready.message or f"Setup `{context.setup_id}` is not ready for commands yet.",
                details={"blocked_operations": [item.value for item in ready.blocked_operations], **ready.details},
            )

        context.runtime.status = RuntimeStatus.READY
        context.runtime.last_error = None
        return ProcedureExecutionResult(
            procedure=ProcedureKind.CONNECT,
            ok=True,
            message=f"Connected setup `{context.setup_id}` on target `{context.target.id}`.",
            details={"probe": probe.details, "target_id": context.target.id},
        )

    async def execute_move(
        self,
        context: ExecutionContext,
        *,
        primitive_name: str,
        primitive_args: dict[str, Any] | None = None,
    ) -> ProcedureExecutionResult:
        adapter = self._adapter(context)
        if context.runtime.status == RuntimeStatus.DISCONNECTED:
            connected = await self.execute_connect(context)
            if not connected.ok:
                return connected

        context.runtime.status = RuntimeStatus.BUSY
        pre_state = await adapter.get_state()
        primitive = await adapter.execute_primitive(primitive_name, primitive_args or {})
        if not primitive.accepted:
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = primitive.message or primitive.error_code
            return ProcedureExecutionResult(
                procedure=ProcedureKind.MOVE,
                ok=False,
                message=primitive.message or f"Primitive `{primitive_name}` was rejected.",
                details={"error_code": primitive.error_code, **primitive.output},
            )

        post_state = await adapter.get_state()
        context.runtime.status = RuntimeStatus.READY
        context.runtime.last_error = None
        completion = "completed" if primitive.completed is not False else "accepted"
        state_message = self._state_confirmation(primitive_name, pre_state.values, post_state.values)
        message = f"Primitive `{primitive_name}` {completion} on setup `{context.setup_id}`."
        if state_message:
            message = f"{message} {state_message}"
        return ProcedureExecutionResult(
            procedure=ProcedureKind.MOVE,
            ok=True,
            message=message,
            details={
                "state_before": pre_state.values,
                "state_after": post_state.values,
                **primitive.output,
            },
        )

    async def execute_debug(self, context: ExecutionContext) -> ProcedureExecutionResult:
        adapter = self._adapter(context)
        probe = adapter.probe_env()
        state = await adapter.get_state()
        snapshot = await adapter.debug_snapshot()
        context.runtime.status = RuntimeStatus.READY if snapshot.captured else RuntimeStatus.ERROR
        if not snapshot.captured:
            context.runtime.last_error = snapshot.message or snapshot.summary
        return ProcedureExecutionResult(
            procedure=ProcedureKind.DEBUG,
            ok=snapshot.captured,
            message=snapshot.summary,
            details={"probe": probe.details, "state": state.values, "artifacts": snapshot.artifacts},
        )

    async def execute_reset(self, context: ExecutionContext) -> ProcedureExecutionResult:
        adapter = self._adapter(context)
        stop_result = await adapter.stop(scope="all")
        reset_result = await adapter.reset(mode="home")
        if not reset_result.ok:
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = reset_result.message or reset_result.error_code
            return ProcedureExecutionResult(
                procedure=ProcedureKind.RESET,
                ok=False,
                message=reset_result.message or f"Failed to reset setup `{context.setup_id}`.",
                details={
                    "stop_message": stop_result.message,
                    "error_code": reset_result.error_code,
                    **reset_result.details,
                },
            )

        context.runtime.status = RuntimeStatus.READY
        context.runtime.last_error = None
        return ProcedureExecutionResult(
            procedure=ProcedureKind.RESET,
            ok=True,
            message=reset_result.message or f"Reset setup `{context.setup_id}` to home.",
            details={"stop_message": stop_result.message, **reset_result.details},
        )

    async def execute_calibrate(self, context: ExecutionContext) -> ProcedureExecutionResult:
        return ProcedureExecutionResult(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message=(
                f"Setup `{context.setup_id}` does not expose a stage-1 calibration bridge yet. "
                "Try `connect`, `open gripper`, `debug`, or `reset`."
            ),
        )

    @staticmethod
    def _state_confirmation(
        primitive_name: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
    ) -> str:
        gripper_before = ProcedureExecutor._gripper_percent(pre_state)
        gripper_after = ProcedureExecutor._gripper_percent(post_state)
        if primitive_name in {"gripper_open", "gripper_close"} and gripper_after is not None:
            if primitive_name == "gripper_open":
                label = "open" if gripper_after >= 60 else "partially open"
            elif primitive_name == "gripper_close":
                label = "closed" if gripper_after <= 40 else "partially open"
            elif gripper_after >= 80:
                label = "open"
            elif gripper_after <= 20:
                label = "closed"
            else:
                label = "partially open"
            if gripper_before is not None:
                return (
                    f"Current gripper state: {label} "
                    f"({gripper_after:.1f}% open, was {gripper_before:.1f}%)."
                )
            return f"Current gripper state: {label} ({gripper_after:.1f}% open)."
        return ""

    @staticmethod
    def _gripper_percent(values: dict[str, Any]) -> float | None:
        raw = values.get("gripper_percent")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
