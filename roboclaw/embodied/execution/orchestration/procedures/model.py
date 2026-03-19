"""Cross-robot procedures for embodied control interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily


class ProcedureKind(StrEnum):
    """Procedure category."""

    CONNECT = "connect"
    CALIBRATE = "calibrate"
    MOVE = "move"
    DEBUG = "debug"
    RESET = "reset"


class PreconditionSource(StrEnum):
    """Where a procedure precondition is evaluated from."""

    RUNTIME = "runtime"
    ADAPTER = "adapter"
    DEPLOYMENT = "deployment"
    INPUT = "input"
    STEP_RESULT = "step_result"


class PreconditionOperator(StrEnum):
    """Operator used to evaluate one precondition."""

    EXISTS = "exists"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN_SET = "in_set"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


class InterventionTiming(StrEnum):
    """When a human operator intervention should happen."""

    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    ON_FAILURE = "on_failure"


class CancellationMode(StrEnum):
    """How a procedure/step can be cancelled."""

    NON_CANCELLABLE = "non_cancellable"
    SAFE_POINT = "safe_point"
    IMMEDIATE = "immediate"


class CompensationTrigger(StrEnum):
    """When one compensation action should be applied."""

    ON_FAILURE = "on_failure"
    ON_CANCEL = "on_cancel"
    ON_TIMEOUT = "on_timeout"


class RollbackStrategy(StrEnum):
    """How rollback should be applied for a procedure."""

    NONE = "none"
    REVERSE_COMPENSATION = "reverse_compensation"
    DECLARED_ORDER = "declared_order"


class IdempotencyMode(StrEnum):
    """Idempotency strictness for repeated procedure requests."""

    NONE = "none"
    BEST_EFFORT = "best_effort"
    STRICT = "strict"


class IdempotencyConflictPolicy(StrEnum):
    """How idempotency conflicts are resolved."""

    REUSE_RESULT = "reuse_result"
    REJECT_DUPLICATE = "reject_duplicate"
    RESTART_EXECUTION = "restart_execution"


class ProcedureActionTarget(StrEnum):
    """Execution target family for one procedure action."""

    ADAPTER = "adapter"
    ORCHESTRATOR = "orchestrator"


class AdapterProcedureAction(StrEnum):
    """Adapter-facing actions used by embodied control procedures."""

    PROBE_ENV = "probe_env"
    CHECK_DEPENDENCIES = "check_dependencies"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    READY = "ready"
    STOP = "stop"
    RESET = "reset"
    RECOVER = "recover"
    GET_STATE = "get_state"
    EXECUTE_PRIMITIVE = "execute_primitive"
    CAPTURE_SENSOR = "capture_sensor"
    DEBUG_SNAPSHOT = "debug_snapshot"


class OrchestratorProcedureAction(StrEnum):
    """Orchestrator-level actions resolved by runtime services."""

    RESOLVE_TARGET = "resolve_target"
    RESOLVE_PRIMITIVE = "resolve_primitive"
    LIST_CALIBRATION_TARGETS = "list_calibration_targets"
    START_CALIBRATION = "start_calibration"
    TRACK_CALIBRATION = "track_calibration"
    CANCEL_CALIBRATION = "cancel_calibration"


@dataclass(frozen=True)
class ProcedureActionRef:
    """Typed action reference used by procedure steps and policies."""

    target: ProcedureActionTarget
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Procedure action name cannot be empty.")


def adapter_action(action: AdapterProcedureAction) -> ProcedureActionRef:
    """Create an adapter action reference."""

    return ProcedureActionRef(target=ProcedureActionTarget.ADAPTER, name=action.value)


def orchestrator_action(action: OrchestratorProcedureAction) -> ProcedureActionRef:
    """Create an orchestrator action reference."""

    return ProcedureActionRef(target=ProcedureActionTarget.ORCHESTRATOR, name=action.value)


@dataclass(frozen=True)
class ProcedureRetryPolicy:
    """Retry policy for a step or a whole procedure."""

    max_retries: int = 0
    backoff_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("Procedure retry max_retries cannot be negative.")
        if self.backoff_s < 0:
            raise ValueError("Procedure retry backoff_s cannot be negative.")


@dataclass(frozen=True)
class ProcedureCancellationPolicy:
    """Cancellation behavior for one step or whole procedure."""

    mode: CancellationMode = CancellationMode.SAFE_POINT
    cancel_action: ProcedureActionRef | None = None
    timeout_s: float = 10.0
    requires_operator_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.mode == CancellationMode.NON_CANCELLABLE and self.cancel_action is not None:
            raise ValueError("Non-cancellable policy cannot define cancel_action.")
        if self.timeout_s <= 0:
            raise ValueError("Cancellation policy timeout_s must be > 0.")


@dataclass(frozen=True)
class ProcedureCompensationSpec:
    """Compensation action executed during rollback."""

    action: ProcedureActionRef
    description: str
    triggers: tuple[CompensationTrigger, ...] = (
        CompensationTrigger.ON_FAILURE,
        CompensationTrigger.ON_CANCEL,
    )
    timeout_s: float | None = None
    best_effort: bool = True

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Compensation description cannot be empty.")
        if not self.triggers:
            raise ValueError("Compensation must declare at least one trigger.")
        if len(set(self.triggers)) != len(self.triggers):
            raise ValueError("Compensation triggers cannot contain duplicates.")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("Compensation timeout_s must be > 0 when specified.")


@dataclass(frozen=True)
class ProcedureIdempotencyPolicy:
    """Idempotency behavior for repeated calls."""

    mode: IdempotencyMode = IdempotencyMode.NONE
    key_fields: tuple[str, ...] = field(default_factory=tuple)
    conflict_policy: IdempotencyConflictPolicy = IdempotencyConflictPolicy.REUSE_RESULT
    cache_window_s: float | None = None
    persist_result: bool = True

    def __post_init__(self) -> None:
        if self.mode == IdempotencyMode.NONE and self.key_fields:
            raise ValueError("Idempotency key_fields must be empty when mode is NONE.")
        if self.mode != IdempotencyMode.NONE and not self.key_fields:
            raise ValueError("Idempotency key_fields are required when mode is enabled.")
        if any(not field_name.strip() for field_name in self.key_fields):
            raise ValueError("Idempotency key_fields cannot contain empty values.")
        if self.cache_window_s is not None and self.cache_window_s <= 0:
            raise ValueError("Idempotency cache_window_s must be > 0 when specified.")


@dataclass(frozen=True)
class ProcedurePrecondition:
    """Machine-checkable precondition for one step."""

    id: str
    source: PreconditionSource
    key: str
    operator: PreconditionOperator
    expected: Any | None = None
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ProcedureStepEdge:
    """Directed edge in a procedure step graph."""

    from_step_id: str
    to_step_id: str
    condition: str | None = None


@dataclass(frozen=True)
class OperatorInterventionPoint:
    """Human-in-the-loop intervention point in one procedure."""

    id: str
    step_id: str
    timing: InterventionTiming
    instruction: str
    blocking: bool = True


@dataclass(frozen=True)
class ProcedureStep:
    """One executable step in a procedure."""

    id: str
    action: ProcedureActionRef
    description: str
    preconditions: tuple[ProcedurePrecondition, ...] = field(default_factory=tuple)
    timeout_s: float | None = None
    retry_policy: ProcedureRetryPolicy = field(default_factory=ProcedureRetryPolicy)
    cancellation: ProcedureCancellationPolicy | None = None
    compensation: ProcedureCompensationSpec | None = None
    idempotency: ProcedureIdempotencyPolicy | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Procedure step id cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Step '{self.id}' description cannot be empty.")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"Step '{self.id}' timeout_s must be > 0 when specified.")


@dataclass(frozen=True)
class ProcedureDefinition:
    """Named procedure composed from stable steps."""

    id: str
    kind: ProcedureKind
    description: str
    required_capabilities: tuple[CapabilityFamily, ...] = field(default_factory=tuple)
    steps: tuple[ProcedureStep, ...] = field(default_factory=tuple)
    step_edges: tuple[ProcedureStepEdge, ...] = field(default_factory=tuple)
    entry_step_ids: tuple[str, ...] = field(default_factory=tuple)
    terminal_step_ids: tuple[str, ...] = field(default_factory=tuple)
    default_timeout_s: float | None = None
    default_retry_policy: ProcedureRetryPolicy = field(default_factory=ProcedureRetryPolicy)
    operator_interventions: tuple[OperatorInterventionPoint, ...] = field(default_factory=tuple)
    cancellation_policy: ProcedureCancellationPolicy = field(default_factory=ProcedureCancellationPolicy)
    rollback_strategy: RollbackStrategy = RollbackStrategy.REVERSE_COMPENSATION
    idempotency_policy: ProcedureIdempotencyPolicy = field(default_factory=ProcedureIdempotencyPolicy)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"Procedure '{self.id}' must contain at least one step.")
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError(f"Procedure '{self.id}' default_timeout_s must be > 0 when specified.")

        step_ids = tuple(step.id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError(f"Procedure '{self.id}' contains duplicate step ids.")
        step_id_set = set(step_ids)

        if self.step_edges:
            for edge in self.step_edges:
                if edge.from_step_id not in step_id_set:
                    raise ValueError(
                        f"Procedure '{self.id}' has edge from unknown step '{edge.from_step_id}'."
                    )
                if edge.to_step_id not in step_id_set:
                    raise ValueError(
                        f"Procedure '{self.id}' has edge to unknown step '{edge.to_step_id}'."
                    )
            edges = self.step_edges
        else:
            # Default linear graph for simple procedures.
            edges = tuple(
                ProcedureStepEdge(from_step_id=step_ids[i], to_step_id=step_ids[i + 1])
                for i in range(len(step_ids) - 1)
            )
            object.__setattr__(self, "step_edges", edges)

        incoming = {step_id: 0 for step_id in step_ids}
        outgoing = {step_id: 0 for step_id in step_ids}
        for edge in edges:
            outgoing[edge.from_step_id] += 1
            incoming[edge.to_step_id] += 1

        if self.entry_step_ids:
            entries = self.entry_step_ids
            unknown_entries = set(entries) - step_id_set
            if unknown_entries:
                names = ", ".join(sorted(unknown_entries))
                raise ValueError(f"Procedure '{self.id}' has unknown entry steps: {names}.")
        else:
            entries = tuple(step_id for step_id in step_ids if incoming[step_id] == 0)
            object.__setattr__(self, "entry_step_ids", entries)

        if self.terminal_step_ids:
            terminals = self.terminal_step_ids
            unknown_terminals = set(terminals) - step_id_set
            if unknown_terminals:
                names = ", ".join(sorted(unknown_terminals))
                raise ValueError(f"Procedure '{self.id}' has unknown terminal steps: {names}.")
        else:
            terminals = tuple(step_id for step_id in step_ids if outgoing[step_id] == 0)
            object.__setattr__(self, "terminal_step_ids", terminals)

        intervention_ids = [point.id for point in self.operator_interventions]
        if len(set(intervention_ids)) != len(intervention_ids):
            raise ValueError(f"Procedure '{self.id}' contains duplicate intervention ids.")
        for point in self.operator_interventions:
            if point.step_id not in step_id_set:
                raise ValueError(
                    f"Procedure '{self.id}' intervention '{point.id}' references unknown step '{point.step_id}'."
                )

        has_compensation = any(step.compensation is not None for step in self.steps)
        if self.rollback_strategy != RollbackStrategy.NONE and not has_compensation:
            raise ValueError(
                f"Procedure '{self.id}' requires compensation steps when rollback is enabled."
            )
