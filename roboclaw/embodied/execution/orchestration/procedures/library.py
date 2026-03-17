"""Built-in first-landscape procedures."""

from __future__ import annotations

from roboclaw.embodied.execution.orchestration.procedures.model import (
    AdapterProcedureAction,
    CancellationMode,
    CompensationTrigger,
    IdempotencyConflictPolicy,
    IdempotencyMode,
    InterventionTiming,
    OperatorInterventionPoint,
    OrchestratorProcedureAction,
    PreconditionOperator,
    PreconditionSource,
    ProcedureCancellationPolicy,
    ProcedureCompensationSpec,
    ProcedureDefinition,
    ProcedureIdempotencyPolicy,
    ProcedureKind,
    ProcedurePrecondition,
    ProcedureRetryPolicy,
    ProcedureStep,
    ProcedureStepEdge,
    RollbackStrategy,
    adapter_action,
    orchestrator_action,
)
from roboclaw.embodied.definition.foundation.schema import CapabilityFamily

CONNECT_PROCEDURE = ProcedureDefinition(
    id="connect_default",
    kind=ProcedureKind.CONNECT,
    description="Probe environment, select target, connect adapter, and verify ready state.",
    required_capabilities=(CapabilityFamily.LIFECYCLE,),
    steps=(
        ProcedureStep(
            "probe_env",
            adapter_action(AdapterProcedureAction.PROBE_ENV),
            "Probe the environment and available transport.",
            timeout_s=10.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=1.0),
        ),
        ProcedureStep(
            "select_target",
            orchestrator_action(OrchestratorProcedureAction.RESOLVE_TARGET),
            "Resolve the desired execution target.",
            preconditions=(
                ProcedurePrecondition(
                    id="target_hint_present",
                    source=PreconditionSource.INPUT,
                    key="target_id",
                    operator=PreconditionOperator.EXISTS,
                    required=False,
                    description="Target hint may be provided by the user or deployment defaults.",
                ),
            ),
            timeout_s=5.0,
        ),
        ProcedureStep(
            "connect",
            adapter_action(AdapterProcedureAction.CONNECT),
            "Connect the adapter to the target.",
            timeout_s=30.0,
            retry_policy=ProcedureRetryPolicy(max_retries=2, backoff_s=1.0),
            cancellation=ProcedureCancellationPolicy(
                mode=CancellationMode.IMMEDIATE,
                cancel_action=adapter_action(AdapterProcedureAction.DISCONNECT),
                timeout_s=5.0,
            ),
            compensation=ProcedureCompensationSpec(
                action=adapter_action(AdapterProcedureAction.DISCONNECT),
                description="Disconnect transport if connection partially succeeded.",
                triggers=(
                    CompensationTrigger.ON_FAILURE,
                    CompensationTrigger.ON_CANCEL,
                    CompensationTrigger.ON_TIMEOUT,
                ),
                timeout_s=10.0,
            ),
            idempotency=ProcedureIdempotencyPolicy(
                mode=IdempotencyMode.BEST_EFFORT,
                key_fields=("deployment_id", "target_id"),
                conflict_policy=IdempotencyConflictPolicy.REUSE_RESULT,
                cache_window_s=30.0,
            ),
        ),
        ProcedureStep(
            "verify_state",
            adapter_action(AdapterProcedureAction.READY),
            "Verify the runtime is ready.",
            timeout_s=10.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=0.5),
        ),
    ),
    step_edges=(
        ProcedureStepEdge("probe_env", "select_target"),
        ProcedureStepEdge("select_target", "connect"),
        ProcedureStepEdge("connect", "verify_state"),
    ),
    default_timeout_s=20.0,
    default_retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=1.0),
    cancellation_policy=ProcedureCancellationPolicy(
        mode=CancellationMode.SAFE_POINT,
        cancel_action=adapter_action(AdapterProcedureAction.DISCONNECT),
        timeout_s=10.0,
    ),
    rollback_strategy=RollbackStrategy.REVERSE_COMPENSATION,
    idempotency_policy=ProcedureIdempotencyPolicy(
        mode=IdempotencyMode.BEST_EFFORT,
        key_fields=("deployment_id", "target_id"),
        conflict_policy=IdempotencyConflictPolicy.REUSE_RESULT,
        cache_window_s=60.0,
    ),
    operator_interventions=(
        OperatorInterventionPoint(
            id="manual_network_check",
            step_id="connect",
            timing=InterventionTiming.ON_FAILURE,
            instruction="Check cable/network power and retry connection.",
        ),
    ),
)

CALIBRATE_PROCEDURE = ProcedureDefinition(
    id="calibrate_default",
    kind=ProcedureKind.CALIBRATE,
    description="List calibration targets, launch calibration, and track task progress.",
    required_capabilities=(CapabilityFamily.CALIBRATION,),
    steps=(
        ProcedureStep(
            "list_targets",
            orchestrator_action(OrchestratorProcedureAction.LIST_CALIBRATION_TARGETS),
            "List calibration targets.",
            timeout_s=10.0,
        ),
        ProcedureStep(
            "start",
            orchestrator_action(OrchestratorProcedureAction.START_CALIBRATION),
            "Start calibration for the selected targets.",
            timeout_s=30.0,
            cancellation=ProcedureCancellationPolicy(
                mode=CancellationMode.SAFE_POINT,
                cancel_action=orchestrator_action(OrchestratorProcedureAction.CANCEL_CALIBRATION),
                timeout_s=10.0,
            ),
            compensation=ProcedureCompensationSpec(
                action=orchestrator_action(OrchestratorProcedureAction.CANCEL_CALIBRATION),
                description="Cancel partially started calibration tasks.",
                triggers=(
                    CompensationTrigger.ON_FAILURE,
                    CompensationTrigger.ON_CANCEL,
                    CompensationTrigger.ON_TIMEOUT,
                ),
                timeout_s=30.0,
            ),
        ),
        ProcedureStep(
            "track",
            orchestrator_action(OrchestratorProcedureAction.TRACK_CALIBRATION),
            "Track calibration progress until completion.",
            timeout_s=120.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=2.0),
        ),
    ),
    default_timeout_s=60.0,
    cancellation_policy=ProcedureCancellationPolicy(
        mode=CancellationMode.SAFE_POINT,
        cancel_action=orchestrator_action(OrchestratorProcedureAction.CANCEL_CALIBRATION),
        timeout_s=15.0,
    ),
    rollback_strategy=RollbackStrategy.REVERSE_COMPENSATION,
    idempotency_policy=ProcedureIdempotencyPolicy(
        mode=IdempotencyMode.STRICT,
        key_fields=("deployment_id", "target_ids"),
        conflict_policy=IdempotencyConflictPolicy.REJECT_DUPLICATE,
        cache_window_s=120.0,
    ),
    operator_interventions=(
        OperatorInterventionPoint(
            id="pose_robot_for_calibration",
            step_id="start",
            timing=InterventionTiming.BEFORE_STEP,
            instruction="Move robot to safe calibration posture before starting.",
        ),
    ),
)

MOVE_PROCEDURE = ProcedureDefinition(
    id="move_default",
    kind=ProcedureKind.MOVE,
    description="Resolve a normalized primitive and execute it safely.",
    required_capabilities=(CapabilityFamily.JOINT_MOTION,),
    steps=(
        ProcedureStep(
            "read_state",
            adapter_action(AdapterProcedureAction.GET_STATE),
            "Read current state before moving.",
            timeout_s=5.0,
        ),
        ProcedureStep(
            "resolve_primitive",
            orchestrator_action(OrchestratorProcedureAction.RESOLVE_PRIMITIVE),
            "Resolve the normalized primitive.",
            preconditions=(
                ProcedurePrecondition(
                    id="request_has_primitive",
                    source=PreconditionSource.INPUT,
                    key="primitive_name",
                    operator=PreconditionOperator.EXISTS,
                    description="Move procedure requires a normalized primitive identifier.",
                ),
            ),
            timeout_s=5.0,
        ),
        ProcedureStep(
            "execute",
            adapter_action(AdapterProcedureAction.EXECUTE_PRIMITIVE),
            "Execute the primitive through the adapter.",
            timeout_s=30.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=1.0),
            cancellation=ProcedureCancellationPolicy(
                mode=CancellationMode.IMMEDIATE,
                cancel_action=adapter_action(AdapterProcedureAction.STOP),
                timeout_s=3.0,
            ),
            compensation=ProcedureCompensationSpec(
                action=adapter_action(AdapterProcedureAction.STOP),
                description="Stop motion and hold position when execution aborts.",
                triggers=(
                    CompensationTrigger.ON_FAILURE,
                    CompensationTrigger.ON_CANCEL,
                    CompensationTrigger.ON_TIMEOUT,
                ),
                timeout_s=5.0,
                best_effort=False,
            ),
            idempotency=ProcedureIdempotencyPolicy(
                mode=IdempotencyMode.STRICT,
                key_fields=("session_id", "primitive_name", "primitive_args_hash"),
                conflict_policy=IdempotencyConflictPolicy.REJECT_DUPLICATE,
                cache_window_s=15.0,
            ),
        ),
    ),
    default_timeout_s=20.0,
    cancellation_policy=ProcedureCancellationPolicy(
        mode=CancellationMode.IMMEDIATE,
        cancel_action=adapter_action(AdapterProcedureAction.STOP),
        timeout_s=3.0,
    ),
    rollback_strategy=RollbackStrategy.REVERSE_COMPENSATION,
    idempotency_policy=ProcedureIdempotencyPolicy(
        mode=IdempotencyMode.STRICT,
        key_fields=("session_id", "primitive_name", "primitive_args_hash"),
        conflict_policy=IdempotencyConflictPolicy.REJECT_DUPLICATE,
        cache_window_s=30.0,
    ),
    operator_interventions=(
        OperatorInterventionPoint(
            id="confirm_high_risk_motion",
            step_id="execute",
            timing=InterventionTiming.BEFORE_STEP,
            instruction="Confirm the workspace is clear before executing high-risk motion.",
            blocking=True,
        ),
    ),
)

DEBUG_PROCEDURE = ProcedureDefinition(
    id="debug_default",
    kind=ProcedureKind.DEBUG,
    description="Collect environment probe, state, sensor snapshots, and debug bundle.",
    required_capabilities=(CapabilityFamily.DIAGNOSTICS,),
    steps=(
        ProcedureStep(
            "probe_env",
            adapter_action(AdapterProcedureAction.PROBE_ENV),
            "Probe environment health.",
            timeout_s=8.0,
        ),
        ProcedureStep(
            "state",
            adapter_action(AdapterProcedureAction.GET_STATE),
            "Read normalized state.",
            timeout_s=8.0,
        ),
        ProcedureStep(
            "sensor",
            adapter_action(AdapterProcedureAction.CAPTURE_SENSOR),
            "Capture a primary sensor snapshot if available.",
            preconditions=(
                ProcedurePrecondition(
                    id="sensor_available",
                    source=PreconditionSource.RUNTIME,
                    key="primary_sensor_id",
                    operator=PreconditionOperator.EXISTS,
                    required=False,
                    description="Sensor capture is optional when no sensor is configured.",
                ),
            ),
            timeout_s=10.0,
        ),
        ProcedureStep(
            "bundle",
            adapter_action(AdapterProcedureAction.DEBUG_SNAPSHOT),
            "Collect the debug bundle.",
            timeout_s=15.0,
        ),
    ),
    default_timeout_s=15.0,
    cancellation_policy=ProcedureCancellationPolicy(
        mode=CancellationMode.IMMEDIATE,
        cancel_action=adapter_action(AdapterProcedureAction.STOP),
        timeout_s=5.0,
    ),
    rollback_strategy=RollbackStrategy.NONE,
    idempotency_policy=ProcedureIdempotencyPolicy(
        mode=IdempotencyMode.BEST_EFFORT,
        key_fields=("session_id", "debug_scope"),
        conflict_policy=IdempotencyConflictPolicy.REUSE_RESULT,
        cache_window_s=10.0,
    ),
)

RESET_PROCEDURE = ProcedureDefinition(
    id="reset_default",
    kind=ProcedureKind.RESET,
    description="Stop active work, recover if needed, and reset to the default safe pose.",
    required_capabilities=(CapabilityFamily.RECOVERY,),
    steps=(
        ProcedureStep(
            "stop",
            adapter_action(AdapterProcedureAction.STOP),
            "Stop active motion or tasks.",
            timeout_s=5.0,
        ),
        ProcedureStep(
            "recover",
            adapter_action(AdapterProcedureAction.RECOVER),
            "Run recovery if the system is in a bad state.",
            preconditions=(
                ProcedurePrecondition(
                    id="error_state_present",
                    source=PreconditionSource.RUNTIME,
                    key="status",
                    operator=PreconditionOperator.IN_SET,
                    expected=("error", "busy"),
                    required=False,
                    description="Recover is optional if runtime is already healthy.",
                ),
            ),
            timeout_s=20.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=1.0),
        ),
        ProcedureStep(
            "reset",
            adapter_action(AdapterProcedureAction.RESET),
            "Reset to the default safe pose or mode.",
            timeout_s=20.0,
            retry_policy=ProcedureRetryPolicy(max_retries=1, backoff_s=1.0),
            compensation=ProcedureCompensationSpec(
                action=adapter_action(AdapterProcedureAction.RECOVER),
                description="Run recovery if reset leaves system unstable.",
                triggers=(
                    CompensationTrigger.ON_FAILURE,
                    CompensationTrigger.ON_TIMEOUT,
                ),
                timeout_s=15.0,
            ),
            idempotency=ProcedureIdempotencyPolicy(
                mode=IdempotencyMode.STRICT,
                key_fields=("session_id", "reset_mode"),
                conflict_policy=IdempotencyConflictPolicy.REUSE_RESULT,
                cache_window_s=20.0,
            ),
        ),
    ),
    default_timeout_s=20.0,
    cancellation_policy=ProcedureCancellationPolicy(
        mode=CancellationMode.NON_CANCELLABLE,
        timeout_s=5.0,
    ),
    rollback_strategy=RollbackStrategy.REVERSE_COMPENSATION,
    idempotency_policy=ProcedureIdempotencyPolicy(
        mode=IdempotencyMode.STRICT,
        key_fields=("session_id", "reset_mode"),
        conflict_policy=IdempotencyConflictPolicy.REUSE_RESULT,
        cache_window_s=60.0,
    ),
    operator_interventions=(
        OperatorInterventionPoint(
            id="clear_workspace",
            step_id="reset",
            timing=InterventionTiming.BEFORE_STEP,
            instruction="Clear workspace around robot before reset.",
        ),
    ),
)

DEFAULT_PROCEDURES = (
    CONNECT_PROCEDURE,
    CALIBRATE_PROCEDURE,
    MOVE_PROCEDURE,
    DEBUG_PROCEDURE,
    RESET_PROCEDURE,
)
