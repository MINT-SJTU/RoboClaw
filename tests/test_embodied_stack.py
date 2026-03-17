from datetime import datetime, timezone
from pathlib import Path

from roboclaw.embodied import RGB_CAMERA, SO101_ROBOT, build_default_catalog
from roboclaw.embodied.definition.foundation.schema import (
    CarrierKind,
    CommandMode,
    CompletionSemantics,
    RobotType,
    TransportKind,
    ValueUnit,
)
from roboclaw.embodied.definition.systems.assemblies import (
    AssemblyBlueprint,
    ControlGroup,
    FailureDomain,
    FrameTransform,
    ResourceLockScope,
    ResourceOwnership,
    RobotAttachment,
    SafetyBoundary,
    SafetyZone,
    ToolAttachment,
    Transform3D,
    compose_assemblies,
)
from roboclaw.embodied.definition.systems.assemblies.model import SensorAttachment
from roboclaw.embodied.execution.integration.adapters import (
    AdapterBinding,
    AdapterCompatibilitySpec,
    AdapterHealthMode,
    AdapterLifecycleContract,
    AdapterOperation,
    CompatibilityComponent,
    DegradedModeSpec,
    DependencyKind,
    DependencySpec,
    ErrorCategory,
    ErrorCodeSpec,
    OperationTimeout,
    TimeoutPolicy,
    VersionConstraint,
)
from roboclaw.embodied.execution.integration.bridges import (
    ARM_HAND_BRIDGE,
    DEFAULT_DOMAIN_BRIDGES,
    DRONE_BRIDGE,
    HUMANOID_WHOLE_BODY_BRIDGE,
    MOBILE_BASE_FLEET_BRIDGE,
    SIMULATOR_BRIDGE,
    BridgeDomain,
    BridgeKind,
    DomainBridgeContract,
)
from roboclaw.embodied.execution.integration.carriers.real import build_real_ros2_target
from roboclaw.embodied.execution.integration.transports.ros2 import build_standard_ros2_contract
from roboclaw.embodied.execution.observability import (
    RawEvidenceHandle,
    TelemetryEvent,
    TelemetryKind,
    TelemetryPhase,
    TelemetrySeverity,
)
from roboclaw.embodied.execution.orchestration.procedures import (
    AdapterProcedureAction,
    CancellationMode,
    CompensationTrigger,
    DEFAULT_PROCEDURES,
    IdempotencyMode,
    OrchestratorProcedureAction,
    ProcedureActionTarget,
    RollbackStrategy,
)
from roboclaw.embodied.execution.orchestration.runtime import RuntimeManager, RuntimeStatus
from roboclaw.embodied.workspace import (
    WorkspaceAssetKind,
    WorkspaceInspectOptions,
    WorkspaceIssueLevel,
    WorkspaceLintProfile,
    WorkspaceMigrationPolicy,
    WorkspaceProvenance,
    WorkspaceValidationStage,
    inspect_workspace_assets,
)


def _workspace_blueprint() -> AssemblyBlueprint:
    return AssemblyBlueprint(
        id="workspace_so101",
        name="Workspace SO101",
        description="Workspace-generated embodied assembly.",
        robots=(
            RobotAttachment(
                attachment_id="primary",
                robot_id="so101",
            ),
        ),
        sensors=(
            SensorAttachment(
                attachment_id="wrist_camera",
                sensor_id="rgb_camera",
                mount="wrist",
                mount_frame="tool0",
                mount_transform=Transform3D(),
            ),
        ),
        execution_targets=(
            build_real_ros2_target(
                target_id="real",
                description="Real target",
                ros2=build_standard_ros2_contract("workspace_so101", "real"),
            ),
        ),
        default_execution_target_id="real",
        frame_transforms=(
            FrameTransform(
                parent_frame="world",
                child_frame="base_link",
                transform=Transform3D(),
            ),
            FrameTransform(
                parent_frame="base_link",
                child_frame="tool0",
                transform=Transform3D(),
            ),
        ),
        tools=(
            ToolAttachment(
                attachment_id="primary_tool",
                robot_attachment_id="primary",
                tool_id="parallel_gripper",
                mount_frame="tool0",
                tcp_frame="tcp",
                kind="end_effector",
            ),
        ),
        control_groups=(
            ControlGroup(
                id="manipulation",
                robot_attachment_ids=("primary",),
                sensor_attachment_ids=("wrist_camera",),
                mode_hints=("position",),
            ),
        ),
        default_control_group_id="manipulation",
        safety_zones=(
            SafetyZone(
                id="workspace_zone",
                frame="base_link",
                min_xyz=(-0.5, -0.5, 0.0),
                max_xyz=(0.5, 0.5, 0.8),
            ),
        ),
        safety_boundaries=(
            SafetyBoundary(
                id="manipulation_safety",
                control_group_ids=("manipulation",),
                robot_attachment_ids=("primary",),
                sensor_attachment_ids=("wrist_camera",),
                zone_ids=("workspace_zone",),
                max_linear_speed_mps=0.2,
                max_joint_speed_scale=0.5,
            ),
        ),
        failure_domains=(
            FailureDomain(
                id="arm_cell",
                robot_attachment_ids=("primary",),
                sensor_attachment_ids=("wrist_camera",),
                target_ids=("real",),
                containment_actions=("stop", "recover"),
            ),
        ),
        resource_ownerships=(
            ResourceOwnership(
                id="manipulation_lock",
                control_group_id="manipulation",
                resource_ids=("joint_controller", "camera_stream"),
                lock_scope=ResourceLockScope.EXCLUSIVE,
                robot_attachment_ids=("primary",),
                sensor_attachment_ids=("wrist_camera",),
                failure_domain_id="arm_cell",
            ),
        ),
    )


def test_embodied_catalog_contains_reusable_definitions_only() -> None:
    catalog = build_default_catalog()

    assert catalog.robots.get("so101").robot_type == RobotType.ARM
    assert catalog.sensors.get("rgb_camera").default_topic_name == "image_raw"
    assert catalog.assemblies.list() == ()
    assert len(catalog.bridges.list()) == len(DEFAULT_DOMAIN_BRIDGES)
    assert catalog.adapters.for_assembly("workspace_so101") == ()
    assert catalog.deployments.for_assembly("workspace_so101") == ()
    assert RGB_CAMERA.supports_intrinsics is True


def test_domain_bridge_contract_is_machine_checkable() -> None:
    catalog = build_default_catalog()

    domains = {bridge.domain for bridge in catalog.bridges.list()}
    assert BridgeDomain.ARM_HAND in domains
    assert BridgeDomain.HUMANOID_WHOLE_BODY in domains
    assert BridgeDomain.MOBILE_BASE_FLEET in domains
    assert BridgeDomain.DRONE in domains
    assert BridgeDomain.SIMULATOR in domains

    arm_bridge = catalog.bridges.get(ARM_HAND_BRIDGE.id)
    assert arm_bridge.kind == BridgeKind.ROS2_CONTROL
    assert arm_bridge.control_surfaces[0].id == "joint_trajectory"
    assert arm_bridge.supports_robot_type(RobotType.ARM)

    drone_bridge = catalog.bridges.get(DRONE_BRIDGE.id)
    assert drone_bridge.supports_robot_type(RobotType.DRONE)

    adapter = AdapterBinding(
        id="workspace_ros2_adapter_with_bridge",
        assembly_id="workspace_so101",
        transport=TransportKind.ROS2,
        implementation="workspace.adapters.ros2:Adapter",
        supported_targets=("real",),
        bridge_id=ARM_HAND_BRIDGE.id,
        compatibility=AdapterCompatibilitySpec(
            adapter_api_version="1.0",
            constraints=(
                VersionConstraint(
                    component=CompatibilityComponent.TRANSPORT,
                    target="ros2",
                    requirement=">=1.0,<2.0",
                ),
                VersionConstraint(
                    component=CompatibilityComponent.BRIDGE,
                    target=ARM_HAND_BRIDGE.id,
                    requirement=">=1.0,<2.0",
                ),
            ),
        ),
    )
    assert adapter.bridge_id == ARM_HAND_BRIDGE.id
    assert len(catalog.bridges.for_domain(BridgeDomain.HUMANOID_WHOLE_BODY)) == 1
    assert len(catalog.bridges.for_domain(BridgeDomain.MOBILE_BASE_FLEET)) == 1
    assert len(catalog.bridges.for_domain(BridgeDomain.SIMULATOR)) == 1
    assert isinstance(HUMANOID_WHOLE_BODY_BRIDGE, DomainBridgeContract)
    assert isinstance(MOBILE_BASE_FLEET_BRIDGE, DomainBridgeContract)
    assert isinstance(SIMULATOR_BRIDGE, DomainBridgeContract)


def test_workspace_blueprint_can_be_composed_into_a_variant() -> None:
    base = _workspace_blueprint()
    overhead_variant = base.remap_sensor(
        "wrist_camera",
        to_mount="overhead",
    )
    composed = compose_assemblies(base, overhead_variant).build()

    assert composed.sensors == (
        SensorAttachment(
            attachment_id="wrist_camera",
            sensor_id="rgb_camera",
            mount="overhead",
            mount_frame="tool0",
            mount_transform=Transform3D(),
            config=None,
            optional=False,
        ),
    )
    assert composed.execution_target("real").carrier == CarrierKind.REAL
    assert composed.default_control_group_id == "manipulation"
    assert composed.control_group().id == "manipulation"
    assert composed.frame_transforms[0].child_frame == "base_link"
    assert composed.tools[0].attachment_id == "primary_tool"


def test_runtime_manager_tracks_active_session() -> None:
    manager = RuntimeManager()
    session = manager.create(
        session_id="demo",
        assembly_id="workspace_so101",
        target_id="real",
        deployment_id="workspace_local",
        adapter_id="workspace_ros2_adapter",
    )

    manager.mark_status("demo", RuntimeStatus.READY)

    assert session.assembly_id == "workspace_so101"
    assert session.adapter_id == "workspace_ros2_adapter"
    assert session.status == RuntimeStatus.READY
    assert SO101_ROBOT.suggested_sensor_ids == ("rgb_camera",)


def test_so101_action_and_observation_contracts_are_machine_checkable() -> None:
    move_joint = SO101_ROBOT.primitive("move_joint")
    assert move_joint is not None
    assert move_joint.command_mode == CommandMode.POSITION
    assert move_joint.parameters[0].unit == ValueUnit.RADIAN
    assert move_joint.action_schema is not None
    assert move_joint.action_schema.command_mode == CommandMode.POSITION
    assert move_joint.action_schema.parameter_order == ("positions",)
    assert move_joint.completion is not None
    assert move_joint.completion.semantics == CompletionSemantics.GOAL_REACHED

    scan = SO101_ROBOT.primitive("scan_panorama")
    assert scan is not None
    assert scan.command_mode == CommandMode.MISSION

    assert SO101_ROBOT.observation_schema.id == "so101_observation_v1"
    assert len(SO101_ROBOT.observation_schema.fields) > 0
    assert SO101_ROBOT.health_schema.id == "so101_health_v1"
    assert len(SO101_ROBOT.health_schema.fields) > 0


def test_assembly_topology_contract_is_machine_checkable() -> None:
    assembly = _workspace_blueprint().build()

    assert assembly.default_execution_target_id == "real"
    assert assembly.execution_target().id == "real"
    assert assembly.default_control_group_id == "manipulation"
    assert assembly.control_group().robot_attachment_ids == ("primary",)
    assert assembly.tools[0].robot_attachment_id == "primary"
    assert assembly.frame_transforms[1].parent_frame == "base_link"
    assert assembly.safety_zones[0].id == "workspace_zone"
    assert assembly.safety_boundaries[0].zone_ids == ("workspace_zone",)
    assert assembly.failure_domains[0].containment_actions == ("stop", "recover")
    assert assembly.resource_ownerships[0].control_group_id == "manipulation"


def test_adapter_lifecycle_contract_is_machine_checkable() -> None:
    lifecycle = AdapterLifecycleContract(
        dependencies=(
            DependencySpec(
                id="ros2_bridge",
                kind=DependencyKind.ROS2_NODE,
                description="ROS2 bridge node must be available.",
            ),
        ),
        timeout_policy=TimeoutPolicy(
            default_timeout_s=10.0,
            operations=(
                OperationTimeout(
                    operation=AdapterOperation.CONNECT,
                    timeout_s=30.0,
                    retries=2,
                    backoff_s=1.0,
                ),
            ),
        ),
        error_codes=(
            ErrorCodeSpec(
                code="DEP_MISSING",
                category=ErrorCategory.DEPENDENCY,
                description="Dependency check failed.",
                recoverable=False,
                related_operation=AdapterOperation.DEPENDENCY_CHECK,
            ),
        ),
    )
    binding = AdapterBinding(
        id="workspace_ros2_adapter",
        assembly_id="workspace_so101",
        transport=TransportKind.ROS2,
        implementation="workspace.adapters.ros2:Adapter",
        supported_targets=("real",),
        lifecycle=lifecycle,
        degraded_modes=(
            DegradedModeSpec(
                mode=AdapterHealthMode.DEGRADED,
                description="Transport degraded; only stop/recover paths are available.",
                allowed_operations=(
                    AdapterOperation.READY,
                    AdapterOperation.STOP,
                    AdapterOperation.RECOVER,
                ),
                entered_on_error_codes=("DEP_MISSING",),
            ),
        ),
        compatibility=AdapterCompatibilitySpec(
            adapter_api_version="1.0",
            constraints=(
                VersionConstraint(
                    component=CompatibilityComponent.TRANSPORT,
                    target="ros2",
                    requirement=">=1.0,<2.0",
                ),
                VersionConstraint(
                    component=CompatibilityComponent.BRIDGE,
                    target=ARM_HAND_BRIDGE.id,
                    requirement=">=1.0,<2.0",
                    required=False,
                ),
            ),
        ),
    )

    assert binding.lifecycle.supports(AdapterOperation.CONNECT)
    assert binding.lifecycle.supports(AdapterOperation.READY)
    assert binding.lifecycle.timeout_policy.timeout_for(AdapterOperation.CONNECT).timeout_s == 30.0
    assert binding.lifecycle.error_codes[0].category == ErrorCategory.DEPENDENCY
    assert binding.degraded_modes[0].mode == AdapterHealthMode.DEGRADED
    assert binding.degraded_modes[0].entered_on_error_codes == ("DEP_MISSING",)
    assert binding.compatibility.for_component(CompatibilityComponent.TRANSPORT)[0].target == "ros2"


def test_procedure_contract_is_machine_checkable() -> None:
    connect = next(procedure for procedure in DEFAULT_PROCEDURES if procedure.id == "connect_default")

    assert connect.required_capabilities
    assert connect.step_edges
    assert connect.entry_step_ids == ("probe_env",)
    assert connect.terminal_step_ids == ("verify_state",)

    connect_step = next(step for step in connect.steps if step.id == "connect")
    assert connect_step.timeout_s == 30.0
    assert connect_step.retry_policy.max_retries == 2
    assert connect.cancellation_policy.mode == CancellationMode.SAFE_POINT
    assert connect.rollback_strategy == RollbackStrategy.REVERSE_COMPENSATION
    assert connect.idempotency_policy.mode == IdempotencyMode.BEST_EFFORT
    assert connect.idempotency_policy.key_fields == ("deployment_id", "target_id")
    select_target_step = next(step for step in connect.steps if step.id == "select_target")
    assert select_target_step.action.target == ProcedureActionTarget.ORCHESTRATOR
    assert select_target_step.action.name == OrchestratorProcedureAction.RESOLVE_TARGET.value

    assert connect_step.cancellation is not None
    assert connect_step.cancellation.mode == CancellationMode.IMMEDIATE
    assert connect_step.action.target == ProcedureActionTarget.ADAPTER
    assert connect_step.action.name == AdapterProcedureAction.CONNECT.value
    assert connect_step.cancellation.cancel_action is not None
    assert connect_step.cancellation.cancel_action.target == ProcedureActionTarget.ADAPTER
    assert connect_step.cancellation.cancel_action.name == AdapterProcedureAction.DISCONNECT.value
    assert connect_step.compensation is not None
    assert connect_step.compensation.action.target == ProcedureActionTarget.ADAPTER
    assert connect_step.compensation.action.name == AdapterProcedureAction.DISCONNECT.value
    assert CompensationTrigger.ON_CANCEL in connect_step.compensation.triggers
    assert connect_step.idempotency is not None
    assert connect_step.idempotency.mode == IdempotencyMode.BEST_EFFORT
    assert connect.operator_interventions[0].step_id == "connect"

    reset = next(procedure for procedure in DEFAULT_PROCEDURES if procedure.id == "reset_default")
    assert reset.cancellation_policy.mode == CancellationMode.NON_CANCELLABLE
    assert reset.idempotency_policy.mode == IdempotencyMode.STRICT


def test_telemetry_contract_is_machine_checkable() -> None:
    event = TelemetryEvent(
        timestamp=datetime.now(timezone.utc),
        correlation_id="corr-123",
        source_component="runtime.session_manager",
        kind=TelemetryKind.ACTION,
        severity=TelemetrySeverity.INFO,
        message="Executed move primitive.",
        payload={"primitive": "move_joint", "status": "success"},
        raw_evidence=(
            RawEvidenceHandle(
                id="bag-1",
                uri="file:///tmp/trace/demo.bag",
                media_type="application/x-rosbag2",
            ),
        ),
        phase=TelemetryPhase.COMPLETE,
        tags=("move", "primitive"),
        replay_handle="replay://demo/corr-123",
    )

    assert event.correlation_id == "corr-123"
    assert event.source_component == "runtime.session_manager"
    assert event.kind == TelemetryKind.ACTION
    assert event.severity == TelemetrySeverity.INFO
    assert event.raw_evidence[0].id == "bag-1"


def test_workspace_asset_contract_detects_duplicate_ids(tmp_path: Path) -> None:
    robots_dir = tmp_path / "embodied" / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)

    robots_dir.joinpath("robot_a.py").write_text(
        "\n".join(
            [
                "from roboclaw.embodied import SO101_ROBOT",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ROBOT,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ROBOT,",
                ")",
                "",
                "ROBOT = SO101_ROBOT",
                "",
            ]
        ),
        encoding="utf-8",
    )
    robots_dir.joinpath("robot_b.py").write_text(
        "\n".join(
            [
                "from roboclaw.embodied import SO101_ROBOT",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ROBOT,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ROBOT,",
                ")",
                "",
                "ROBOT = SO101_ROBOT",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workspace_assets(tmp_path)

    assert report.has_errors
    assert any(issue.code == "DUPLICATE_ASSET_ID" for issue in report.issues)
    assert any(issue.stage == WorkspaceValidationStage.SCHEMA for issue in report.issues)
    assert report.loaded_counts[WorkspaceAssetKind.ROBOT] == 1


def test_workspace_asset_contract_supports_migration_policy(tmp_path: Path) -> None:
    robots_dir = tmp_path / "embodied" / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)

    robots_dir.joinpath("future_schema.py").write_text(
        "\n".join(
            [
                "from roboclaw.embodied import SO101_ROBOT",
                "from roboclaw.embodied.workspace import (",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                "    WorkspaceMigrationPolicy,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ROBOT,",
                "    schema_version='2.0',",
                "    export_convention=WorkspaceExportConvention.ROBOT,",
                "    migration_policy=WorkspaceMigrationPolicy.ACCEPT_UNSUPPORTED,",
                ")",
                "",
                "ROBOT = SO101_ROBOT",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workspace_assets(tmp_path)

    assert report.has_errors is False
    assert report.has_warnings
    assert any(issue.level == WorkspaceIssueLevel.WARNING for issue in report.issues)
    assert any(issue.code == "UNSUPPORTED_SCHEMA_VERSION" for issue in report.issues)
    assert any(issue.stage == WorkspaceValidationStage.SCHEMA for issue in report.issues)
    assert any("accept_unsupported" in issue.message for issue in report.issues)
    assert report.loaded_counts[WorkspaceAssetKind.ROBOT] == 1
    assert WorkspaceMigrationPolicy.ACCEPT_UNSUPPORTED.value == "accept_unsupported"


def test_workspace_dry_run_reports_staged_provenance(tmp_path: Path) -> None:
    robots_dir = tmp_path / "embodied" / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)

    robots_dir.joinpath("robot.py").write_text(
        "\n".join(
            [
                "from roboclaw.embodied import SO101_ROBOT",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                "    WorkspaceProvenance,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ROBOT,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ROBOT,",
                "    provenance=WorkspaceProvenance(",
                "        source='workspace',",
                "        generator='roboclaw_agent',",
                "        generated_by='agent:test',",
                "        generated_at='2026-03-17T15:00:00Z',",
                "    ),",
                ")",
                "",
                "ROBOT = SO101_ROBOT",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workspace_assets(tmp_path)

    assert report.has_errors is False
    assert report.staged_assets
    staged = report.staged_assets[0]
    assert staged.kind == WorkspaceAssetKind.ROBOT
    assert staged.provenance.generator == "roboclaw_agent"
    assert staged.provenance.generated_by == "agent:test"
    assert report.stage_counts[WorkspaceValidationStage.LINT] == 0


def test_workspace_strict_lint_blocks_missing_contract_metadata(tmp_path: Path) -> None:
    robots_dir = tmp_path / "embodied" / "robots"
    robots_dir.mkdir(parents=True, exist_ok=True)

    robots_dir.joinpath("legacy_robot.py").write_text(
        "\n".join(
            [
                "from roboclaw.embodied import SO101_ROBOT",
                "ROBOT = SO101_ROBOT",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_workspace_assets(
        tmp_path,
        options=WorkspaceInspectOptions(lint_profile=WorkspaceLintProfile.STRICT),
    )

    assert report.has_errors
    assert any(issue.code == "CONTRACT_METADATA_MISSING" for issue in report.issues)
    assert any(issue.stage == WorkspaceValidationStage.LINT for issue in report.issues)
