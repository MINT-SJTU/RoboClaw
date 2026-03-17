"""SO101 robot definition."""

from __future__ import annotations

from roboclaw.embodied.definition.components.robots.model import PrimitiveSpec, RobotManifest
from roboclaw.embodied.definition.foundation.schema import (
    ActionSchema,
    CapabilityFamily,
    CommandMode,
    CompletionSemantics,
    CompletionSpec,
    FeedbackMode,
    HealthFieldSpec,
    HealthSchema,
    HealthLevel,
    ObservationFieldSpec,
    ObservationSchema,
    ParameterSpec,
    PrimitiveKind,
    RobotType,
    SafetyProfile,
    ToleranceSpec,
    ValueUnit,
)

SO101_PRIMITIVES = (
    PrimitiveSpec(
        name="move_joint",
        kind=PrimitiveKind.MOTION,
        capability_family=CapabilityFamily.JOINT_MOTION,
        command_mode=CommandMode.POSITION,
        description="Move one or more joints to target positions.",
        parameters=(
            ParameterSpec(
                "positions",
                "dict[str,float]",
                "Joint name to target value map.",
                True,
                ValueUnit.RADIAN,
                "joint_space",
            ),
        ),
        tolerance=ToleranceSpec(absolute=0.02, settle_time_s=0.2),
        action_schema=ActionSchema(
            id="so101_move_joint_action_v1",
            command_mode=CommandMode.POSITION,
            feedback_mode=FeedbackMode.TRAJECTORY_STATUS,
            parameter_order=("positions",),
            command_frame="joint_space",
            command_rate_hz=30.0,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=5.0,
        ),
        backed_by=("send_action",),
    ),
    PrimitiveSpec(
        name="move_cartesian_delta",
        kind=PrimitiveKind.MOTION,
        capability_family=CapabilityFamily.CARTESIAN_MOTION,
        command_mode=CommandMode.CARTESIAN_DELTA,
        description="Move the end effector by a small Cartesian delta in the base frame.",
        parameters=(
            ParameterSpec("dx", "float", "Delta x in meters.", False, ValueUnit.METER, "base_link"),
            ParameterSpec("dy", "float", "Delta y in meters.", False, ValueUnit.METER, "base_link"),
            ParameterSpec("dz", "float", "Delta z in meters.", False, ValueUnit.METER, "base_link"),
        ),
        tolerance=ToleranceSpec(absolute=0.005, settle_time_s=0.15),
        action_schema=ActionSchema(
            id="so101_move_cartesian_delta_action_v1",
            command_mode=CommandMode.CARTESIAN_DELTA,
            feedback_mode=FeedbackMode.TRAJECTORY_STATUS,
            parameter_order=("dx", "dy", "dz"),
            command_frame="base_link",
            command_rate_hz=20.0,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=3.0,
        ),
        backed_by=("move_ee_delta",),
    ),
    PrimitiveSpec(
        name="spin_wrist",
        kind=PrimitiveKind.MOTION,
        capability_family=CapabilityFamily.JOINT_MOTION,
        command_mode=CommandMode.POSITION,
        description="Spin the wrist roll joint by a relative angle.",
        parameters=(
            ParameterSpec(
                "delta_deg",
                "float",
                "Relative wrist rotation in degrees.",
                True,
                ValueUnit.DEGREE,
                "wrist_roll_joint",
            ),
        ),
        tolerance=ToleranceSpec(absolute=2.0),
        action_schema=ActionSchema(
            id="so101_spin_wrist_action_v1",
            command_mode=CommandMode.POSITION,
            feedback_mode=FeedbackMode.TRAJECTORY_STATUS,
            parameter_order=("delta_deg",),
            command_frame="wrist_roll_joint",
            command_rate_hz=20.0,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=2.0,
        ),
        backed_by=("primitive_action:spin_left_or_right",),
    ),
    PrimitiveSpec(
        name="gripper_open",
        kind=PrimitiveKind.END_EFFECTOR,
        capability_family=CapabilityFamily.END_EFFECTOR,
        command_mode=CommandMode.DISCRETE_TRIGGER,
        description="Open the gripper.",
        action_schema=ActionSchema(
            id="so101_gripper_open_action_v1",
            command_mode=CommandMode.DISCRETE_TRIGGER,
            feedback_mode=FeedbackMode.EVENT_STATUS,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=2.0,
        ),
        backed_by=("primitive_action:gripper_open",),
    ),
    PrimitiveSpec(
        name="gripper_close",
        kind=PrimitiveKind.END_EFFECTOR,
        capability_family=CapabilityFamily.END_EFFECTOR,
        command_mode=CommandMode.DISCRETE_TRIGGER,
        description="Close the gripper.",
        action_schema=ActionSchema(
            id="so101_gripper_close_action_v1",
            command_mode=CommandMode.DISCRETE_TRIGGER,
            feedback_mode=FeedbackMode.EVENT_STATUS,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=2.0,
        ),
        backed_by=("primitive_action:gripper_close",),
    ),
    PrimitiveSpec(
        name="save_named_pose",
        kind=PrimitiveKind.POSE,
        capability_family=CapabilityFamily.NAMED_POSE,
        command_mode=CommandMode.DISCRETE_TRIGGER,
        description="Save the current pose under a stable name.",
        parameters=(
            ParameterSpec("name", "str", "Named pose identifier.", True),
        ),
        action_schema=ActionSchema(
            id="so101_save_named_pose_action_v1",
            command_mode=CommandMode.DISCRETE_TRIGGER,
            feedback_mode=FeedbackMode.EVENT_STATUS,
            parameter_order=("name",),
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.COMMAND_ACCEPTED,
        ),
        backed_by=("save_named_pose",),
    ),
    PrimitiveSpec(
        name="go_named_pose",
        kind=PrimitiveKind.POSE,
        capability_family=CapabilityFamily.NAMED_POSE,
        command_mode=CommandMode.WAYPOINT,
        description="Move to a named pose such as home, ready, rest, or work.",
        parameters=(
            ParameterSpec("name", "str", "Named pose identifier.", True),
        ),
        action_schema=ActionSchema(
            id="so101_go_named_pose_action_v1",
            command_mode=CommandMode.WAYPOINT,
            feedback_mode=FeedbackMode.TRAJECTORY_STATUS,
            parameter_order=("name",),
            command_rate_hz=10.0,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.GOAL_REACHED,
            timeout_s=5.0,
        ),
        backed_by=("move_to_named_pose",),
    ),
    PrimitiveSpec(
        name="scan_panorama",
        kind=PrimitiveKind.PERCEPTION,
        capability_family=CapabilityFamily.CAMERA,
        command_mode=CommandMode.MISSION,
        description="Run a shoulder-pan scan using an attached wrist camera.",
        parameters=(
            ParameterSpec("focus", "str", "What the scan should focus on."),
        ),
        action_schema=ActionSchema(
            id="so101_scan_panorama_action_v1",
            command_mode=CommandMode.MISSION,
            feedback_mode=FeedbackMode.EVENT_STATUS,
            parameter_order=("focus",),
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.EVENT_CONFIRMED,
            timeout_s=10.0,
        ),
        backed_by=("look_around_scan",),
    ),
    PrimitiveSpec(
        name="release_torque",
        kind=PrimitiveKind.MAINTENANCE,
        capability_family=CapabilityFamily.TORQUE_CONTROL,
        command_mode=CommandMode.DISCRETE_TRIGGER,
        description="Release torque on the arm while keeping the session alive.",
        action_schema=ActionSchema(
            id="so101_release_torque_action_v1",
            command_mode=CommandMode.DISCRETE_TRIGGER,
            feedback_mode=FeedbackMode.EVENT_STATUS,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.COMMAND_ACCEPTED,
        ),
        backed_by=("primitive_action:release_torque",),
    ),
    PrimitiveSpec(
        name="lock_torque",
        kind=PrimitiveKind.MAINTENANCE,
        capability_family=CapabilityFamily.TORQUE_CONTROL,
        command_mode=CommandMode.DISCRETE_TRIGGER,
        description="Re-enable torque and hold the current pose.",
        action_schema=ActionSchema(
            id="so101_lock_torque_action_v1",
            command_mode=CommandMode.DISCRETE_TRIGGER,
            feedback_mode=FeedbackMode.EVENT_STATUS,
        ),
        completion=CompletionSpec(
            semantics=CompletionSemantics.COMMAND_ACCEPTED,
        ),
        backed_by=("primitive_action:lock_torque",),
    ),
)

SO101_OBSERVATION_SCHEMA = ObservationSchema(
    id="so101_observation_v1",
    fields=(
        ObservationFieldSpec(
            name="joint_positions",
            value_type="dict[str,float]",
            description="Joint positions for all controllable joints.",
            unit=ValueUnit.RADIAN,
            frame="joint_space",
        ),
        ObservationFieldSpec(
            name="joint_velocities",
            value_type="dict[str,float]",
            description="Joint velocity estimate for all controllable joints.",
            unit=ValueUnit.RADIAN_PER_SECOND,
            frame="joint_space",
        ),
        ObservationFieldSpec(
            name="ee_position",
            value_type="dict[str,float]",
            description="End-effector Cartesian position in base frame.",
            unit=ValueUnit.METER,
            frame="base_link",
        ),
        ObservationFieldSpec(
            name="ee_orientation_rpy",
            value_type="dict[str,float]",
            description="End-effector orientation in roll/pitch/yaw.",
            unit=ValueUnit.RADIAN,
            frame="base_link",
        ),
        ObservationFieldSpec(
            name="gripper_open_ratio",
            value_type="float",
            description="Normalized gripper opening ratio.",
            unit=ValueUnit.PERCENT,
            frame="gripper",
        ),
    ),
    frequency_hz=30.0,
    notes=(
        "Observation schema is transport-independent and consumed by semantic skills.",
    ),
)

SO101_HEALTH_SCHEMA = HealthSchema(
    id="so101_health_v1",
    fields=(
        HealthFieldSpec(
            name="level",
            value_type="str",
            description="Normalized health level.",
        ),
        HealthFieldSpec(
            name="connection_state",
            value_type="str",
            description="Primary control connection state.",
        ),
        HealthFieldSpec(
            name="fault_code",
            value_type="str",
            description="Vendor or adapter fault code.",
        ),
        HealthFieldSpec(
            name="estop_latched",
            value_type="bool",
            description="Emergency stop latch state.",
        ),
    ),
    severity_levels=(
        HealthLevel.OK,
        HealthLevel.WARN,
        HealthLevel.ERROR,
        HealthLevel.STALE,
    ),
    notes=(
        "Health schema remains reusable across real and simulated execution targets.",
    ),
)

SO101_ROBOT = RobotManifest(
    id="so101",
    name="SO101",
    description="Standard SO101 single-arm robot definition independent from carriers and sensors.",
    robot_type=RobotType.ARM,
    capability_families=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.JOINT_MOTION,
        CapabilityFamily.CARTESIAN_MOTION,
        CapabilityFamily.END_EFFECTOR,
        CapabilityFamily.CAMERA,
        CapabilityFamily.CALIBRATION,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
        CapabilityFamily.NAMED_POSE,
        CapabilityFamily.TORQUE_CONTROL,
    ),
    primitives=SO101_PRIMITIVES,
    observation_schema=SO101_OBSERVATION_SCHEMA,
    health_schema=SO101_HEALTH_SCHEMA,
    default_named_poses=("home", "ready", "rest", "work"),
    suggested_sensor_ids=("rgb_camera",),
    safety=SafetyProfile(
        emergency_stop_required=True,
        supports_soft_stop=True,
        default_reset_mode="home",
        notes=(
            "Visual servo flows should stay above the robot manifest and consume normalized primitives.",
        ),
    ),
    setup_hints=(
        "For first-run setup, treat SO101 as a known framework robot and start intake immediately once the user names it.",
        "Default the integration path to ROS2 unless the user explicitly asks for another stack.",
        "For a real SO101, assume the common deployment path is a local USB/serial connection.",
        "Do not start by asking whether SO101 uses USB, serial, or IP. Ask for the concrete serial device only when deployment generation actually needs it and it cannot be inferred locally.",
    ),
    notes=(
        "SO101 only defines robot-local motion and maintenance capabilities here.",
        "Sensors and carrier targets are composed separately through assemblies.",
    ),
)
