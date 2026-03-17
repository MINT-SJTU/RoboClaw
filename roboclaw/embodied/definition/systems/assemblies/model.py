"""Assembly manifests compose robots, sensors, transports, and carriers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from roboclaw.embodied.execution.integration.carriers import ExecutionTarget

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


@dataclass(frozen=True)
class RobotAttachment:
    """Attach a robot manifest into one assembly."""

    attachment_id: str
    robot_id: str
    role: str = "primary"
    config: Any | None = None


@dataclass(frozen=True)
class SensorAttachment:
    """Attach a sensor manifest into one assembly."""

    attachment_id: str
    sensor_id: str
    mount: str
    mount_frame: str | None = None
    mount_transform: Transform3D | None = None
    config: Any | None = None
    optional: bool = False


@dataclass(frozen=True)
class Transform3D:
    """Rigid transform represented in XYZ translation + RPY rotation."""

    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FrameTransform:
    """Frame relation used by assembly topology."""

    parent_frame: str
    child_frame: str
    transform: Transform3D = field(default_factory=Transform3D)
    static: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolAttachment:
    """Attach a tool or end-effector to one robot attachment."""

    attachment_id: str
    robot_attachment_id: str
    tool_id: str
    mount_frame: str
    tcp_frame: str | None = None
    kind: str = "end_effector"
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ControlGroup:
    """Named control group spanning robot and sensor attachments."""

    id: str
    robot_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    sensor_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    mode_hints: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


class ResourceLockScope(StrEnum):
    """Lock scope for one resource ownership declaration."""

    EXCLUSIVE = "exclusive"
    SHARED_READ = "shared_read"
    SHARED_WRITE = "shared_write"


@dataclass(frozen=True)
class ResourceOwnership:
    """Ownership mapping for resources used by one control group."""

    id: str
    control_group_id: str
    resource_ids: tuple[str, ...]
    lock_scope: ResourceLockScope = ResourceLockScope.EXCLUSIVE
    robot_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    sensor_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    failure_domain_id: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Resource ownership id cannot be empty.")
        if not self.control_group_id.strip():
            raise ValueError(f"Resource ownership '{self.id}' control_group_id cannot be empty.")
        if not self.resource_ids:
            raise ValueError(f"Resource ownership '{self.id}' must define at least one resource id.")
        if any(not resource_id.strip() for resource_id in self.resource_ids):
            raise ValueError(f"Resource ownership '{self.id}' resource_ids cannot contain empty ids.")
        if self.failure_domain_id is not None and not self.failure_domain_id.strip():
            raise ValueError(
                f"Resource ownership '{self.id}' failure_domain_id cannot be empty when specified."
            )


@dataclass(frozen=True)
class SafetyZone:
    """A named safety zone for one assembly."""

    id: str
    frame: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Safety zone id cannot be empty.")
        if not self.frame.strip():
            raise ValueError(f"Safety zone '{self.id}' frame cannot be empty.")
        if any(low >= high for low, high in zip(self.min_xyz, self.max_xyz)):
            raise ValueError(
                f"Safety zone '{self.id}' requires min_xyz < max_xyz for all dimensions."
            )


@dataclass(frozen=True)
class SafetyBoundary:
    """Safety limits and zone references for one control scope."""

    id: str
    control_group_ids: tuple[str, ...] = field(default_factory=tuple)
    robot_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    sensor_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    zone_ids: tuple[str, ...] = field(default_factory=tuple)
    max_linear_speed_mps: float | None = None
    max_angular_speed_radps: float | None = None
    max_joint_speed_scale: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Safety boundary id cannot be empty.")
        if self.max_linear_speed_mps is not None and self.max_linear_speed_mps <= 0:
            raise ValueError(
                f"Safety boundary '{self.id}' max_linear_speed_mps must be > 0 when specified."
            )
        if self.max_angular_speed_radps is not None and self.max_angular_speed_radps <= 0:
            raise ValueError(
                f"Safety boundary '{self.id}' max_angular_speed_radps must be > 0 when specified."
            )
        if self.max_joint_speed_scale is not None and not (0 < self.max_joint_speed_scale <= 1.0):
            raise ValueError(
                f"Safety boundary '{self.id}' max_joint_speed_scale must be in (0, 1]."
            )


@dataclass(frozen=True)
class FailureDomain:
    """Fault containment grouping for assembly members."""

    id: str
    robot_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    sensor_attachment_ids: tuple[str, ...] = field(default_factory=tuple)
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    containment_actions: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Failure domain id cannot be empty.")
        if not (self.robot_attachment_ids or self.sensor_attachment_ids or self.target_ids):
            raise ValueError(
                f"Failure domain '{self.id}' must include at least one robot/sensor/target member."
            )
        if any(not action.strip() for action in self.containment_actions):
            raise ValueError(
                f"Failure domain '{self.id}' containment_actions cannot contain empty values."
            )


@dataclass(frozen=True)
class AssemblyManifest:
    """Composed system definition."""

    id: str
    name: str
    description: str
    robots: tuple[RobotAttachment, ...]
    sensors: tuple[SensorAttachment, ...]
    execution_targets: tuple[ExecutionTarget, ...]
    default_execution_target_id: str | None = None
    frame_transforms: tuple[FrameTransform, ...] = field(default_factory=tuple)
    tools: tuple[ToolAttachment, ...] = field(default_factory=tuple)
    control_groups: tuple[ControlGroup, ...] = field(default_factory=tuple)
    default_control_group_id: str | None = None
    safety_zones: tuple[SafetyZone, ...] = field(default_factory=tuple)
    safety_boundaries: tuple[SafetyBoundary, ...] = field(default_factory=tuple)
    failure_domains: tuple[FailureDomain, ...] = field(default_factory=tuple)
    resource_ownerships: tuple[ResourceOwnership, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.robots:
            raise ValueError("Assembly manifest must contain at least one robot.")
        if not self.execution_targets:
            raise ValueError("Assembly manifest must declare at least one execution target.")

        target_ids = [target.id for target in self.execution_targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"Duplicate execution target ids in assembly '{self.id}'.")
        robot_attachment_ids = [robot.attachment_id for robot in self.robots]
        if len(set(robot_attachment_ids)) != len(robot_attachment_ids):
            raise ValueError(f"Duplicate robot attachment ids in assembly '{self.id}'.")
        sensor_attachment_ids = [sensor.attachment_id for sensor in self.sensors]
        if len(set(sensor_attachment_ids)) != len(sensor_attachment_ids):
            raise ValueError(f"Duplicate sensor attachment ids in assembly '{self.id}'.")
        frame_child_ids = [frame.child_frame for frame in self.frame_transforms]
        if len(set(frame_child_ids)) != len(frame_child_ids):
            raise ValueError(f"Duplicate frame child ids in assembly '{self.id}'.")
        zone_ids = [zone.id for zone in self.safety_zones]
        if len(set(zone_ids)) != len(zone_ids):
            raise ValueError(f"Duplicate safety zone ids in assembly '{self.id}'.")
        tool_attachment_ids = [tool.attachment_id for tool in self.tools]
        if len(set(tool_attachment_ids)) != len(tool_attachment_ids):
            raise ValueError(f"Duplicate tool attachment ids in assembly '{self.id}'.")
        control_group_ids = [group.id for group in self.control_groups]
        if len(set(control_group_ids)) != len(control_group_ids):
            raise ValueError(f"Duplicate control group ids in assembly '{self.id}'.")
        failure_domain_ids = [domain.id for domain in self.failure_domains]
        if len(set(failure_domain_ids)) != len(failure_domain_ids):
            raise ValueError(f"Duplicate failure domain ids in assembly '{self.id}'.")
        resource_ownership_ids = [ownership.id for ownership in self.resource_ownerships]
        if len(set(resource_ownership_ids)) != len(resource_ownership_ids):
            raise ValueError(f"Duplicate resource ownership ids in assembly '{self.id}'.")
        safety_boundary_ids = [boundary.id for boundary in self.safety_boundaries]
        if len(set(safety_boundary_ids)) != len(safety_boundary_ids):
            raise ValueError(f"Duplicate safety boundary ids in assembly '{self.id}'.")

        robot_attachment_set = set(robot_attachment_ids)
        sensor_attachment_set = set(sensor_attachment_ids)
        target_set = set(target_ids)
        control_group_set = set(control_group_ids)
        failure_domain_set = set(failure_domain_ids)
        zone_set = set(zone_ids)
        frame_set = set(frame.parent_frame for frame in self.frame_transforms) | set(frame_child_ids)
        frame_set.update({"world"})
        frame_set.update(
            tool_frame
            for tool in self.tools
            for tool_frame in (tool.mount_frame, tool.tcp_frame)
            if tool_frame is not None
        )

        for sensor in self.sensors:
            if sensor.mount_frame is not None and sensor.mount_frame not in frame_set:
                raise ValueError(
                    f"Sensor attachment '{sensor.attachment_id}' references unknown mount_frame "
                    f"'{sensor.mount_frame}' in assembly '{self.id}'."
                )
            if sensor.mount_transform is not None and sensor.mount_frame is None:
                raise ValueError(
                    f"Sensor attachment '{sensor.attachment_id}' defines mount_transform "
                    f"without mount_frame in assembly '{self.id}'."
                )

        for tool in self.tools:
            if tool.robot_attachment_id not in robot_attachment_set:
                raise ValueError(
                    f"Tool '{tool.attachment_id}' references unknown robot attachment "
                    f"'{tool.robot_attachment_id}' in assembly '{self.id}'."
                )
        for group in self.control_groups:
            missing_robots = set(group.robot_attachment_ids) - robot_attachment_set
            missing_sensors = set(group.sensor_attachment_ids) - sensor_attachment_set
            if missing_robots:
                raise ValueError(
                    f"Control group '{group.id}' references unknown robot attachments "
                    f"{sorted(missing_robots)} in assembly '{self.id}'."
                )
            if missing_sensors:
                raise ValueError(
                    f"Control group '{group.id}' references unknown sensor attachments "
                    f"{sorted(missing_sensors)} in assembly '{self.id}'."
                )
        for zone in self.safety_zones:
            if zone.frame not in frame_set:
                raise ValueError(
                    f"Safety zone '{zone.id}' references unknown frame '{zone.frame}' "
                    f"in assembly '{self.id}'."
                )
        for domain in self.failure_domains:
            missing_robots = set(domain.robot_attachment_ids) - robot_attachment_set
            missing_sensors = set(domain.sensor_attachment_ids) - sensor_attachment_set
            missing_targets = set(domain.target_ids) - target_set
            if missing_robots:
                raise ValueError(
                    f"Failure domain '{domain.id}' references unknown robot attachments "
                    f"{sorted(missing_robots)} in assembly '{self.id}'."
                )
            if missing_sensors:
                raise ValueError(
                    f"Failure domain '{domain.id}' references unknown sensor attachments "
                    f"{sorted(missing_sensors)} in assembly '{self.id}'."
                )
            if missing_targets:
                raise ValueError(
                    f"Failure domain '{domain.id}' references unknown execution targets "
                    f"{sorted(missing_targets)} in assembly '{self.id}'."
                )
        for ownership in self.resource_ownerships:
            if ownership.control_group_id not in control_group_set:
                raise ValueError(
                    f"Resource ownership '{ownership.id}' references unknown control group "
                    f"'{ownership.control_group_id}' in assembly '{self.id}'."
                )
            missing_robots = set(ownership.robot_attachment_ids) - robot_attachment_set
            missing_sensors = set(ownership.sensor_attachment_ids) - sensor_attachment_set
            if missing_robots:
                raise ValueError(
                    f"Resource ownership '{ownership.id}' references unknown robot attachments "
                    f"{sorted(missing_robots)} in assembly '{self.id}'."
                )
            if missing_sensors:
                raise ValueError(
                    f"Resource ownership '{ownership.id}' references unknown sensor attachments "
                    f"{sorted(missing_sensors)} in assembly '{self.id}'."
                )
            if ownership.failure_domain_id is not None and ownership.failure_domain_id not in failure_domain_set:
                raise ValueError(
                    f"Resource ownership '{ownership.id}' references unknown failure domain "
                    f"'{ownership.failure_domain_id}' in assembly '{self.id}'."
                )
        for boundary in self.safety_boundaries:
            missing_groups = set(boundary.control_group_ids) - control_group_set
            missing_robots = set(boundary.robot_attachment_ids) - robot_attachment_set
            missing_sensors = set(boundary.sensor_attachment_ids) - sensor_attachment_set
            missing_zones = set(boundary.zone_ids) - zone_set
            if missing_groups:
                raise ValueError(
                    f"Safety boundary '{boundary.id}' references unknown control groups "
                    f"{sorted(missing_groups)} in assembly '{self.id}'."
                )
            if missing_robots:
                raise ValueError(
                    f"Safety boundary '{boundary.id}' references unknown robot attachments "
                    f"{sorted(missing_robots)} in assembly '{self.id}'."
                )
            if missing_sensors:
                raise ValueError(
                    f"Safety boundary '{boundary.id}' references unknown sensor attachments "
                    f"{sorted(missing_sensors)} in assembly '{self.id}'."
                )
            if missing_zones:
                raise ValueError(
                    f"Safety boundary '{boundary.id}' references unknown safety zones "
                    f"{sorted(missing_zones)} in assembly '{self.id}'."
                )

        default_target = self.default_execution_target_id or self.execution_targets[0].id
        if default_target not in target_ids:
            raise ValueError(
                f"Default execution target '{default_target}' is not defined in assembly '{self.id}'."
            )
        object.__setattr__(self, "default_execution_target_id", default_target)

        default_control_group = self.default_control_group_id
        if self.control_groups:
            resolved_group = default_control_group or self.control_groups[0].id
            if resolved_group not in control_group_ids:
                raise ValueError(
                    f"Default control group '{resolved_group}' is not defined in assembly '{self.id}'."
                )
            object.__setattr__(self, "default_control_group_id", resolved_group)
        elif default_control_group is not None:
            raise ValueError(
                f"Assembly '{self.id}' defines default control group '{default_control_group}' "
                "without any control groups."
            )

    def execution_target(self, target_id: str | None = None) -> ExecutionTarget:
        resolved_id = target_id or self.default_execution_target_id
        for target in self.execution_targets:
            if target.id == resolved_id:
                return target
        raise KeyError(f"Unknown execution target '{resolved_id}' for assembly '{self.id}'.")

    def control_group(self, group_id: str | None = None) -> ControlGroup:
        if not self.control_groups:
            raise KeyError(f"Assembly '{self.id}' has no control groups.")

        resolved_id = group_id or self.default_control_group_id
        for group in self.control_groups:
            if group.id == resolved_id:
                return group
        raise KeyError(f"Unknown control group '{resolved_id}' for assembly '{self.id}'.")
