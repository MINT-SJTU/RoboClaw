"""Robot manifests and primitive contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.definition.foundation.schema import (
    ActionSchema,
    CapabilityFamily,
    CommandMode,
    CompletionSpec,
    HealthSchema,
    ObservationSchema,
    ParameterSpec,
    PrimitiveKind,
    RobotType,
    SafetyProfile,
    ToleranceSpec,
)


@dataclass(frozen=True)
class PrimitiveSpec:
    """One reusable primitive exposed by a robot."""

    name: str
    kind: PrimitiveKind
    capability_family: CapabilityFamily
    command_mode: CommandMode
    description: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    action_schema: ActionSchema | None = None
    tolerance: ToleranceSpec | None = None
    completion: CompletionSpec | None = None
    backed_by: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Primitive name cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Primitive '{self.name}' description cannot be empty.")
        if self.action_schema is not None:
            if self.action_schema.command_mode != self.command_mode:
                raise ValueError(
                    f"Primitive '{self.name}' command_mode '{self.command_mode.value}' does not "
                    f"match action_schema.command_mode '{self.action_schema.command_mode.value}'."
                )
            parameter_names = {param.name for param in self.parameters}
            unknown_names = set(self.action_schema.parameter_order) - parameter_names
            if unknown_names:
                names = ", ".join(sorted(unknown_names))
                raise ValueError(
                    f"Primitive '{self.name}' action_schema references unknown parameters: {names}."
                )


@dataclass(frozen=True)
class RobotManifest:
    """Static robot definition independent from sensors and carriers."""

    id: str
    name: str
    description: str
    robot_type: RobotType
    capability_families: tuple[CapabilityFamily, ...]
    primitives: tuple[PrimitiveSpec, ...]
    observation_schema: ObservationSchema
    health_schema: HealthSchema
    default_named_poses: tuple[str, ...] = field(default_factory=tuple)
    suggested_sensor_ids: tuple[str, ...] = field(default_factory=tuple)
    safety: SafetyProfile = field(default_factory=SafetyProfile)
    setup_hints: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        primitive_names = [primitive.name for primitive in self.primitives]
        if len(set(primitive_names)) != len(primitive_names):
            raise ValueError(f"Duplicate primitive names in robot '{self.id}'.")
        if not self.observation_schema.fields:
            raise ValueError(f"Robot '{self.id}' must define at least one observation field.")
        if not self.health_schema.fields:
            raise ValueError(f"Robot '{self.id}' must define at least one health field.")

    def supports(self, capability: CapabilityFamily) -> bool:
        return capability in self.capability_families

    def primitive(self, name: str) -> PrimitiveSpec | None:
        return next((primitive for primitive in self.primitives if primitive.name == name), None)
