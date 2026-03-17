"""Assembly exports."""

from roboclaw.embodied.definition.systems.assemblies.blueprint import AssemblyBlueprint, compose_assemblies
from roboclaw.embodied.definition.systems.assemblies.model import (
    AssemblyManifest,
    ControlGroup,
    FailureDomain,
    FrameTransform,
    ResourceLockScope,
    ResourceOwnership,
    RobotAttachment,
    SafetyBoundary,
    SafetyZone,
    SensorAttachment,
    ToolAttachment,
    Transform3D,
)
from roboclaw.embodied.definition.systems.assemblies.registry import AssemblyRegistry

__all__ = [
    "AssemblyBlueprint",
    "AssemblyManifest",
    "AssemblyRegistry",
    "ControlGroup",
    "FailureDomain",
    "FrameTransform",
    "ResourceLockScope",
    "ResourceOwnership",
    "RobotAttachment",
    "SafetyBoundary",
    "SafetyZone",
    "SensorAttachment",
    "ToolAttachment",
    "Transform3D",
    "compose_assemblies",
]
