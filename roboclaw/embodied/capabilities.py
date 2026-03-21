"""Layer 2 capability inference from robot manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily

if TYPE_CHECKING:
    from roboclaw.embodied.definition.components.robots.model import RobotManifest
    from roboclaw.embodied.execution.orchestration.skills import SkillSpec


CAPABILITY_LABELS = {
    CapabilityFamily.JOINT_MOTION: "has_joints",
    CapabilityFamily.END_EFFECTOR: "has_gripper",
    CapabilityFamily.BASE_MOTION: "has_base",
    CapabilityFamily.HEAD_MOTION: "has_head",
    CapabilityFamily.CARTESIAN_MOTION: "has_cartesian",
    CapabilityFamily.CAMERA: "has_camera",
    CapabilityFamily.CALIBRATION: "has_calibration",
    CapabilityFamily.NAMED_POSE: "has_named_poses",
    CapabilityFamily.TORQUE_CONTROL: "has_torque_control",
}
_HARDWARE_CAPABILITIES = frozenset(
    {
        CapabilityFamily.JOINT_MOTION,
        CapabilityFamily.CARTESIAN_MOTION,
        CapabilityFamily.BASE_MOTION,
        CapabilityFamily.HEAD_MOTION,
        CapabilityFamily.END_EFFECTOR,
        CapabilityFamily.CAMERA,
    }
)


@dataclass(frozen=True)
class CapabilityProfile:
    capabilities: frozenset[CapabilityFamily]
    labels: frozenset[str]

    def has(self, label: str) -> bool:
        return label in self.labels

    def supports(self, capability: CapabilityFamily) -> bool:
        return capability in self.capabilities

    def can_run_skill(self, skill: SkillSpec) -> bool:
        return all(cap in self.capabilities for cap in skill.required_capabilities)


@dataclass(frozen=True)
class CapabilityGap:
    requested_action: str
    missing_capabilities: tuple[str, ...]
    has_hardware: bool
    suggestion: str


def infer_capabilities(manifest: RobotManifest) -> CapabilityProfile:
    capabilities = frozenset(manifest.capability_families)
    from_primitives = frozenset(primitive.capability_family for primitive in manifest.primitives)
    all_caps = capabilities | from_primitives
    labels = frozenset(CAPABILITY_LABELS.get(capability, capability.value) for capability in all_caps)
    return CapabilityProfile(capabilities=all_caps, labels=labels)


def resolve_available_skills(
    profile: CapabilityProfile,
    all_skills: tuple[SkillSpec, ...],
) -> tuple[SkillSpec, ...]:
    """Return skills whose required_capabilities are satisfied by this profile."""
    return tuple(skill for skill in all_skills if profile.can_run_skill(skill))


def diagnose_gap(
    profile: CapabilityProfile,
    skill: SkillSpec | None = None,
    requested_capabilities: tuple[CapabilityFamily, ...] = (),
) -> CapabilityGap | None:
    """Return None when all capabilities are met, or a small diagnosis for what is missing."""
    required = set(skill.required_capabilities if skill else requested_capabilities)
    missing = tuple(
        sorted(required - profile.capabilities, key=lambda capability: CAPABILITY_LABELS.get(capability, capability.value))
    )
    if not missing:
        return None
    missing_labels = tuple(CAPABILITY_LABELS.get(capability, capability.value) for capability in missing)
    has_hardware = not any(capability in _HARDWARE_CAPABILITIES for capability in missing)
    return CapabilityGap(
        requested_action=skill.name if skill else "unknown",
        missing_capabilities=missing_labels,
        has_hardware=has_hardware,
        suggestion="Train a policy for this task" if has_hardware else "This robot does not have the required hardware",
    )


diagnose_capability_gap = diagnose_gap
