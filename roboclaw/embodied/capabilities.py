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


def infer_capabilities(manifest: RobotManifest) -> CapabilityProfile:
    capabilities = frozenset(manifest.capability_families)
    from_primitives = frozenset(primitive.capability_family for primitive in manifest.primitives)
    all_caps = capabilities | from_primitives
    labels = frozenset(CAPABILITY_LABELS.get(capability, capability.value) for capability in all_caps)
    return CapabilityProfile(capabilities=all_caps, labels=labels)
