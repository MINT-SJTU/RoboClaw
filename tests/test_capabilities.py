from roboclaw.embodied.capabilities import infer_capabilities
from roboclaw.embodied.definition.components.robots.model import PrimitiveSpec, quick_manifest
from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, CommandMode, PrimitiveKind, RobotType
from roboclaw.embodied.execution.orchestration.skills import SkillSpec


def _manifest():
    return quick_manifest(
        id="demo",
        name="Demo",
        robot_type=RobotType.ARM,
        capability_families=(CapabilityFamily.JOINT_MOTION,),
        primitives=(
            PrimitiveSpec("joint_move", PrimitiveKind.MOTION, CapabilityFamily.JOINT_MOTION, CommandMode.POSITION, "Move joints."),
            PrimitiveSpec("gripper_open", PrimitiveKind.END_EFFECTOR, CapabilityFamily.END_EFFECTOR, CommandMode.DISCRETE_TRIGGER, "Open gripper."),
        ),
    )


def test_infer_capabilities_produces_labels_from_manifest() -> None:
    profile = infer_capabilities(_manifest())
    assert profile.capabilities == frozenset((CapabilityFamily.JOINT_MOTION, CapabilityFamily.END_EFFECTOR))
    assert profile.labels == frozenset(("has_joints", "has_gripper"))
    assert _manifest().capability_profile().has("has_gripper") is True


def test_can_run_skill_when_capabilities_match() -> None:
    skill = SkillSpec("pick", "Pick.", (), required_capabilities=(CapabilityFamily.END_EFFECTOR,))
    assert infer_capabilities(_manifest()).can_run_skill(skill) is True


def test_can_run_skill_when_capability_is_missing() -> None:
    skill = SkillSpec("inspect", "Inspect.", (), required_capabilities=(CapabilityFamily.CAMERA,))
    assert infer_capabilities(_manifest()).can_run_skill(skill) is False
