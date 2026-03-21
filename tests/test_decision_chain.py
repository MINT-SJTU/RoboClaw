from roboclaw.embodied.capabilities import diagnose_gap, infer_capabilities
from roboclaw.embodied.definition.components.robots.model import PrimitiveSpec, quick_manifest
from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, CommandMode, PrimitiveKind, RobotType
from roboclaw.embodied.execution.orchestration.skills import SkillSpec


def _profile():
    return infer_capabilities(
        quick_manifest(
            id="demo",
            name="Demo",
            robot_type=RobotType.ARM,
            capability_families=(CapabilityFamily.JOINT_MOTION,),
            primitives=(PrimitiveSpec("joint_move", PrimitiveKind.MOTION, CapabilityFamily.JOINT_MOTION, CommandMode.POSITION, "Move joints."),),
        )
    )


def test_diagnose_gap_returns_none_when_capabilities_are_met() -> None:
    assert diagnose_gap(_profile(), requested_capabilities=(CapabilityFamily.JOINT_MOTION,)) is None


def test_diagnose_gap_returns_missing_labels() -> None:
    gap = diagnose_gap(_profile(), skill=SkillSpec("inspect", "Inspect.", (), required_capabilities=(CapabilityFamily.CAMERA, CapabilityFamily.END_EFFECTOR)))
    assert gap is not None
    assert gap.missing_capabilities == ("has_camera", "has_gripper")
    assert gap.has_hardware is False


def test_diagnose_gap_marks_policy_like_gaps_as_hardware_ready() -> None:
    gap = diagnose_gap(_profile(), skill=SkillSpec("reset_arm", "Reset.", (), required_capabilities=(CapabilityFamily.NAMED_POSE,)))
    assert gap is not None
    assert gap.has_hardware is True
    assert gap.suggestion == "Train a policy for this task"
