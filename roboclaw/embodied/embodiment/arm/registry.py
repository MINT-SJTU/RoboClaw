"""Arm type registry for discovery and runtime motor access."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmProbeConfig:
    """Hardware discovery probe parameters — lerobot doesn't provide these."""

    protocol: str  # "feetech" | "dynamixel"
    motor_ids: tuple[int, ...]
    baudrate: int


@dataclass(frozen=True)
class ArmRuntimeSpec:
    """Runtime motor access configuration for an arm model."""

    motor_bus_module: str
    motor_bus_class: str
    default_motor: str
    default_joint_names: tuple[str, ...]
    supports_temperature: bool


@dataclass(frozen=True)
class ArmModelSpec:
    """Full spec for an arm model."""

    probe: ArmProbeConfig
    runtime: ArmRuntimeSpec


_DEFAULT_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_MODELS: dict[str, ArmModelSpec] = {
    "so101": ArmModelSpec(
        probe=ArmProbeConfig("feetech", (1, 2, 3, 4, 5, 6), 1_000_000),
        runtime=ArmRuntimeSpec(
            motor_bus_module="lerobot.motors.feetech",
            motor_bus_class="FeetechMotorsBus",
            default_motor="sts3215",
            default_joint_names=_DEFAULT_ARM_JOINT_NAMES,
            supports_temperature=True,
        ),
    ),
    "koch": ArmModelSpec(
        probe=ArmProbeConfig("dynamixel", (1, 2, 3, 4, 5, 6), 1_000_000),
        runtime=ArmRuntimeSpec(
            motor_bus_module="lerobot.motors.dynamixel",
            motor_bus_class="DynamixelMotorsBus",
            default_motor="xl330-m288",
            default_joint_names=_DEFAULT_ARM_JOINT_NAMES,
            supports_temperature=True,
        ),
    ),
}

_ALL_TYPES = (
    "so101_follower",
    "so101_leader",
    "koch_follower",
    "koch_leader",
)


def all_arm_types() -> tuple[str, ...]:
    """Return all registered arm types."""
    return _ALL_TYPES


def get_role(arm_type: str) -> str:
    """Extract role from arm type string.

    >>> get_role("so101_follower")
    'follower'
    """
    return arm_type.rsplit("_", 1)[1]


def get_model(arm_type: str) -> str:
    """Extract model name from arm type string.

    >>> get_model("so101_follower")
    'so101'
    """
    return arm_type.rsplit("_", 1)[0].lower()


def _get_model_spec(model_or_type: str) -> ArmModelSpec:
    model = get_model(model_or_type)
    if model not in _MODELS:
        raise ValueError(f"Unknown arm model: {model}")
    return _MODELS[model]


def get_probe_config(model: str) -> ArmProbeConfig:
    """Look up discovery probe config by arm model or arm type."""
    return _get_model_spec(model).probe


def get_runtime_spec(model_or_type: str) -> ArmRuntimeSpec:
    """Look up runtime motor config by arm model or arm type."""
    return _get_model_spec(model_or_type).runtime
