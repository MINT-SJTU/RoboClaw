"""Arm spec registry — single source of truth for supported arm types."""

from __future__ import annotations

from roboclaw.embodied.embodiment.arm.base import ServoArmSpec

# ---------------------------------------------------------------------------
# SO101 (Feetech STS3215)
# ---------------------------------------------------------------------------

_SO101_MOTORS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
_SO101_MODEL = "sts3215"

SO101 = ServoArmSpec(
    name="so101",
    motor_names=_SO101_MOTORS,
    supports_bimanual=True,
    bimanual_follower_type="bi_so_follower",
    bimanual_leader_type="bi_so_leader",
    motor_bus_module="lerobot.motors.feetech",
    motor_bus_class="FeetechMotorsBus",
    follower_motor_models={m: _SO101_MODEL for m in _SO101_MOTORS},
    leader_motor_models={m: _SO101_MODEL for m in _SO101_MOTORS},
    full_turn_motors=("wrist_roll",),
    probe_protocol="feetech",
)

# ---------------------------------------------------------------------------
# Koch v1.1 (Dynamixel XL430 / XL330)
# ---------------------------------------------------------------------------

_KOCH_MOTORS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)

KOCH = ServoArmSpec(
    name="koch",
    motor_names=_KOCH_MOTORS,
    supports_bimanual=False,
    motor_bus_module="lerobot.motors.dynamixel",
    motor_bus_class="DynamixelMotorsBus",
    follower_motor_models={
        "shoulder_pan": "xl430-w250",
        "shoulder_lift": "xl430-w250",
        "elbow_flex": "xl330-m288",
        "wrist_flex": "xl330-m288",
        "wrist_roll": "xl330-m288",
        "gripper": "xl330-m288",
    },
    leader_motor_models={m: "xl330-m077" for m in _KOCH_MOTORS},
    full_turn_motors=("wrist_roll",),
    probe_protocol="dynamixel",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ServoArmSpec] = {
    "so101": SO101,
    "koch": KOCH,
}


def get_arm_spec(arm_type: str) -> ServoArmSpec:
    """Extract spec from arm type string.

    >>> get_arm_spec("so101_follower").name
    'so101'
    >>> get_arm_spec("koch_leader").name
    'koch'
    """
    name = arm_type.rsplit("_", 1)[0]
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown arm type '{name}' from '{arm_type}'")
    return spec


def get_role(arm_type: str) -> str:
    """Extract role from arm type string.

    >>> get_role("so101_follower")
    'follower'
    """
    return arm_type.rsplit("_", 1)[1]


def get_arm_spec_by_name(name: str) -> ServoArmSpec:
    """Look up arm spec by model name (e.g., 'so101', 'koch')."""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown arm model: {name}")
    return _REGISTRY[name]


def all_arm_types() -> tuple[str, ...]:
    """Return all registered arm types (follower + leader for each spec)."""
    result: list[str] = []
    for spec in _REGISTRY.values():
        result.extend(spec.arm_types)
    return tuple(result)


def all_arm_specs() -> dict[str, ServoArmSpec]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)
