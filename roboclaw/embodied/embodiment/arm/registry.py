"""Arm family registry — single source of truth for supported arm types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArmFamily:
    """Describes one family of robot arms (e.g. SO101, Koch)."""

    name: str
    motor_bus_module: str
    motor_bus_class: str
    follower_motor_models: dict[str, str]
    leader_motor_models: dict[str, str]
    motor_names: tuple[str, ...]
    full_turn_motors: tuple[str, ...] = ()
    supports_bimanual: bool = False
    bimanual_follower_type: str = ""
    bimanual_leader_type: str = ""

    # Low-level probe parameters (for identify.py)
    probe_protocol: str = "feetech"  # "feetech" or "dynamixel"

    def motor_models(self, role: str) -> dict[str, str]:
        """Return motor models for 'follower' or 'leader'."""
        if role == "leader":
            return dict(self.leader_motor_models)
        return dict(self.follower_motor_models)

    @property
    def follower_type(self) -> str:
        return f"{self.name}_follower"

    @property
    def leader_type(self) -> str:
        return f"{self.name}_leader"

    @property
    def arm_types(self) -> tuple[str, str]:
        return (self.follower_type, self.leader_type)


# ---------------------------------------------------------------------------
# SO101 (Feetech STS3215)
# ---------------------------------------------------------------------------

_SO101_MOTORS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
_SO101_MODEL = "sts3215"

SO101 = ArmFamily(
    name="so101",
    motor_bus_module="lerobot.motors.feetech",
    motor_bus_class="FeetechMotorsBus",
    follower_motor_models={m: _SO101_MODEL for m in _SO101_MOTORS},
    leader_motor_models={m: _SO101_MODEL for m in _SO101_MOTORS},
    motor_names=_SO101_MOTORS,
    full_turn_motors=("wrist_roll",),
    supports_bimanual=True,
    bimanual_follower_type="bi_so_follower",
    bimanual_leader_type="bi_so_leader",
    probe_protocol="feetech",
)

# ---------------------------------------------------------------------------
# Koch v1.1 (Dynamixel XL430 / XL330)
# ---------------------------------------------------------------------------

_KOCH_MOTORS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)

KOCH = ArmFamily(
    name="koch",
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
    motor_names=_KOCH_MOTORS,
    full_turn_motors=("wrist_roll",),
    supports_bimanual=False,
    probe_protocol="dynamixel",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ArmFamily] = {
    "so101": SO101,
    "koch": KOCH,
}


def get_family(arm_type: str) -> ArmFamily:
    """Extract family from arm type string.

    >>> get_family("so101_follower").name
    'so101'
    >>> get_family("koch_leader").name
    'koch'
    """
    name = arm_type.rsplit("_", 1)[0]
    family = _REGISTRY.get(name)
    if family is None:
        raise ValueError(f"Unknown arm family '{name}' from type '{arm_type}'")
    return family


def get_role(arm_type: str) -> str:
    """Extract role from arm type string.

    >>> get_role("so101_follower")
    'follower'
    """
    return arm_type.rsplit("_", 1)[1]


def all_arm_types() -> tuple[str, ...]:
    """Return all registered arm types (follower + leader for each family)."""
    result: list[str] = []
    for family in _REGISTRY.values():
        result.extend(family.arm_types)
    return tuple(result)


def all_families() -> dict[str, ArmFamily]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)
