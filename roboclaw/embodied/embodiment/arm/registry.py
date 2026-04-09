"""Arm type registry — thin index pointing at lerobot robot configs.

Motor specs, calibration, bus classes are all in lerobot.
We only keep: type name enumeration + hardware probe config.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmProbeConfig:
    """Hardware discovery probe parameters — lerobot doesn't provide these."""

    protocol: str  # "feetech" | "dynamixel"
    motor_ids: tuple[int, ...]
    baudrate: int


_PROBES: dict[str, ArmProbeConfig] = {
    "so101": ArmProbeConfig("feetech", (1, 2, 3, 4, 5, 6), 1_000_000),
    "koch": ArmProbeConfig("dynamixel", (1, 2, 3, 4, 5, 6), 1_000_000),
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
    return arm_type.rsplit("_", 1)[0]


def get_probe_config(model: str) -> ArmProbeConfig:
    """Look up probe config by model name (e.g., 'so101', 'koch')."""
    model = model.lower()
    if model not in _PROBES:
        raise ValueError(f"Unknown arm model: {model}")
    return _PROBES[model]
