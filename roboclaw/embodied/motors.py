"""Motor bus utilities for reading servo positions.

Provides a hardware-aware interface for reading motor positions from
configured arms, using calibration files for motor discovery with
SO101 defaults as fallback.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from roboclaw.embodied.setup import load_calibration, load_setup

SO101_MOTOR_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
SO101_MOTOR_MODEL = "sts3215"


def _motor_config_from_calibration(arm: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """Read calibration file to discover motor names and IDs.

    Returns {name: (id, model)}. Falls back to SO101 defaults if no
    calibration file exists.
    """
    cal = load_calibration(arm)
    if cal:
        return {
            name: (cfg["id"], SO101_MOTOR_MODEL)
            for name, cfg in cal.items()
            if isinstance(cfg, dict) and "id" in cfg
        }
    return {
        name: (i + 1, SO101_MOTOR_MODEL)
        for i, name in enumerate(SO101_MOTOR_NAMES)
    }


def read_servo_positions(setup: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read current servo positions for all arms (followers + leaders).

    Returns ``{"error": None, "arms": {alias: {motor_name: position}}}``.
    """
    if setup is None:
        setup = load_setup()
    arms = setup.get("arms", [])
    result: dict[str, Any] = {"error": None, "arms": {}}

    active_arms = [a for a in arms if a.get("port")]
    if not active_arms:
        return result

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    for arm in active_arms:
        alias = arm.get("alias", "")
        port = arm["port"]
        motor_config = _motor_config_from_calibration(arm)
        motors = {
            name: Motor(id=mid, model=model, norm_mode=MotorNormMode.RANGE_M100_100)
            for name, (mid, model) in motor_config.items()
        }
        bus = FeetechMotorsBus(port=port, motors=motors)
        try:
            bus.connect()
            positions: dict[str, int | None] = {}
            for name in motor_config:
                try:
                    positions[name] = int(bus.read("Present_Position", name, normalize=False))
                except Exception:
                    logger.debug("Failed to read motor '{}' on arm '{}'", name, alias)
                    positions[name] = None
            result["arms"][alias] = positions
        finally:
            bus.disconnect()
    return result
