"""Motor bus utilities for reading servo positions per arm model."""

from __future__ import annotations

import importlib
from typing import Any

from roboclaw.embodied.embodiment.arm.registry import ArmRuntimeSpec, get_runtime_spec
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.embodiment.manifest.helpers import load_calibration


def _active_arms_with_specs(arms: list[Binding]) -> list[tuple[Binding, ArmRuntimeSpec]]:
    active_arms = [arm for arm in arms if arm.port]
    return [(arm, get_runtime_spec(arm.arm_type)) for arm in active_arms]


def _import_motor_types() -> tuple[Any, Any]:
    module = importlib.import_module("lerobot.motors.motors_bus")
    return module.Motor, module.MotorNormMode


def _import_bus_class(spec: ArmRuntimeSpec) -> Any:
    module = importlib.import_module(spec.motor_bus_module)
    return getattr(module, spec.motor_bus_class)


def _motor_config_from_arm(
    arm: Binding, spec: ArmRuntimeSpec,
) -> dict[str, tuple[int, str]]:
    """Build motor config from calibration first, model defaults second."""
    cal = load_calibration(arm)
    if cal:
        return {
            name: (cfg["id"], spec.default_motor)
            for name, cfg in cal.items()
            if isinstance(cfg, dict) and "id" in cfg
        }
    return {
        name: (index + 1, spec.default_motor)
        for index, name in enumerate(spec.default_joint_names)
    }


def _read_arm(
    arm: Binding,
    spec: ArmRuntimeSpec,
    motor_type: Any,
    motor_norm_mode: Any,
) -> dict[str, Any]:
    motor_config = _motor_config_from_arm(arm, spec)
    motors = {
        name: motor_type(id=motor_id, model=model, norm_mode=motor_norm_mode.RANGE_M100_100)
        for name, (motor_id, model) in motor_config.items()
    }
    bus_class = _import_bus_class(spec)
    bus = bus_class(port=arm.port, motors=motors)
    try:
        bus.connect()
        positions = {
            name: int(bus.read("Present_Position", name, normalize=False))
            for name in motor_config
        }
        temperatures = {
            name: (
                int(bus.read("Present_Temperature", name, normalize=False))
                if spec.supports_temperature else None
            )
            for name in motor_config
        }
        return {"positions": positions, "temperatures": temperatures}
    finally:
        bus.disconnect()


def read_servo_positions(arms: list[Binding]) -> dict[str, Any]:
    """Read current servo positions for all configured arms."""
    result: dict[str, Any] = {"error": None, "arms": {}}
    active_arms = _active_arms_with_specs(arms)
    if not active_arms:
        return result

    motor_type, motor_norm_mode = _import_motor_types()
    for arm, spec in active_arms:
        result["arms"][arm.alias] = _read_arm(arm, spec, motor_type, motor_norm_mode)
    return result
