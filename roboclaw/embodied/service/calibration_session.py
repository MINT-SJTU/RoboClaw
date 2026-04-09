"""CalibrationSession — CLI calibration entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from roboclaw.embodied.engine.command_builder import builder_for_arms
from roboclaw.embodied.engine.helpers import (
    _NO_TTY_MSG,
    _arm_id,
    _format_tty_failure,
    _is_interrupted,
    _resolve_action_arms,
    _run_tty,
)
from roboclaw.embodied.manifest.binding import Binding

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


_MOTOR_BUS_INFO: dict[str, tuple[str, str, str]] = {
    "so101": ("lerobot.motors.feetech", "FeetechMotorsBus", "sts3215"),
    "koch": ("lerobot.motors.dynamixel", "DynamixelMotorsBus", "xl330-m288"),
}


class CalibrationSession:
    """Wraps the CLI interactive calibration flow."""

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    async def calibrate(
        self, manifest: Manifest, kwargs: dict[str, Any], tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.manifest.helpers import arm_display_name
        from roboclaw.embodied.runner import LocalLeRobotRunner

        configured = manifest.arms
        if not configured:
            return "No arms configured."
        selected = _resolve_action_arms(manifest, kwargs)
        targets = selected if kwargs.get("arms", "") else [arm for arm in selected if not arm.calibrated]
        if not targets:
            return "All arms are already calibrated."
        if not tty_handoff:
            return _NO_TTY_MSG

        controller = builder_for_arms(targets)
        runner = LocalLeRobotRunner()
        succeeded = 0
        failed = 0
        results: list[str] = []
        for arm in targets:
            display = arm_display_name(arm)
            argv = controller.calibrate(
                arm.type_name,
                arm.port,
                arm.calibration_dir,
                _arm_id(arm),
            )
            rc, stderr_text = await _run_tty(tty_handoff, runner, argv, f"Calibrating: {display}")
            if _is_interrupted(rc):
                return "interrupted"
            if rc == 0:
                succeeded += 1
                manifest.mark_arm_calibrated(arm.alias)
                self._sync_calibration_to_motors(arm)
                results.append(f"{display}: OK")
                continue
            failed += 1
            results.append(_format_tty_failure(f"{display}: FAILED", rc, stderr_text))

        self._parent.manifest.reload()
        return (
            f"{succeeded} succeeded, {failed} failed.\n"
            + "\n".join(results)
            + "\nNote: wrist_roll is auto-calibrated by LeRobot (expected)."
        )

    def _sync_calibration_to_motors(self, arm: Binding) -> None:
        cal_dir = arm.calibration_dir
        serial = Path(cal_dir).name
        cal_path = Path(cal_dir) / f"{serial}.json"
        if not cal_path.exists():
            return
        try:
            from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
        except ImportError:
            logger.debug("lerobot.motors not installed, skipping EEPROM sync")
            return

        from roboclaw.embodied.embodiment.arm.registry import get_model

        model_name = get_model(arm.type_name)
        bus_module, bus_cls_name, default_motor = _MOTOR_BUS_INFO[model_name]
        cal = json.loads(cal_path.read_text())

        motors = {}
        calibration = {}
        for name, cfg in cal.items():
            motors[name] = Motor(id=cfg["id"], model=default_motor, norm_mode=MotorNormMode.DEGREES)
            calibration[name] = MotorCalibration(
                id=cfg["id"],
                drive_mode=cfg["drive_mode"],
                homing_offset=cfg["homing_offset"],
                range_min=cfg["range_min"],
                range_max=cfg["range_max"],
            )

        import importlib

        mod = importlib.import_module(bus_module)
        bus_class = getattr(mod, bus_cls_name)
        bus = bus_class(port=arm.port, motors=motors, calibration=calibration)
        try:
            bus.connect()
            for name, cfg in cal.items():
                bus.write("Homing_Offset", name, cfg["homing_offset"], normalize=False)
                bus.write("Min_Position_Limit", name, cfg["range_min"], normalize=False)
                bus.write("Max_Position_Limit", name, cfg["range_max"], normalize=False)
        except (OSError, ConnectionError):
            logger.debug("Motor EEPROM sync failed for %s", arm.alias)
        finally:
            bus.disconnect()
