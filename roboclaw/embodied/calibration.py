"""Shared calibration service — used by both CLI and Web dashboard.

CalibrationSession is a step-by-step state machine that drives the
calibration process for a single arm using LeRobot's MotorsBus API
directly (no subprocess).

States: idle → connected → homing → recording → done
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.arm.registry import ArmFamily, get_family, get_role


@dataclass
class RangeSnapshot:
    """Live min/pos/max data for each motor during range recording."""

    positions: dict[str, int]
    mins: dict[str, int]
    maxes: dict[str, int]


class CalibrationSession:
    """Drives calibration for a single arm, step by step.

    Usage (CLI)::

        session = CalibrationSession(arm_config)
        session.connect()
        # user moves arm to middle
        session.set_homing()
        # user moves joints through range
        while not done:
            snapshot = session.read_range_positions()
            display(snapshot)
        session.finish()

    Usage (Web)::

        Same methods, wrapped in asyncio.to_thread and exposed via HTTP.
    """

    def __init__(self, arm: dict[str, Any]) -> None:
        self._arm = arm
        self._family = get_family(arm["type"])
        self._role = get_role(arm["type"])
        self._bus: Any = None
        self._state = "idle"
        self._homing_offsets: dict[str, int] = {}
        self._range_motors: list[str] = []
        self._mins: dict[str, int] = {}
        self._maxes: dict[str, int] = {}

    @property
    def state(self) -> str:
        return self._state

    @property
    def family(self) -> ArmFamily:
        return self._family

    def connect(self) -> None:
        """Create motor bus, connect, disable torque, set position mode."""
        if self._state != "idle":
            raise RuntimeError(f"Cannot connect in state '{self._state}'")

        model_map = self._family.motor_models(self._role)
        Motor, MotorNormMode = self._import_motor_types()

        motors = {}
        for i, name in enumerate(self._family.motor_names):
            model = model_map.get(name, list(model_map.values())[0])
            motors[name] = Motor(id=i + 1, model=model, norm_mode=MotorNormMode.RANGE_M100_100)

        BusClass = self._import_bus_class()
        self._bus = BusClass(port=self._arm["port"], motors=motors)
        self._bus.connect()
        self._bus.disable_torque()

        # Set all motors to position mode
        OperatingMode = self._import_operating_mode()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self._state = "connected"

    def set_homing(self) -> dict[str, int]:
        """User confirmed middle position. Compute and write homing offsets.

        Returns the homing offsets dict.
        """
        if self._state != "connected":
            raise RuntimeError(f"Cannot set homing in state '{self._state}'")

        self._homing_offsets = self._bus.set_half_turn_homings()

        # Prepare range recording
        self._range_motors = [
            m for m in self._bus.motors if m not in self._family.full_turn_motors
        ]
        start_positions = self._bus.sync_read(
            "Present_Position", self._range_motors, normalize=False,
        )
        self._mins = dict(start_positions)
        self._maxes = dict(start_positions)

        self._state = "recording"
        return dict(self._homing_offsets)

    def read_range_positions(self) -> RangeSnapshot:
        """Single read of current positions, updating min/max.

        Call this in a loop (CLI) or on a polling endpoint (Web).
        """
        if self._state != "recording":
            raise RuntimeError(f"Cannot read range in state '{self._state}'")

        positions = self._bus.sync_read(
            "Present_Position", self._range_motors, normalize=False,
        )
        for motor in self._range_motors:
            val = positions[motor]
            if val < self._mins[motor]:
                self._mins[motor] = val
            if val > self._maxes[motor]:
                self._maxes[motor] = val

        return RangeSnapshot(
            positions=dict(positions),
            mins=dict(self._mins),
            maxes=dict(self._maxes),
        )

    def finish(self) -> dict[str, Any]:
        """Stop recording, build calibration, write to EEPROM + JSON file.

        Returns the calibration dict.
        """
        if self._state != "recording":
            raise RuntimeError(f"Cannot finish in state '{self._state}'")

        MotorCalibration = self._import_motor_calibration()

        # Determine motor bus resolution for full-turn motors
        max_resolution = 4095  # default for 12-bit encoders

        calibration: dict[str, Any] = {}
        for motor, m in self._bus.motors.items():
            if motor in self._family.full_turn_motors:
                range_min = 0
                range_max = max_resolution
            else:
                range_min = self._mins[motor]
                range_max = self._maxes[motor]

            if range_min == range_max:
                raise ValueError(
                    f"Motor '{motor}' has same min and max ({range_min}). "
                    "Move each joint through its full range."
                )

            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=self._homing_offsets[motor],
                range_min=range_min,
                range_max=range_max,
            )

        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        self._state = "done"
        return self._calibration_to_dict(calibration)

    def cancel(self) -> None:
        """Abort calibration and disconnect."""
        self.disconnect()
        self._state = "idle"

    def disconnect(self) -> None:
        """Clean up the motor bus connection."""
        if self._bus is not None:
            try:
                self._bus.disconnect()
            except Exception:
                pass
            self._bus = None

    # -- Private helpers ---------------------------------------------------

    def _save_calibration(self, calibration: dict) -> None:
        """Save calibration dict to JSON file."""
        cal_dir = self._arm.get("calibration_dir", "")
        if not cal_dir:
            return
        path = Path(cal_dir)
        path.mkdir(parents=True, exist_ok=True)
        serial = path.name
        file_path = path / f"{serial}.json"
        file_path.write_text(
            json.dumps(self._calibration_to_dict(calibration), indent=4)
        )

    @staticmethod
    def _calibration_to_dict(calibration: dict) -> dict[str, Any]:
        """Convert MotorCalibration dataclass instances to plain dicts."""
        result: dict[str, Any] = {}
        for name, cal in calibration.items():
            result[name] = {
                "id": cal.id,
                "drive_mode": cal.drive_mode,
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
            }
        return result

    def _import_bus_class(self) -> type:
        mod = importlib.import_module(self._family.motor_bus_module)
        return getattr(mod, self._family.motor_bus_class)

    @staticmethod
    def _import_motor_types() -> tuple:
        from lerobot.motors.motors_bus import Motor, MotorNormMode
        return Motor, MotorNormMode

    @staticmethod
    def _import_motor_calibration() -> type:
        from lerobot.motors.motors_bus import MotorCalibration
        return MotorCalibration

    @staticmethod
    def _import_operating_mode() -> type:
        from lerobot.motors.motors_bus import OperatingMode
        return OperatingMode
