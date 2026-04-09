"""Calibration — CLI subprocess flow + Web direct motor bus state machine.

CalibrationSession  — CLI: runs lerobot-calibrate subprocess per arm.
CalibrationEngine   — Web: direct motor bus, step-by-step state machine.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from roboclaw.embodied.board.channels import CH_CALIBRATION
from roboclaw.embodied.command import CommandBuilder, resolve_action_arms
from roboclaw.embodied.embodiment.arm.registry import get_model, get_role
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.executor import SubprocessExecutor

# Minimal spec data needed for motor bus operations.
# Replaces the deleted ServoArmSpec — only what calibration actually uses.
_MOTOR_SPECS: dict[str, dict[str, Any]] = {
    "so101": {
        "motor_bus_module": "lerobot.motors.feetech",
        "motor_bus_class": "FeetechMotorsBus",
        "motor_names": ("shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll", "gripper"),
        "full_turn_motors": ("wrist_roll",),
        "default_motor": "sts3215",
    },
    "koch": {
        "motor_bus_module": "lerobot.motors.dynamixel",
        "motor_bus_class": "DynamixelMotorsBus",
        "motor_names": ("shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll", "gripper"),
        "full_turn_motors": ("wrist_roll",),
        "default_motor": "xl330-m288",
    },
}


def _get_spec(arm_type: str) -> dict[str, Any]:
    """Look up motor spec by arm type name."""
    model = get_model(arm_type)
    if model not in _MOTOR_SPECS:
        raise ValueError(f"No motor spec for model '{model}'")
    return _MOTOR_SPECS[model]

if TYPE_CHECKING:
    from roboclaw.embodied.board.board import Board
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


# ── CalibrationSession (CLI) ─────────────────────────────────────────────


class CalibrationSession:
    """CLI calibration -- runs lerobot-calibrate subprocess for each arm.

    Iterates over uncalibrated arms, launches a PassthroughSpec subprocess
    per arm, and syncs EEPROM on success.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    async def calibrate(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        """Run calibration for each uncalibrated arm."""
        if not tty_handoff:
            return "This action requires a local terminal."

        configured = manifest.arms
        if not configured:
            return "No arms configured."

        targets = _resolve_targets(manifest, kwargs)
        if not targets:
            return "All arms are already calibrated."

        runner = SubprocessExecutor()
        results: list[str] = []

        for arm in targets:
            result = await self._calibrate_one(arm, manifest, runner, tty_handoff)
            if result == "interrupted":
                return "interrupted"
            results.append(result)

        self._parent.manifest.reload()
        ok = sum(1 for r in results if r.endswith(": OK"))
        fail = len(results) - ok
        return f"{ok} succeeded, {fail} failed.\n" + "\n".join(results)

    async def _calibrate_one(
        self,
        arm: Binding,
        manifest: Manifest,
        runner: SubprocessExecutor,
        tty_handoff: Any,
    ) -> str:
        """Calibrate a single arm. Returns result string or "interrupted"."""
        display = arm.alias
        argv = CommandBuilder.calibrate(arm)
        await tty_handoff(start=True, label=f"Calibrating: {display}")
        try:
            rc, stderr_text = await runner.run_interactive(argv)
        finally:
            await tty_handoff(start=False, label=f"Calibrating: {display}")

        if rc in (130, -2):
            return "interrupted"
        if rc == 0:
            manifest.mark_arm_calibrated(arm.alias)
            _sync_calibration_to_motors(arm)
            return f"{display}: OK"
        return _format_failure(display, rc, stderr_text)


def _format_failure(display: str, rc: int, stderr_text: str) -> str:
    """Format a calibration failure message."""
    msg = f"{display}: FAILED (exit {rc})"
    if stderr_text.strip():
        msg += f"\nstderr: {stderr_text.strip()}"
    return msg


def _resolve_targets(manifest: Manifest, kwargs: dict[str, Any]) -> list[Binding]:
    """Select calibration targets -- all uncalibrated arms, or explicit selection."""
    selected = resolve_action_arms(manifest, kwargs.get("arms", ""))
    if kwargs.get("arms", ""):
        return selected
    return [arm for arm in selected if not arm.calibrated]


def _sync_calibration_to_motors(arm: Binding) -> None:
    """Sync calibration data to motor EEPROM after successful CLI calibration."""
    cal_dir = arm.calibration_dir
    serial = Path(cal_dir).name
    cal_path = Path(cal_dir) / f"{serial}.json"
    if not cal_path.exists():
        return

    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

    spec = _get_spec(arm.type_name)
    default_motor = spec["default_motor"]
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

    mod = importlib.import_module(spec["motor_bus_module"])
    bus_class = getattr(mod, spec["motor_bus_class"])
    bus = bus_class(port=arm.port, motors=motors, calibration=calibration)
    try:
        bus.connect()
        for name, cfg in cal.items():
            bus.write("Homing_Offset", name, cfg["homing_offset"], normalize=False)
            bus.write("Min_Position_Limit", name, cfg["range_min"], normalize=False)
            bus.write("Max_Position_Limit", name, cfg["range_max"], normalize=False)
    except (OSError, ConnectionError):
        logger.warning("Motor EEPROM sync failed for {}", arm.alias)
    finally:
        bus.disconnect()


# ── CalibrationEngine (Web) ──────────────────────────────────────────────


@dataclass
class RangeSnapshot:
    """Live min/pos/max data for each motor during range recording."""

    positions: dict[str, int]
    mins: dict[str, int]
    maxes: dict[str, int]


class CalibrationEngine:
    """Step-by-step calibration for a single arm via direct motor bus.

    States: idle -> connected -> recording -> done

    Used by Web dashboard's CalibrationService for interactive
    browser-driven calibration without subprocess.

    Usage::

        engine = CalibrationEngine(arm_binding)
        engine.connect()
        # user moves arm to middle position
        engine.set_homing()
        # user moves each joint through range
        while not done:
            snapshot = engine.read_range_positions()
            display(snapshot)
        result = engine.finish()
    """

    def __init__(self, arm: Binding) -> None:
        self._arm = arm
        self._spec = _get_spec(arm.type_name)
        self._role = get_role(arm.type_name)
        self._bus: Any = None
        self._state = "idle"
        self._homing_offsets: dict[str, int] = {}
        self._range_motors: list[str] = []
        self._mins: dict[str, int] = {}
        self._maxes: dict[str, int] = {}

    @property
    def state(self) -> str:
        return self._state

    def connect(self) -> None:
        """Create motor bus, connect, disable torque."""
        if self._state != "idle":
            raise RuntimeError(f"Cannot connect in state '{self._state}'")

        default_motor = self._spec["default_motor"]
        Motor, MotorNormMode = _import_motor_types()

        motors = {}
        for i, name in enumerate(self._spec["motor_names"]):
            motors[name] = Motor(
                id=i + 1, model=default_motor, norm_mode=MotorNormMode.RANGE_M100_100,
            )

        BusClass = _import_bus_class(self._spec)
        self._bus = BusClass(port=self._arm.port, motors=motors)
        self._bus.connect()
        self._bus.disable_torque()
        self._state = "connected"

    def set_homing(self) -> dict[str, int]:
        """User confirmed middle position. Compute and write homing offsets.

        Returns the homing offsets dict.
        """
        if self._state != "connected":
            raise RuntimeError(f"Cannot set homing in state '{self._state}'")

        self._homing_offsets = self._bus.set_half_turn_homings()

        self._range_motors = [
            m for m in self._bus.motors if m not in self._spec["full_turn_motors"]
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

        Call in a loop (CLI) or on a polling endpoint (Web).
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
        """Stop recording, build calibration, write to EEPROM + JSON.

        Returns the calibration dict.
        """
        if self._state != "recording":
            raise RuntimeError(f"Cannot finish in state '{self._state}'")

        calibration = self._build_calibration()
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        self._state = "done"
        return _calibration_to_dict(calibration)

    def _build_calibration(self) -> dict[str, Any]:
        """Compute MotorCalibration for every motor."""
        MotorCalibration = _import_motor_calibration()
        max_resolution = 4095  # default for 12-bit encoders
        calibration: dict[str, Any] = {}

        for motor, m in self._bus.motors.items():
            range_min, range_max = self._motor_range(motor, max_resolution)
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
        return calibration

    def _motor_range(self, motor: str, max_resolution: int) -> tuple[int, int]:
        """Return (range_min, range_max) for a motor."""
        if motor in self._spec["full_turn_motors"]:
            return 0, max_resolution
        return self._mins[motor], self._maxes[motor]

    def cancel(self) -> None:
        """Abort calibration and disconnect."""
        self.disconnect()
        self._state = "idle"

    def disconnect(self) -> None:
        """Clean up the motor bus connection."""
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None

    # -- Private helpers ---------------------------------------------------

    def _save_calibration(self, calibration: dict) -> None:
        """Save calibration dict to JSON file."""
        cal_dir = self._arm.calibration_dir
        if not cal_dir:
            return
        path = Path(cal_dir)
        path.mkdir(parents=True, exist_ok=True)
        serial = path.name
        file_path = path / f"{serial}.json"
        file_path.write_text(
            json.dumps(_calibration_to_dict(calibration), indent=4),
        )


# ── Shared helpers ───────────────────────────────────────────────────────


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


def _import_motor_types() -> tuple:
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    return Motor, MotorNormMode


def _import_motor_calibration() -> type:
    from lerobot.motors.motors_bus import MotorCalibration

    return MotorCalibration


def _import_bus_class(spec: dict[str, Any]) -> type:
    mod = importlib.import_module(spec["motor_bus_module"])
    return getattr(mod, spec["motor_bus_class"])


# ── CalibrationService (Web coordinator) ────────────────────────────────

class CalibrationService:
    """Manages arm calibration sessions with port-lock coordination.

    Used by EmbodiedService for web dashboard calibration flow.
    """

    def __init__(self, parent: EmbodiedService, board: "Board") -> None:
        self._parent = parent
        self._board = board
        self._session: CalibrationEngine | None = None
        self._arm_alias: str = ""
        self._port_cm: Any = None

    async def start(self, arm_alias: str) -> dict[str, Any]:
        self._parent.acquire_embodiment("calibrating")
        binding = self._parent.manifest.find_binding(arm_alias)
        if binding is None:
            self._parent.release_embodiment()
            raise RuntimeError(f"Arm '{arm_alias}' not found in manifest.")
        self._port_cm = binding.guard.acquire("calibrating")
        await self._port_cm.__aenter__()

        arm = self._parent.manifest.find_arm(arm_alias)
        if arm is None:
            await self._cleanup()
            raise RuntimeError(f"Arm '{arm_alias}' not found in manifest.")
        session = CalibrationEngine(arm)
        try:
            await asyncio.to_thread(session.connect)
        except Exception:
            await self._cleanup()
            raise

        self._session = session
        self._arm_alias = arm_alias
        result = {"state": session.state, "arm_alias": arm_alias}
        await self._emit_state(session.state, arm_alias)
        return result

    def get_status(self) -> dict[str, Any]:
        if self._session is None:
            return {"state": "idle", "arm_alias": ""}
        return {"state": self._session.state, "arm_alias": self._arm_alias}

    async def set_homing(self) -> dict[str, Any]:
        self._require_session()
        offsets = await asyncio.to_thread(self._session.set_homing)
        result = {"state": self._session.state, "homing_offsets": offsets}
        await self._emit_state(self._session.state, self._arm_alias)
        return result

    async def read_positions(self) -> dict[str, Any]:
        self._require_session()
        if self._session.state != "recording":
            raise RuntimeError(f"Not recording (state={self._session.state})")
        snapshot = await asyncio.to_thread(self._session.read_range_positions)
        return {
            "positions": snapshot.positions,
            "mins": snapshot.mins,
            "maxes": snapshot.maxes,
        }

    async def finish(self) -> dict[str, Any]:
        self._require_session()
        calibration = await asyncio.to_thread(self._session.finish)
        self._parent.manifest.mark_arm_calibrated(self._arm_alias)
        arm_alias = self._arm_alias
        await self._cleanup()
        await self._emit_state("done", arm_alias)
        return {"state": "done", "calibration": calibration}

    async def cancel(self) -> None:
        arm_alias = self._arm_alias
        if self._session is not None:
            await asyncio.to_thread(self._session.cancel)
        await self._cleanup()
        await self._emit_state("idle", arm_alias)

    @property
    def active(self) -> bool:
        return self._session is not None

    async def _emit_state(self, state: str, arm_alias: str) -> None:
        await self._board.emit(CH_CALIBRATION, {
            "state": state,
            "arm_alias": arm_alias,
            "timestamp": _time.time(),
        })

    def _require_session(self) -> None:
        if self._session is None:
            raise RuntimeError("No calibration session active.")

    async def _cleanup(self) -> None:
        if self._port_cm is not None:
            await self._port_cm.__aexit__(None, None, None)
            self._port_cm = None
        if self._session is not None:
            await asyncio.to_thread(self._session.disconnect)
            self._session = None
        self._arm_alias = ""
        self._parent.release_embodiment()
