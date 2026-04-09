"""Calibration Session — runs lerobot-calibrate subprocess per arm.

Same architecture as TeleopSession / RecordSession:
  Session → subprocess → OutputConsumer → Board → WebSocket → Frontend
  Frontend → Board.post_command() → InputConsumer → subprocess stdin
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from roboclaw.embodied.board.constants import Command, SessionState
from roboclaw.embodied.board.consumer import InputConsumer, OutputConsumer
from roboclaw.embodied.command import CommandBuilder, resolve_action_arms
from roboclaw.embodied.embodiment.arm.registry import get_model
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService

# Minimal spec data needed for EEPROM sync after calibration.
_MOTOR_SPECS: dict[str, dict[str, Any]] = {
    "so101": {
        "motor_bus_module": "lerobot.motors.feetech",
        "motor_bus_class": "FeetechMotorsBus",
        "default_motor": "sts3215",
    },
    "koch": {
        "motor_bus_module": "lerobot.motors.dynamixel",
        "motor_bus_class": "DynamixelMotorsBus",
        "default_motor": "xl330-m288",
    },
}

_RE_POSITION_ROW = re.compile(
    r"(\w+)\s+\|\s+(-?\d+)\s+\|\s+(-?\d+)\s+\|\s+(-?\d+)"
)


def _get_spec(arm_type: str) -> dict[str, Any]:
    model = get_model(arm_type)
    if model not in _MOTOR_SPECS:
        raise ValueError(f"No motor spec for model '{model}'")
    return _MOTOR_SPECS[model]


def _has_existing_calibration(arm: Binding) -> bool:
    cal_dir = Path(arm.calibration_dir).expanduser()
    serial = cal_dir.name
    return bool(serial) and (cal_dir / f"{serial}.json").exists()


def _print_existing_calibration_tip() -> None:
    print(
        "Calibration tip: if LeRobot asks "
        "'Press ENTER to use provided calibration file...', "
        "then ENTER reuses the existing calibration and "
        "c + ENTER starts a fresh calibration."
    )


# ── Consumers ────────────────────────────────────────────────────────────


class CalibrationOutputConsumer(OutputConsumer):
    """Parse lerobot-calibrate subprocess stdout → Board state updates.

    Reads chunks instead of lines because input() prompts don't end
    with newlines, which would block the line-based StreamReader.
    """

    def __init__(self, board: Any, stdout: Any) -> None:
        super().__init__(board, stdout)
        self._positions: dict[str, dict[str, int]] = {}
        self._buffer = ""

    async def _run(self) -> None:
        """Read stdout in chunks, split into lines + partial prompts."""
        try:
            while True:
                chunk = await self._stdout.read(4096)
                if not chunk:
                    break
                self._buffer += chunk.decode("utf-8", errors="replace")
                await self._process_buffer()
        except (OSError, ConnectionError):
            pass

    async def _process_buffer(self) -> None:
        """Process complete lines and detect input() prompts (no trailing newline)."""
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.board.log(line)
                await self._parse(line)
        # Check if remaining buffer contains an input() prompt (no newline)
        if self._buffer and (":" in self._buffer or "..." in self._buffer):
            await self._parse(self._buffer)

    async def _parse(self, text: str) -> None:
        m = _RE_POSITION_ROW.search(text)
        if m:
            name = m.group(1)
            new_val = {"min": int(m.group(2)), "pos": int(m.group(3)), "max": int(m.group(4))}
            if self._positions.get(name) != new_val:
                self._positions[name] = new_val
                await self.board.update(calibration_positions=dict(self._positions))
            return

        low = text.lower()
        if "press enter to use provided calibration" in low:
            await self.board.update(state=SessionState.CALIBRATING, calibration_step="choose")
        elif "running calibration" in low:
            await self.board.update(calibration_step="starting")
        elif "move" in low and "middle" in low and "press enter" in low:
            await self.board.update(calibration_step="homing")
        elif "recording positions" in low and "press enter to stop" in low:
            await self.board.update(calibration_step="recording")
        elif "calibration saved" in low:
            await self.board.update(calibration_step="done")


class CalibrationInputConsumer(InputConsumer):
    """Extend keymap with RECALIBRATE command."""

    def translate(self, command: str) -> bytes | None:
        if command == Command.RECALIBRATE:
            return b"c\n"
        return super().translate(command)


# ── Session ──────────────────────────────────────────────────────────────


class CalibrationSession(Session):
    """Calibration as a proper Session subclass.

    Lifecycle: start_calibration(arm) → subprocess runs → OutputConsumer
    parses prompts → Board state updates → frontend/agent reacts →
    InputConsumer sends user responses → subprocess completes.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent
        self._arm: Binding | None = None
        self._cal_manifest: Manifest | None = None

    def _make_output_consumer(self, board: Any, stdout: Any) -> OutputConsumer:
        return CalibrationOutputConsumer(board, stdout)

    def _make_input_consumer(self, board: Any, stdin: Any) -> InputConsumer:
        return CalibrationInputConsumer(board, stdin)

    async def start_calibration(self, arm: Binding, manifest: Manifest) -> None:
        """Start calibration subprocess for a single arm."""
        self._arm = arm
        self._cal_manifest = manifest
        await self.board.update(calibration_arm=arm.alias)
        argv = CommandBuilder.calibrate(arm)
        await self.start(argv, initial_state=SessionState.CALIBRATING, auto_confirm=False)

    async def _wait_process(self) -> None:
        """Override to sync EEPROM and release embodiment lock on exit."""
        assert self._process is not None
        rc = self._process.returncode
        if rc is None:
            rc = await self._process.wait()
        await super()._wait_process()

        if rc == 0 and self._arm and self._cal_manifest:
            self._cal_manifest.mark_arm_calibrated(self._arm.alias)
            _sync_calibration_to_motors(self._arm)

        self._parent.release_embodiment()

    # -- CLI protocol (used by TtySession when invoked from agent) ---------

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec
        return PollingSpec(label="lerobot-calibrate")

    def status_line(self) -> str:
        state = self.board.state
        step = state.get("calibration_step", "")
        alias = state.get("calibration_arm", "")
        if step == "choose":
            return f"Calibrating {alias}: waiting for choice..."
        if step == "homing":
            return f"Calibrating {alias}: move to middle position"
        if step == "recording":
            return f"Calibrating {alias}: recording range of motion"
        if step == "done":
            return f"Calibrating {alias}: done"
        return f"Calibrating {alias}: {step or 'preparing'}"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()

    def result(self) -> str:
        state = self.board.state
        step = state.get("calibration_step", "")
        alias = state.get("calibration_arm", "")
        if step == "done":
            return f"Calibration of {alias} completed successfully."
        if state.get("state") == SessionState.ERROR:
            return f"Calibration of {alias} failed: {state.get('error', 'unknown error')}"
        return f"Calibration of {alias} ended."

    async def stop(self) -> None:
        if not self._parent.embodiment_busy:
            return
        await super().stop()

    # -- Agent entry point -------------------------------------------------

    async def calibrate(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        """Agent/CLI entry: calibrate arms via TtySession."""
        if not tty_handoff:
            return "This action requires a local terminal."

        configured = manifest.arms
        if not configured:
            return "No arms configured."

        targets = _resolve_targets(manifest, kwargs)
        if not targets:
            return "All arms are already calibrated."

        from roboclaw.embodied.toolkit.tty import TtySession

        results: list[str] = []
        for arm in targets:
            self._arm = arm
            self._cal_manifest = manifest
            argv = CommandBuilder.calibrate(arm)
            if _has_existing_calibration(arm):
                _print_existing_calibration_tip()

            await tty_handoff(start=True, label=f"Calibrating: {arm.alias}")
            try:
                await self.board.update(calibration_arm=arm.alias)
                await self.start(argv, initial_state=SessionState.CALIBRATING, auto_confirm=False)
                await TtySession(tty_handoff).run(self)
                step = self.board.get("calibration_step", "")
                if step == "done":
                    manifest.mark_arm_calibrated(arm.alias)
                    _sync_calibration_to_motors(arm)
                    results.append(f"{arm.alias}: OK")
                else:
                    results.append(f"{arm.alias}: incomplete")
            except Exception:
                logger.exception("Calibration failed for {}", arm.alias)
                results.append(f"{arm.alias}: FAILED")
            finally:
                await tty_handoff(start=False, label=f"Calibrating: {arm.alias}")

        manifest.reload()
        ok = sum(1 for r in results if r.endswith(": OK"))
        fail = len(results) - ok
        return f"{ok} succeeded, {fail} failed.\n" + "\n".join(results)


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolve_targets(manifest: Manifest, kwargs: dict[str, Any]) -> list[Binding]:
    selected = resolve_action_arms(manifest, kwargs.get("arms", ""))
    if kwargs.get("arms", ""):
        return selected
    return [arm for arm in selected if not arm.calibrated]


def _sync_calibration_to_motors(arm: Binding) -> None:
    """Sync calibration data to motor EEPROM after successful calibration."""
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
