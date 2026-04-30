"""Calibration Session — runs lerobot-calibrate subprocess per arm.

Same architecture as TeleopSession / RecordSession:
  Session → subprocess → OutputConsumer → Board → WebSocket → Frontend
  Frontend → Board.post_command() → InputConsumer → subprocess stdin
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from roboclaw.embodied.board.constants import Command, SessionState
from roboclaw.embodied.board.consumer import InputConsumer, OutputConsumer
from roboclaw.embodied.command import CommandBuilder, resolve_action_arms
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService

_RE_POSITION_ROW = re.compile(
    r"(\w+)\s+\|\s+(-?\d+)\s+\|\s+(-?\d+)\s+\|\s+(-?\d+)"
)


# ── Consumers ────────────────────────────────────────────────────────────


class CalibrationOutputConsumer(OutputConsumer):
    """Parse lerobot-calibrate subprocess stdout → Board state updates."""

    def __init__(self, board: Any, stdout: Any) -> None:
        super().__init__(board, stdout)
        self._positions: dict[str, dict[str, int]] = {}

    async def parse_line(self, line: str) -> None:
        # Hot path: position table rows are the most frequent output
        m = _RE_POSITION_ROW.search(line)
        if m:
            name = m.group(1)
            new_val = {"min": int(m.group(2)), "pos": int(m.group(3)), "max": int(m.group(4))}
            if self._positions.get(name) != new_val:
                self._positions[name] = new_val
                await self.board.update(calibration_positions=dict(self._positions))
            return

        low = line.lower()
        if "press enter to use provided calibration" in low:
            await self.board.update(
                state=SessionState.CALIBRATING,
                calibration_step="choose",
                calibration_phase="choose",
            )
        elif "running calibration" in low:
            await self.board.update(calibration_step="starting", calibration_phase="starting")
        elif "move" in low and "middle" in low and "press enter" in low:
            await self.board.update(calibration_step="homing", calibration_phase="homing")
        elif "recording positions" in low and "press enter to stop" in low:
            await self.board.update(calibration_step="recording", calibration_phase="recording")
        elif "calibration saved" in low:
            await self.board.update(calibration_step="done", calibration_phase="done")


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
        argv = CommandBuilder.calibrate(arm)
        await self.start(argv, initial_state=SessionState.CALIBRATING, auto_confirm=False)
        # Set after start() — start() calls board.reset() which would clear it
        await self.board.update(
            calibration_mode="manual",
            calibration_scope="single",
            calibration_phase="starting",
            calibration_current_arm=arm.alias,
            calibration_index=1,
            calibration_total=1,
            calibration_results=[{
                "alias": arm.alias,
                "status": "running",
                "reason": "",
                "started_at": None,
                "finished_at": None,
            }],
            calibration_arm=arm.alias,
        )

    async def _wait_process(self) -> None:
        """Finalize calibration only after lerobot saves the standard JSON."""
        assert self._process is not None
        try:
            rc = self._process.returncode
            if rc is None:
                rc = await self._process.wait()
            logger.info("Calibration subprocess exited with code {}", rc)
            await self._teardown()

            if self._stopped:
                return

            if rc not in (0, None, -2, 130, -15):
                await self.board.update(
                    state=SessionState.ERROR,
                    error=self._format_exit_error(rc),
                )
                return

            if rc != 0:
                await self.board.update(state=SessionState.IDLE)
                return

            try:
                _mark_calibration_success(self._arm, self._cal_manifest)
            except Exception as exc:
                await self.board.update(
                    state=SessionState.ERROR,
                    error=str(exc),
                    calibration_phase="failed",
                    calibration_error=str(exc),
                    calibration_results=[{
                        "alias": self._arm.alias if self._arm else "",
                        "status": "failed",
                        "reason": str(exc),
                        "started_at": None,
                        "finished_at": None,
                    }],
                )
                return

            await self.board.update(
                state=SessionState.IDLE,
                calibration_phase="done",
                calibration_results=[{
                    "alias": self._arm.alias if self._arm else "",
                    "status": "success",
                    "reason": "",
                    "started_at": None,
                    "finished_at": None,
                }],
            )
        finally:
            if self._exit_callback and not self._stopped:
                self._exit_callback(self)
            elif not self._exit_callback:
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
        if state.get("state") == SessionState.ERROR:
            return f"Calibration of {alias} failed: {state.get('error', 'unknown error')}"
        if step == "done":
            return f"Calibration of {alias} completed successfully."
        return f"Calibration of {alias} ended."

    async def stop(self) -> None:
        if not self._parent.embodiment_busy:
            return
        await super().stop()

    # -- Agent entry point (passthrough to TTY) ------------------------------

    async def calibrate(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        """Agent/CLI entry: run calibration subprocess with inherited TTY.

        Unlike the web path (start_calibration → Session.start), the agent
        path passes the subprocess directly to the terminal so the user
        can interact with lerobot-calibrate's input() prompts natively.
        """
        if not tty_handoff:
            return "This action requires a local terminal."

        configured = manifest.arms
        if not configured:
            return "No arms configured."

        targets = _resolve_targets(manifest, kwargs)
        if not targets:
            return "All arms are already calibrated."

        from roboclaw.embodied.executor import SubprocessExecutor

        self._parent.acquire_embodiment("calibrating")
        try:
            runner = SubprocessExecutor()
            results: list[str] = []
            for arm in targets:
                result = await self._calibrate_one_tty(arm, manifest, runner, tty_handoff)
                if result == "interrupted":
                    return "interrupted"
                results.append(result)

            ok = sum(1 for r in results if r.endswith(": OK"))
            fail = len(results) - ok
            return f"{ok} succeeded, {fail} failed.\n" + "\n".join(results)
        finally:
            self._parent.release_embodiment()

    async def _calibrate_one_tty(
        self,
        arm: Binding,
        manifest: Manifest,
        runner: Any,
        tty_handoff: Any,
    ) -> str:
        """Calibrate one arm with inherited TTY (agent/CLI path)."""
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
            try:
                _mark_calibration_success(arm, manifest)
            except Exception as exc:
                return f"{display}: FAILED ({exc})"
            return f"{display}: OK"
        msg = f"{display}: FAILED (exit {rc})"
        if stderr_text.strip():
            msg += f"\nstderr: {stderr_text.strip()}"
        return msg


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolve_targets(manifest: Manifest, kwargs: dict[str, Any]) -> list[Binding]:
    selected = resolve_action_arms(manifest, kwargs.get("arms", ""))
    if kwargs.get("arms", ""):
        return selected
    return [arm for arm in selected if not arm.calibrated]


def _mark_calibration_success(arm: Binding | None, manifest: Manifest | None) -> None:
    """Persist calibration only after lerobot saves the standard JSON."""
    if arm is None or manifest is None:
        raise RuntimeError("Calibration session is missing arm context.")
    if not arm.calibration_dir or not arm.arm_id:
        raise RuntimeError(f"Calibration for {arm.alias} has no usable calibration directory.")

    cal_path = Path(arm.calibration_dir) / f"{arm.arm_id}.json"
    if not cal_path.exists():
        raise RuntimeError(
            f"Calibration for {arm.alias} did not save {cal_path.name}."
        )

    manifest.mark_arm_calibrated(arm.alias)
