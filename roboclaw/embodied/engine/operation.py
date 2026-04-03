"""Subprocess lifecycle for teleop/record operations.

Manages one LeRobot subprocess at a time (teleop or recording) with:
- State machine: idle → preparing → teleoperating/recording → idle
- stdin control for episode save/discard/skip-reset/ESC
- stdout parsing for episode lifecycle tracking
- Callback-driven progress notifications

Used by EmbodiedService so both Web and CLI can share
the same subprocess orchestration.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import signal
import time
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.embodied.engine.helpers import (
    ActionError,
    prepare_record,
    prepare_teleop,
)
from roboclaw.embodied.hardware.port_lock import port_locks
from roboclaw.embodied.runner import LocalLeRobotRunner

_RE_RECORDING_EP = re.compile(r"Recording episode (\d+)")
_RE_EPISODE_DONE = re.compile(r"Episode (\d+) done")
_RE_FRAME_COUNT = re.compile(r"frame[s]?\s*[:=]\s*(\d+)", re.IGNORECASE)

# Serial devices may still be held by the OS after the previous process exits.
# 5 seconds is empirically sufficient for V4L2/ttyUSB release on Linux.
_DRAIN_SECONDS = 5
_GRACEFUL_STOP_TIMEOUT = 15
_RERUN_GRPC_PORT = 9876
_RERUN_WEB_PORT = 9877
_RERUN_BIND = "0.0.0.0"

StatusCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class OperationEngine:
    """Manages one LeRobot subprocess (teleop or recording) at a time."""

    def __init__(self, on_state_change: StatusCallback | None = None) -> None:
        self._on_state_change = on_state_change
        self._state = "idle"
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task | None = None
        self._wait_task: asyncio.Task | None = None
        self._cancel_prepare = asyncio.Event()
        self._held_port_locks: list[str] = []
        self._rerun_process: asyncio.subprocess.Process | None = None
        self._rerun_grpc_port = 0
        self._rerun_web_port = 0
        self._error_message = ""
        self._stderr_lines: list[str] = []

        # Recording metadata
        self._dataset_name = ""
        self._dataset_root = ""
        self._target_episodes = 0

        # Episode tracking (updated by stdout parser)
        self._episode_phase = ""  # "recording" | "saving" | "resetting" | ""
        self._saved_episodes = 0
        self._current_episode = 0
        self._total_frames = 0
        self._start_time = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def busy(self) -> bool:
        return self._state != "idle"

    def get_status(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return {
            "state": self._state,
            "episode_phase": self._episode_phase,
            "saved_episodes": self._saved_episodes,
            "current_episode": self._current_episode,
            "target_episodes": self._target_episodes,
            "total_frames": self._total_frames,
            "elapsed_seconds": round(elapsed, 1),
            "dataset": self._dataset_name if self._state == "recording" else None,
            "rerun_web_port": self._rerun_web_port if self._rerun_process else 0,
            "error": self._error_message,
        }

    # -- Lifecycle ---------------------------------------------------------

    async def start_teleop(self, *, fps: int = 30, setup: dict[str, Any] | None = None) -> None:
        self._require_idle_or_raise()
        self._error_message = ""
        self._stderr_lines = []
        manifest = setup
        if manifest is None:
            from roboclaw.embodied.manifest.helpers import load_manifest
            manifest = load_manifest()
        await self._start_rerun_server()
        try:
            argv = prepare_teleop(
                manifest, {"fps": fps}, **self._rerun_display_kwargs(),
            )
        except ActionError as exc:
            await self._stop_rerun_server()
            raise RuntimeError(str(exc)) from exc

        await self._transition_through_preparing("teleoperating", argv, manifest)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
        setup: dict[str, Any] | None = None,
    ) -> str:
        """Start recording. Returns the generated dataset_name."""
        self._error_message = ""
        self._stderr_lines = []
        if self._state == "teleoperating":
            await self._kill_subprocess()
            self._set_state("idle")
        self._require_idle_or_raise()

        manifest = setup
        if manifest is None:
            from roboclaw.embodied.manifest.helpers import load_manifest
            manifest = load_manifest()
        await self._start_rerun_server()
        kwargs: dict[str, Any] = {
            "task": task,
            "num_episodes": num_episodes,
            "fps": fps,
            "episode_time_s": episode_time_s,
            "reset_time_s": reset_time_s,
        }
        try:
            argv, dataset_name, dataset_root = prepare_record(
                manifest, kwargs, **self._rerun_display_kwargs(),
            )
        except ActionError as exc:
            await self._stop_rerun_server()
            raise RuntimeError(str(exc)) from exc

        self._dataset_name = dataset_name
        self._dataset_root = dataset_root
        self._target_episodes = num_episodes
        self._saved_episodes = 0
        self._current_episode = 0
        self._total_frames = 0
        self._episode_phase = ""

        await self._transition_through_preparing("recording", argv, manifest)
        return dataset_name

    async def stop(self) -> None:
        if self._state == "idle":
            return
        if self._state == "preparing":
            self._cancel_prepare.set()
            # Wait briefly for the prepare coroutine to notice cancellation
            await asyncio.sleep(0.1)
            if self._state == "idle":
                return
        if self._state == "recording":
            await self._graceful_stop()
        else:
            await self._kill_subprocess()
        self._episode_phase = ""
        self._set_state("idle")

    # -- Episode control ---------------------------------------------------

    async def save_episode(self) -> None:
        await self._send_key(b"\x1b[C")
        logger.info("Sent save-episode signal (right arrow)")

    async def discard_episode(self) -> None:
        await self._send_key(b"\x1b[D")
        logger.info("Sent discard-episode signal (left arrow)")

    async def skip_reset(self) -> None:
        await self._send_key(b"\x1b[C")
        logger.info("Sent skip-reset signal (right arrow)")

    # -- Internal ----------------------------------------------------------

    def _rerun_display_kwargs(self) -> dict[str, Any]:
        """Build display kwargs for ArmCommandBuilder methods."""
        ok = self._rerun_process is not None
        return {
            "display_data": ok,
            "display_ip": _RERUN_BIND,
            "display_port": self._rerun_grpc_port if ok else 0,
        }

    async def _start_rerun_server(self) -> bool:
        """Start a Rerun server, auto-retry on port conflict. Returns True on success."""
        if self._rerun_process is not None:
            return True
        grpc, web = _RERUN_GRPC_PORT, _RERUN_WEB_PORT
        for _ in range(5):
            try:
                result = await self._try_launch_rerun(grpc, web)
            except FileNotFoundError:
                logger.warning("Rerun CLI not found — visualization disabled")
                return False
            except OSError as exc:
                logger.warning("Failed to start Rerun: {}", exc)
                return False
            if result is True:
                return True
            if result is False:
                return False
            grpc += 2
            web += 2
        logger.warning("Rerun: all port attempts exhausted")
        return False

    async def _try_launch_rerun(self, grpc_port: int, web_port: int) -> bool | None:
        """Try one port pair. Returns True=success, False=fatal, None=port conflict."""
        proc = await asyncio.create_subprocess_exec(
            "rerun", "--serve-web",
            "--port", str(grpc_port),
            "--web-viewer-port", str(web_port),
            "--bind", _RERUN_BIND,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(40):
            if proc.returncode is not None:
                break
            await asyncio.sleep(0.05)
        if proc.returncode is None:
            self._rerun_process = proc
            self._rerun_grpc_port = grpc_port
            self._rerun_web_port = web_port
            logger.info("Rerun server started (gRPC={}, web={})", grpc_port, web_port)
            return True
        stderr = await proc.stderr.read() if proc.stderr else b""
        msg = stderr.decode(errors="replace")[:200]
        if "Address already in use" in msg:
            logger.info("Rerun port {} in use, trying next", web_port)
            return None
        logger.warning("Rerun exited (code={}): {}", proc.returncode, msg)
        return False

    async def _stop_rerun_server(self) -> None:
        proc = self._rerun_process
        if proc is None:
            return
        self._rerun_process = None
        self._rerun_grpc_port = 0
        self._rerun_web_port = 0
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, PermissionError):
                logger.debug("Rerun process already terminated")
        logger.info("Rerun server stopped")

    def _require_idle_or_raise(self) -> None:
        if self._state != "idle":
            raise RuntimeError(f"Session busy (state={self._state})")

    async def _transition_through_preparing(
        self, target_state: str, argv: list[str], manifest: dict[str, Any],
    ) -> None:
        self._cancel_prepare.clear()
        self._set_state("preparing")
        # Wait for drain, but allow cancellation via stop()
        try:
            await asyncio.wait_for(self._cancel_prepare.wait(), timeout=_DRAIN_SECONDS)
            self._set_state("idle")
            return
        except asyncio.TimeoutError:
            pass
        if self._cancel_prepare.is_set():
            self._set_state("idle")
            return
        # Acquire port locks for all arm serial ports before launching subprocess
        arm_ports = sorted({arm["port"] for arm in manifest.get("arms", []) if arm.get("port")})
        for port in arm_ports:
            await port_locks._get_lock(port).acquire()
        self._held_port_locks = arm_ports
        # Flush serial buffers to clear residual data from servo polling
        await asyncio.to_thread(self._flush_serial_ports, arm_ports)
        try:
            await self._launch_subprocess(argv)
        except Exception:
            self._release_port_locks()
            raise
        self._set_state(target_state)

    async def _launch_subprocess(self, argv: list[str]) -> None:
        runner = LocalLeRobotRunner()
        proc = await runner.run_streaming_interactive(argv)
        self._process = proc
        self._start_time = time.monotonic()
        # Auto-confirm calibration prompts
        if proc.stdin:
            proc.stdin.write(b"\n\n\n\n")
            await proc.stdin.drain()
        self._stdout_task = asyncio.create_task(
            self._read_stdout(proc), name="dashboard-stdout",
        )
        self._wait_task = asyncio.create_task(
            self._wait_for_exit(proc), name="dashboard-wait",
        )
        logger.info("Launched subprocess pid={}: {}", proc.pid, " ".join(argv[:5]))

    async def _read_stdout(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            logger.info("subprocess: {}", line)
            # Capture error-like lines for crash reporting
            if any(kw in line for kw in ("Error", "Traceback", "Exception", "FATAL")):
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > 20:
                    self._stderr_lines = self._stderr_lines[-20:]
            if self._parse_line(line):
                await self._emit_state_change(self.get_status())

    async def _wait_for_exit(self, proc: asyncio.subprocess.Process) -> None:
        """Wait for subprocess to exit and clean up session state."""
        if self._stdout_task is not None:
            await self._stdout_task
        await proc.wait()
        returncode = proc.returncode
        logger.info("Subprocess exited with code {}", returncode)
        # Detect crash and set error message
        if returncode not in (0, None, -2, -15):  # not OK / SIGINT / SIGTERM
            tail = "\n".join(self._stderr_lines[-5:]) if self._stderr_lines else ""
            self._error_message = f"Process exited with code {returncode}"
            if tail:
                self._error_message += f"\n{tail}"
        # Clean up
        self._close_stdin()
        self._process = None
        self._release_port_locks()
        await self._stop_rerun_server()
        self._episode_phase = ""
        self._set_state("idle")

    def _parse_line(self, line: str) -> bool:
        """Parse LeRobot stdout to track episode lifecycle. Returns True if state changed."""
        m = _RE_RECORDING_EP.search(line)
        if m:
            if self._episode_phase in ("saving", "resetting"):
                self._saved_episodes += 1
            self._current_episode = int(m.group(1))
            self._episode_phase = "recording"
            return True

        if "Right arrow key pressed" in line:
            if self._episode_phase in ("recording", "resetting"):
                self._episode_phase = "saving"
                return True
            return False

        if "Reset the environment" in line:
            self._episode_phase = "resetting"
            return True

        if "Re-record episode" in line:
            self._episode_phase = "recording"
            return True

        if "Stop recording" in line:
            if self._episode_phase in ("saving", "resetting"):
                self._saved_episodes += 1
            self._episode_phase = ""
            return True

        m = _RE_FRAME_COUNT.search(line)
        if m:
            self._total_frames = int(m.group(1))
            return False

        return False

    async def _send_key(self, key: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("No subprocess stdin available")
        self._process.stdin.write(key)
        await self._process.stdin.drain()

    async def _graceful_stop(self) -> None:
        """Send ESC for graceful stop, wait, fallback to SIGINT."""
        if self._process is None:
            return
        # Send ESC key
        if self._process.stdin is not None:
            try:
                self._process.stdin.write(b"\x1b\n")
                await self._process.stdin.drain()
            except (OSError, ConnectionResetError):
                logger.debug("Stdin write failed during graceful stop")

        # Wait for process to exit
        try:
            await asyncio.wait_for(self._process.wait(), timeout=_GRACEFUL_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Subprocess did not exit in {}s, sending SIGINT", _GRACEFUL_STOP_TIMEOUT)
            self._send_sigint()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

        await self._await_stdout_task()
        self._close_stdin()
        self._process = None
        self._release_port_locks()
        await self._stop_rerun_server()

    async def _kill_subprocess(self) -> None:
        if self._process is None:
            return
        self._close_stdin()
        self._send_sigint()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._process.kill()
        await self._await_stdout_task()
        self._process = None
        self._release_port_locks()
        await self._stop_rerun_server()

    async def _await_stdout_task(self) -> None:
        """Wait for the stdout reader to finish before resetting state."""
        if self._stdout_task is not None:
            try:
                await asyncio.wait_for(self._stdout_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stdout_task.cancel()
            self._stdout_task = None
        # Cancel the wait task since we're handling cleanup ourselves
        if self._wait_task is not None:
            self._wait_task.cancel()
            try:
                await self._wait_task
            except asyncio.CancelledError:
                pass
            self._wait_task = None

    def _send_sigint(self) -> None:
        if self._process is None or self._process.returncode is not None:
            return
        try:
            self._process.send_signal(signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass

    def _close_stdin(self) -> None:
        if self._process is not None and self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                logger.debug("Stdin already closed")

    @staticmethod
    def _flush_serial_ports(ports: list[str]) -> None:
        """Open and flush each serial port to clear residual buffer data."""
        import serial
        for port in ports:
            try:
                with serial.Serial(port, baudrate=1_000_000, timeout=0.1) as ser:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
            except (OSError, serial.SerialException):
                logger.debug("Serial flush skipped for %s", port)

    def _release_port_locks(self) -> None:
        for port in self._held_port_locks:
            lock = port_locks._get_lock(port)
            if lock.locked():
                lock.release()
        self._held_port_locks = []

    def _set_state(self, new_state: str) -> None:
        if self._state == new_state:
            return
        self._state = new_state
        status = self.get_status()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_state_change(status), name="dashboard-notify")
        except RuntimeError:
            logger.debug("No event loop for state notification")

    async def _emit_state_change(self, status: dict[str, Any]) -> None:
        if self._on_state_change is None:
            return
        try:
            result = self._on_state_change(status)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Error in dashboard state change callback")
