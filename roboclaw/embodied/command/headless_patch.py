"""Headless-safe keyboard listener patch for LeRobot record flows."""

from __future__ import annotations

import inspect
import os
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from types import TracebackType
from typing import Any

_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _record_diagnostics_enabled() -> bool:
    value = os.environ.get("ROBOCLAW_RECORD_DIAGNOSTICS", "").strip().lower()
    return value not in _FALSE_VALUES


def _record_debug(message: str) -> None:
    if _record_diagnostics_enabled():
        print(f"[roboclaw.record] {message}", flush=True)


class TTYKeyboardListener:
    """Minimal TTY listener compatible with LeRobot's listener usage."""

    def __init__(self, on_press: Callable[[str], None], stream: object | None = None):
        self._on_press = on_press
        self._stream = stream if stream is not None else sys.stdin
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fd = self._stream.fileno()
        self._old_attrs: list[int] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if os.isatty(self._fd):
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="tty-keyboard-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def __enter__(self) -> TTYKeyboardListener:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _run(self) -> None:
        pending = ""
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            chunk = os.read(self._fd, 32).decode("utf-8", errors="ignore")
            if not chunk:
                continue
            pending += chunk
            pending = self._consume_pending(pending)

    def _consume_pending(self, pending: str) -> str:
        while pending:
            if pending.startswith("\x1b[C"):
                self._on_press("right")
                pending = pending[3:]
                continue
            if pending.startswith("\x1b[D"):
                self._on_press("left")
                pending = pending[3:]
                continue
            if pending[0] != "\x1b":
                pending = pending[1:]
                continue
            if len(pending) == 1:
                time.sleep(0.03)
                return pending
            self._on_press("esc")
            pending = pending[1:]
        return pending


def apply_headless_patch() -> None:
    """Patch LeRobot to use a TTY listener instead of relying on pynput/X11."""

    import lerobot.utils.control_utils as control_utils
    import lerobot.utils.utils as lerobot_utils

    # Patch log_say to print to stdout (so the parent process can parse it)
    # and force blocking=False. LeRobot's finally-block calls
    # log_say("Stop recording", blocking=True) which runs `spd-say --wait`
    # and hangs forever if speech-dispatcher is not installed. Using
    # blocking=False lets spd-say fire-and-forget or fail silently.
    _original_say = lerobot_utils.say

    def _log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
        print(f"[lerobot] {text}", flush=True)
        if play_sounds:
            _original_say(text, blocking=False)

    lerobot_utils.log_say = _log_say
    lerobot_utils.say = lambda text, blocking=False: _original_say(text, blocking=False)

    def init_keyboard_listener():
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }

        def on_press(key: str) -> None:
            if key == "right":
                _record_debug(f"key=right before {_events_text(events)}")
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
                return
            if key == "left":
                _record_debug(f"key=left before {_events_text(events)}")
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
                return
            if key == "esc":
                _record_debug(f"key=esc before {_events_text(events)}")
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True

        listener = TTYKeyboardListener(on_press)
        listener.start()
        return listener, events

    control_utils.init_keyboard_listener = init_keyboard_listener
    control_utils.is_headless = lambda: not sys.stdin.isatty()


def apply_motor_read_retry_patch(motors_bus_module: Any | None = None, min_retries: int = 3) -> None:
    """Retry transient record-time position read failures before failing."""

    if motors_bus_module is None:
        import lerobot.motors.motors_bus as motors_bus_module

    bus_cls = motors_bus_module.SerialMotorsBus
    original = bus_cls.sync_read
    if getattr(original, "_roboclaw_present_position_retry", False):
        return
    signature = inspect.signature(original)

    def sync_read_with_present_position_retry(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return original(self, *args, **kwargs)
        except ConnectionError as exc:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            data_name = bound.arguments.get("data_name")
            num_retry = bound.arguments.get("num_retry", 0)
            if not _should_retry_present_position(data_name, num_retry, min_retries, exc):
                raise
            _record_debug(
                "motor_sync_read_retry.retrying "
                f"data={data_name} min_retries={min_retries} error={exc}"
            )
            retry_kwargs = {**bound.kwargs, "num_retry": min_retries}
            result = original(*bound.args, **retry_kwargs)
            _record_debug(f"motor_sync_read_retry.recovered data={data_name}")
            return result

    sync_read_with_present_position_retry._roboclaw_present_position_retry = True
    bus_cls.sync_read = sync_read_with_present_position_retry
    _record_debug(f"motor_sync_read_retry.enabled data=Present_Position retries={min_retries}")


def apply_record_loop_patch(record_module: Any) -> None:
    """Guard LeRobot record loops against stale skip-reset key events."""

    original = record_module.record_loop
    if getattr(original, "_roboclaw_skip_reset_guard", False):
        return

    def guarded_record_loop(*args: Any, **kwargs: Any) -> Any:
        events, dataset = _record_loop_context(kwargs)
        is_reset_loop = dataset is None
        phase = "reset" if is_reset_loop else "episode"
        if _clear_stale_episode_exit(is_reset_loop, events):
            _record_debug("record_loop.stale_exit_early_before_episode_cleared")
        _record_debug(f"record_loop.start phase={phase} events={_events_text(events)}")
        try:
            return original(*args, **kwargs)
        except ConnectionError as exc:
            if is_reset_loop and _skip_reset_requested(events) and _transient_sync_read_error(exc):
                _record_debug(
                    "record_loop.skip_reset_connection_error_suppressed "
                    f"type={type(exc).__name__} message={exc}"
                )
                # Normal reset-loop skip consumes this flag before breaking.
                events["exit_early"] = False
                return None
            raise
        finally:
            _record_debug(f"record_loop.end phase={phase} events={_events_text(events)}")

    guarded_record_loop._roboclaw_skip_reset_guard = True
    record_module.record_loop = guarded_record_loop


def _clear_stale_episode_exit(is_reset_loop: bool, events: Any) -> bool:
    if is_reset_loop or not isinstance(events, dict):
        return False
    if not events.get("exit_early"):
        return False
    if events.get("rerecord_episode") or events.get("stop_recording"):
        return False
    events["exit_early"] = False
    return True


def _record_loop_context(kwargs: dict[str, Any]) -> tuple[Any, Any]:
    if "events" not in kwargs:
        raise RuntimeError("RoboClaw record_loop patch requires LeRobot keyword argument 'events'.")
    return kwargs["events"], kwargs.get("dataset")


def _should_retry_present_position(
    data_name: Any, num_retry: Any, min_retries: int, exc: ConnectionError,
) -> bool:
    if data_name != "Present_Position" or (num_retry or 0) >= min_retries:
        return False
    return _transient_sync_read_error(exc)


def _skip_reset_requested(events: Any) -> bool:
    if not isinstance(events, dict):
        return False
    return (
        bool(events.get("exit_early"))
        and not events.get("rerecord_episode")
        and not events.get("stop_recording")
    )


def _transient_sync_read_error(exc: ConnectionError) -> bool:
    # LeRobot's Dynamixel sync-read timeout is reported only as text.
    message = str(exc).lower()
    return "sync read" in message and "no status packet" in message


def _events_text(events: Any) -> str:
    if not isinstance(events, dict):
        return "unavailable"
    exit_early = events.get("exit_early")
    rerecord = events.get("rerecord_episode")
    stop = events.get("stop_recording")
    return f"exit_early={exit_early} rerecord={rerecord} stop={stop}"
