"""Headless-safe keyboard listener patch for LeRobot record flows."""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from types import TracebackType


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

    # Patch log_say to also print to stdout so the parent process can parse it.
    # The default logging.info() produces no output without a configured handler.
    _original_say = lerobot_utils.say

    def _log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
        print(f"[lerobot] {text}", flush=True)
        if play_sounds:
            _original_say(text, blocking)

    lerobot_utils.log_say = _log_say

    def init_keyboard_listener():
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }

        def on_press(key: str) -> None:
            if key == "right":
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
                return
            if key == "left":
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
                return
            if key == "esc":
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True

        listener = TTYKeyboardListener(on_press)
        listener.start()
        return listener, events

    control_utils.init_keyboard_listener = init_keyboard_listener
    control_utils.is_headless = lambda: not sys.stdin.isatty()
