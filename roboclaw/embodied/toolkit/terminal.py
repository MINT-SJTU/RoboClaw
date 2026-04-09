"""Pure terminal utilities — raw mode and non-blocking key reading.

No domain logic; used by TtySession for polling-mode sessions.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from contextlib import contextmanager


@contextmanager
def raw_terminal():
    """Switch stdin to raw mode, restore on exit."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key_nonblocking() -> str | None:
    """Read a keypress from stdin without blocking.

    Returns one of: ``"right"``, ``"left"``, ``"esc"``, ``"ctrl_c"``, or ``None``.
    """
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b"\x03":
        return "ctrl_c"
    if ch == b"\x1b":
        if not select.select([fd], [], [], 0.05)[0]:
            return "esc"
        ch2 = os.read(fd, 1)
        if ch2 == b"[":
            ch3 = os.read(fd, 1)
            if ch3 == b"C":
                return "right"
            if ch3 == b"D":
                return "left"
        return "esc"
    return None
