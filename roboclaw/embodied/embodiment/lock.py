"""Cross-process embodiment lock using fcntl.flock.

Ensures only one process/task accesses the serial ports at a time.
OS auto-releases the lock when the file descriptor is closed (including
process crash / SIGKILL), so no stale locks.

Two lock modes:
- **Exclusive** (teleop, record, calibrate, …): ``LOCK_EX | LOCK_NB``
- **Shared** (servo polling): ``LOCK_SH | LOCK_NB``, skip on failure
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home


class EmbodimentBusyError(RuntimeError):
    """Raised when the embodiment lock cannot be acquired."""


def _default_lock_path() -> Path:
    return get_roboclaw_home() / "workspace" / "embodied" / ".embodiment.lock"


class EmbodimentFileLock:
    """fcntl.flock-based cross-process lock for the physical embodiment."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_lock_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ex_fd: int | None = None
        # Persistent shared fd — avoids open/close churn on every poll
        self._sh_fd: int | None = None
        self._sh_locked: bool = False

    # -- Exclusive (teleop / record / calibrate / …) -----------------------

    def acquire_exclusive(self, owner: str) -> None:
        """Acquire exclusive lock. Raises ``EmbodimentBusyError`` on failure."""
        # Release our own shared lock first — same-process shared fd blocks exclusive
        self.release_shared()
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise EmbodimentBusyError(
                f"Embodiment busy: {self._read_owner()}"
            ) from None
        # Write owner info (informational, for diagnostics)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({"owner": owner, "pid": os.getpid(), "ts": time.time()}).encode())
        self._ex_fd = fd

    def release_exclusive(self) -> None:
        """Release exclusive lock (idempotent)."""
        fd = self._ex_fd
        if fd is None:
            return
        self._ex_fd = None
        try:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)

    # -- Shared (servo polling) --------------------------------------------

    def try_shared(self) -> bool:
        """Try non-blocking shared lock. Returns True on success."""
        if self._sh_fd is None:
            self._sh_fd = os.open(str(self._path), os.O_RDONLY | os.O_CREAT)
        try:
            fcntl.flock(self._sh_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return False
        self._sh_locked = True
        return True

    def release_shared(self) -> None:
        """Release shared lock (idempotent). Keeps fd open for reuse."""
        if not self._sh_locked:
            return
        self._sh_locked = False
        if self._sh_fd is not None:
            fcntl.flock(self._sh_fd, fcntl.LOCK_UN)

    # -- Query -------------------------------------------------------------

    def owner(self) -> str:
        """Read current lock owner from file content (no lock acquired)."""
        return self._read_owner()

    def _read_owner(self) -> str:
        try:
            text = self._path.read_text().strip()
            if not text:
                return "unknown"
            info = json.loads(text)
            return f"{info.get('owner', 'unknown')} (pid {info.get('pid', '?')})"
        except (OSError, json.JSONDecodeError, KeyError):
            return "unknown"
