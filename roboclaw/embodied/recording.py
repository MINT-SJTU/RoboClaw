"""Recording session manager for LeRobot data collection.

Manages a LeRobot recording subprocess lifecycle with real-time stdout
progress parsing and callback-based state notifications.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import json
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from loguru import logger

from roboclaw.embodied.ops.helpers import _is_interrupted
from roboclaw.embodied.runner import LocalLeRobotRunner

_RE_EPISODE_START = re.compile(r"Recording episode (\d+)")
_RE_EPISODE_DONE = re.compile(r"Episode (\d+) done")
_RE_FRAME_COUNT = re.compile(r"frame[s]?\s*[:=]\s*(\d+)", re.IGNORECASE)

StatusCallback = Callable[["RecordingStatus"], Awaitable[None] | None]


@dataclass
class RecordingStatus:
    """Snapshot of the current recording state."""

    session_id: str
    dataset_name: str
    dataset_root: str
    task: str
    state: str  # "starting", "recording", "completed", "error"
    current_episode: int
    total_episodes: int
    total_frames: int
    elapsed_seconds: float
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecordingSession:
    """Manages a single LeRobot recording subprocess with progress tracking."""

    def __init__(
        self,
        argv: list[str],
        dataset_name: str,
        dataset_root: str,
        task: str,
        total_episodes: int,
        on_progress: StatusCallback,
        on_completed: StatusCallback,
        on_error: StatusCallback,
    ) -> None:
        self.session_id = uuid4().hex
        self._argv = argv
        self._dataset_name = dataset_name
        self._dataset_root = dataset_root
        self._task = task
        self._total_episodes = total_episodes
        self._on_progress = on_progress
        self._on_completed = on_completed
        self._on_error = on_error

        self._process: asyncio.subprocess.Process | None = None
        self._state = "starting"
        self._current_episode = 0
        self._total_frames = 0
        self._start_time = 0.0
        self._stderr_ring: collections.deque[str] = collections.deque(maxlen=50)
        self._error_message = ""

        self._read_stdout_task: asyncio.Task[None] | None = None
        self._read_stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return self._state in {"starting", "recording"}

    @property
    def status(self) -> RecordingStatus:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return RecordingStatus(
            session_id=self.session_id,
            dataset_name=self._dataset_name,
            dataset_root=self._dataset_root,
            task=self._task,
            state=self._state,
            current_episode=self._current_episode,
            total_episodes=self._total_episodes,
            total_frames=self._total_frames,
            elapsed_seconds=round(elapsed, 1),
            error_message=self._error_message,
        )

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_ring)

    async def start(self) -> None:
        if self._process is not None:
            raise RuntimeError(f"Recording session {self.session_id} already started")
        self._start_time = time.monotonic()
        runner = LocalLeRobotRunner()
        self._process = await runner.run_streaming(self._argv)
        logger.info("Recording session {} started (pid={})", self.session_id, self._process.pid)

        self._read_stdout_task = asyncio.create_task(
            self._read_stdout(), name=f"recording-stdout-{self.session_id[:8]}"
        )
        self._read_stderr_task = asyncio.create_task(
            self._read_stderr(), name=f"recording-stderr-{self.session_id[:8]}"
        )
        self._wait_task = asyncio.create_task(
            self._wait_for_exit(), name=f"recording-wait-{self.session_id[:8]}"
        )

    def stop(self) -> None:
        if self._process is None or self._process.returncode is not None:
            return
        logger.info("Stopping recording session {} (SIGINT)", self.session_id)
        self._process.send_signal(signal.SIGINT)

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        async for raw_line in self._process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            logger.debug("recording stdout: {}", line)
            if self._parse_line(line):
                await _emit(self._on_progress, self.status)

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        async for raw_line in self._process.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr_ring.append(line)

    def _parse_line(self, line: str) -> bool:
        """Extract episode/frame progress from a stdout line. Returns True if state changed."""
        m = _RE_EPISODE_START.search(line)
        if m:
            self._current_episode = int(m.group(1))
            self._state = "recording"
            return True

        m = _RE_EPISODE_DONE.search(line)
        if m:
            return True

        m = _RE_FRAME_COUNT.search(line)
        if m:
            self._total_frames = int(m.group(1))
            return True

        return False

    async def _wait_for_exit(self) -> None:
        assert self._process is not None

        if self._read_stdout_task:
            await self._read_stdout_task
        if self._read_stderr_task:
            await self._read_stderr_task

        await self._process.wait()
        returncode = self._process.returncode
        logger.info("Recording session {} exited with code {}", self.session_id, returncode)

        self._read_final_stats()

        if returncode == 0 or _is_interrupted(returncode or 0):
            self._state = "completed"
            await _emit(self._on_completed, self.status)
        else:
            self._state = "error"
            self._error_message = _build_error_message(returncode, self._stderr_ring)
            await _emit(self._on_error, self.status)

    def _read_final_stats(self) -> None:
        info_path = Path(self._dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            return
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total_episodes = info.get("total_episodes")
        if isinstance(total_episodes, int) and total_episodes > 0:
            self._current_episode = total_episodes
        total_frames = info.get("total_frames")
        if isinstance(total_frames, int) and total_frames > 0:
            self._total_frames = total_frames


def _build_error_message(returncode: int | None, stderr_ring: collections.deque[str]) -> str:
    if returncode is not None and returncode < 0:
        try:
            sig_name = signal.Signals(-returncode).name
            msg = f"Recording process killed by signal {sig_name}"
        except ValueError:
            msg = f"Recording process exited with code {returncode}"
    else:
        msg = f"Recording process exited with code {returncode}"
    tail = "\n".join(list(stderr_ring)[-5:])
    if tail:
        return f"{msg}.\nRecent stderr:\n{tail}"
    return msg


async def _emit(callback: Callable[..., Any], *args: Any) -> None:
    """Invoke a callback, awaiting if it returns a coroutine."""
    result = callback(*args)
    if inspect.isawaitable(result):
        await result
