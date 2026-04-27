"""Consumers — bridge between subprocess I/O and the Board."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import TYPE_CHECKING

from roboclaw.embodied.board.constants import Command

_FALSE_VALUES = {"", "0", "false", "no", "off"}

if TYPE_CHECKING:
    from roboclaw.embodied.board.board import Board


class Consumer:
    """Base class for Board I/O consumers."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        raise NotImplementedError


class OutputConsumer(Consumer):
    """Reads subprocess stdout, parses lines, updates Board.

    Subclasses override parse_line() for operation-specific parsing.

    Uses chunked reads with a timeout so that prompts from ``input()``
    (which do **not** end with ``\\n``) are still processed promptly
    instead of being stuck in the StreamReader buffer forever.
    """

    _PARTIAL_LINE_TIMEOUT: float = 0.5

    def __init__(self, board: Board, stdout: asyncio.StreamReader) -> None:
        super().__init__(board)
        self._stdout = stdout

    async def _run(self) -> None:
        buf = b""
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self._stdout.read(4096),
                        timeout=self._PARTIAL_LINE_TIMEOUT,
                    )
                    if not chunk:
                        # EOF — flush remaining buffer
                        for line in self._extract_lines(buf):
                            await self._dispatch(line)
                        break
                    buf += chunk
                except asyncio.TimeoutError:
                    # No new data — flush any partial (non-newline-terminated) line
                    if buf:
                        for line in self._extract_lines(buf):
                            await self._dispatch(line)
                        buf = b""
                    continue

                # Eagerly process all complete lines; keep the trailing fragment
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                    if line:
                        await self._dispatch(line)
        except (OSError, ConnectionError):
            pass

    # -- helpers --

    @staticmethod
    def _extract_lines(buf: bytes) -> list[str]:
        """Decode buffer into non-empty lines (splitting on \\n and \\r)."""
        text = buf.decode("utf-8", errors="replace")
        return [line for line in text.splitlines() if line.strip()]

    async def _dispatch(self, line: str) -> None:
        self.board.log(line)
        await self.parse_line(line)

    async def parse_line(self, line: str) -> None:
        """Override in subclasses for operation-specific parsing."""


class InputConsumer(Consumer):
    """Reads commands from Board, translates to stdin bytes for subprocess."""

    _KEYMAP: dict[str, bytes] = {
        Command.SAVE_EPISODE: b"\x1b[C",       # right arrow
        Command.DISCARD_EPISODE: b"\x1b[D",    # left arrow
        Command.SKIP_RESET: b"\x1b[C",         # right arrow
        Command.STOP: b"\x1b",                 # ESC
        Command.CONFIRM: b"\n",                # Enter
    }

    def __init__(self, board: Board, stdin: asyncio.StreamWriter) -> None:
        super().__init__(board)
        self._stdin = stdin
        self._notify = asyncio.Event()

    def _on_command_posted(self) -> None:
        """Called by Board when a command is posted. Wakes the consumer."""
        self._notify.set()

    async def _run(self) -> None:
        try:
            while True:
                await self._notify.wait()
                self._notify.clear()
                while (cmd := self.board.poll_command()) is not None:
                    data = self.translate(cmd)
                    if data:
                        self._log_command_delivery(cmd, data)
                        self._stdin.write(data)
                        await self._stdin.drain()
        except (OSError, ConnectionError, asyncio.CancelledError):
            pass

    def translate(self, command: str) -> bytes | None:
        """Command string -> stdin bytes. Subclasses can override."""
        return self._KEYMAP.get(command)

    def _log_command_delivery(self, command: str, data: bytes) -> None:
        if not _record_diagnostics_enabled():
            return
        state = self.board.state
        session_state = state.get("state", "")
        episode_phase = state.get("episode_phase", "")
        self.board.log(
            "[roboclaw.input] "
            f"command={command} bytes=0x{data.hex()} "
            f"state={session_state} phase={episode_phase}"
        )


def _record_diagnostics_enabled() -> bool:
    value = os.environ.get("ROBOCLAW_RECORD_DIAGNOSTICS", "").strip().lower()
    return value not in _FALSE_VALUES
