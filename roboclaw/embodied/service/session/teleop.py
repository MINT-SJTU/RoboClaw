"""TeleopSession — interactive teleoperation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board import Board, OutputConsumer, SessionState
from roboclaw.embodied.command import CommandBuilder
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TeleopOutputConsumer(OutputConsumer):
    """Detects teleoperation start from subprocess output."""

    async def parse_line(self, line: str) -> None:
        if self.board.get("state") != SessionState.PREPARING:
            return
        if "Teleop loop time" in line:
            await self.board.update(state=SessionState.TELEOPERATING)


class TeleopSession(Session):
    """Teleoperation session.

    CLI entry: teleoperate(manifest, kwargs, tty_handoff)
    Web entry: EmbodiedService.start_teleop() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent
        self._kwargs: dict[str, Any] = {}

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return TeleopOutputConsumer(board, stdout)

    # -- CLI entry point ---------------------------------------------------

    async def teleoperate(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        self._kwargs = kwargs
        if tty_handoff:
            self._parent.acquire_embodiment("teleop")
            try:
                fps = kwargs.get("fps", 30)
                argv = CommandBuilder.teleop(manifest, fps=fps)
                await self.start(argv)
                from roboclaw.embodied.toolkit.tty import TtySession

                return await TtySession(tty_handoff).run(self)
            finally:
                self._parent.release_embodiment()
        return "This action requires a local terminal."

    # -- CLI protocol ------------------------------------------------------

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="lerobot-teleoperate")

    def status_line(self) -> str:
        s = self.board.state
        state = s.get("state", "idle")
        if state == "idle":
            return "  idle"
        if state == "preparing":
            return "  preparing..."
        elapsed = s.get("elapsed_seconds", 0)
        return f"  teleoperating  | {elapsed:.0f}s  (press Ctrl+C or ESC to stop)"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()

    def result(self) -> str:
        s = self.board.state
        error = s.get("error", "")
        if error:
            return f"Teleoperation failed: {error}"
        return "Teleoperation finished."

    async def stop(self) -> None:
        if self.busy:
            await super().stop()
