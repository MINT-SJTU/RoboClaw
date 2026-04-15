"""ReplaySession — dataset replay on follower arms."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board import Board, OutputConsumer, SessionState
from roboclaw.embodied.command import CommandBuilder
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


_PREPARE_STAGES: tuple[tuple[str, str], ...] = (
    ("loading checkpoint", "Loading checkpoint"),
    ("safetensors", "Loading checkpoint"),
    ("make_policy", "Initializing policy"),
    ("connecting", "Connecting hardware"),
    ("connected", "Hardware connected"),
)

_REPLAYING_TRIGGERS: tuple[str, ...] = (
    "replaying episode",
    "[lerobot] replaying",
    "running policy",
)


class ReplayOutputConsumer(OutputConsumer):
    """Parses replay output and surfaces preparation milestones."""

    async def parse_line(self, line: str) -> None:
        state = self.board.get("state")
        if state != SessionState.PREPARING:
            return

        lowered = line.lower()

        if any(kw in lowered for kw in _REPLAYING_TRIGGERS):
            await self.board.update(state=SessionState.REPLAYING, prepare_stage="")
            return

        stage = ""
        for needle, label in _PREPARE_STAGES:
            if needle in lowered:
                stage = label
        if stage and stage != self.board.get("prepare_stage"):
            await self.board.update(prepare_stage=stage)


class ReplaySession(Session):
    """Dataset replay session.

    CLI entry: replay(manifest, kwargs, tty_handoff)
    Web entry: EmbodiedService.start_replay() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return ReplayOutputConsumer(board, stdout)

    # -- CLI entry point ---------------------------------------------------

    async def replay(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        self._parent.acquire_embodiment("replaying")
        try:
            argv = CommandBuilder.replay(manifest, **self._replay_kwargs(kwargs))
            await self.start(argv)
            if tty_handoff:
                from roboclaw.embodied.toolkit.tty import TtySession

                return await TtySession(tty_handoff).run(self)
            return "Replay started."
        finally:
            self._parent.release_embodiment()

    def _replay_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in kwargs.items()
            if k in ("dataset_name", "episode", "fps", "arms")
        }

    async def _wait_process(self) -> None:
        """Release embodiment lock on natural subprocess exit (web path)."""
        await super()._wait_process()
        self._parent.release_embodiment()

    # -- CLI protocol ------------------------------------------------------

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="lerobot-replay")

    def status_line(self) -> str:
        s = self.board.state
        state = s.get("state", "idle")
        if state == SessionState.PREPARING:
            stage = s.get("prepare_stage", "")
            return f"  preparing... {stage}" if stage else "  preparing..."
        elapsed = s.get("elapsed_seconds", 0)
        return f"  replaying  | {elapsed:.0f}s"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()

    def result(self) -> str:
        s = self.board.state
        if s.get("error"):
            return f"Replay failed: {s['error']}"
        return "Replay finished."
