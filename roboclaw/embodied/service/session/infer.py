"""InferSession — trained policy rollout execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board import Board, OutputConsumer, SessionState
from roboclaw.embodied.command import CommandBuilder
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class InferOutputConsumer(OutputConsumer):
    """Parses policy inference output."""

    async def parse_line(self, line: str) -> None:
        if self.board.get("state") != SessionState.PREPARING:
            return
        if any(kw in line.lower() for kw in ("running policy", "recording episode", "connected")):
            await self.board.update(state=SessionState.INFERRING)


class InferSession(Session):
    """Policy inference session.

    CLI entry: run_policy(manifest, kwargs, tty_handoff)
    Web entry: EmbodiedService.start_inference() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return InferOutputConsumer(board, stdout)

    # ── CLI entry point ──────────────────────────────────────────────────

    async def run_policy(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        argv = CommandBuilder.infer(manifest, **_filter_infer_kwargs(kwargs))
        await self.start(argv, initial_state=SessionState.INFERRING)
        if tty_handoff:
            from roboclaw.embodied.toolkit.tty import TtySession

            return await TtySession(tty_handoff).run(self)
        return "Inference started."

    # ── CLI protocol ─────────────────────────────────────────────────────

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="lerobot-infer")

    def status_line(self) -> str:
        s = self.board.state
        state = s.get("state", "idle")
        if state == SessionState.PREPARING:
            return "  preparing..."
        elapsed = s.get("elapsed_seconds", 0)
        return f"  inferring  | {elapsed:.0f}s"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()

    def result(self) -> str:
        s = self.board.state
        if s.get("error"):
            return f"Inference failed: {s['error']}"
        return "Inference finished."


# ── Private helpers ──────────────────────────────────────────────────────

_INFER_KEYS = frozenset({
    "checkpoint_path", "source_dataset", "dataset_name",
    "task", "num_episodes", "arms", "use_cameras",
})


def _filter_infer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract only the kwargs that CommandBuilder.infer accepts."""
    return {k: v for k, v in kwargs.items() if k in _INFER_KEYS}
