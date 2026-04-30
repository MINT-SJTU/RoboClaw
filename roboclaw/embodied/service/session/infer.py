"""InferSession — trained policy rollout execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from roboclaw.embodied.board import Board, OutputConsumer, SessionState
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


# Generic preparation milestones — later matches win so we always show
# the most recent stage. Keep entries model-agnostic; policy-specific
# keywords (pi05, openpi, etc.) don't belong here.
_PREPARE_STAGES: tuple[tuple[str, str], ...] = (
    ("loading checkpoint", "Loading checkpoint"),
    ("safetensors", "Loading checkpoint"),
    ("make_policy", "Initializing policy"),
    ("connecting", "Connecting hardware"),
    ("connected", "Hardware connected"),
)

_INFERRING_TRIGGERS: tuple[str, ...] = (
    "running policy",
    "recording episode",
    "[lerobot] recording",
)


class InferOutputConsumer(OutputConsumer):
    """Parses policy inference output and surfaces preparation milestones."""

    async def parse_line(self, line: str) -> None:
        state = self.board.get("state")
        if state != SessionState.PREPARING:
            return

        lowered = line.lower()

        if any(kw in lowered for kw in _INFERRING_TRIGGERS):
            await self.board.update(state=SessionState.INFERRING, prepare_stage="")
            return

        stage = ""
        for needle, label in _PREPARE_STAGES:
            if needle in lowered:
                stage = label
        if stage and stage != self.board.get("prepare_stage"):
            await self.board.update(prepare_stage=stage)


class InferSession(Session):
    """Policy inference session.

    Web entry: EmbodiedService.start_inference() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return InferOutputConsumer(board, stdout)

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
