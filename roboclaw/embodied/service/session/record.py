"""RecordSession — dataset recording with episode lifecycle tracking."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board import Board, Command, EpisodePhase, OutputConsumer, SessionState
from roboclaw.embodied.command import CommandBuilder, validate_dataset_name
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService

_RE_RECORDING_EP = re.compile(r"Recording episode (\d+)")


class RecordOutputConsumer(OutputConsumer):
    """Parses lerobot record stdout for episode lifecycle."""

    async def parse_line(self, line: str) -> None:
        if self.board.get("state") == SessionState.PREPARING:
            await self.board.update(state=SessionState.RECORDING)

        m = _RE_RECORDING_EP.search(line)
        if m:
            ep = int(m.group(1))
            phase = self.board.get("episode_phase", "")
            saved = self.board.get("saved_episodes", 0)
            if phase in (EpisodePhase.SAVING, EpisodePhase.RESETTING):
                saved += 1
            await self.board.update(
                current_episode=ep,
                saved_episodes=saved,
                episode_phase=EpisodePhase.RECORDING,
            )
            return

        if "Right arrow key pressed" in line:
            await self.board.update(episode_phase=EpisodePhase.SAVING)
            return

        if "Reset the environment" in line:
            await self.board.update(episode_phase=EpisodePhase.RESETTING)
            return

        if "Re-record" in line or "Left arrow key pressed" in line:
            await self.board.update(episode_phase=EpisodePhase.RECORDING)
            return

        if "Stop recording" in line or "Stopping data recording" in line:
            phase = self.board.get("episode_phase", "")
            saved = self.board.get("saved_episodes", 0)
            if phase in (EpisodePhase.SAVING, EpisodePhase.RESETTING):
                saved += 1
            await self.board.update(
                saved_episodes=saved,
                episode_phase=EpisodePhase.STOPPING,
            )
            return


class RecordSession(Session):
    """Dataset recording session.

    CLI entry: record(manifest, kwargs, tty_handoff)
    Web entry: EmbodiedService.start_recording() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent
        self._kwargs: dict[str, Any] = {}
        self._dataset_name: str = ""

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return RecordOutputConsumer(board, stdout)

    # -- CLI entry point ---------------------------------------------------

    async def record(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        dataset_name = kwargs.get("dataset_name")
        if dataset_name:
            validate_dataset_name(dataset_name)
        self._kwargs = kwargs
        if tty_handoff:
            argv, self._dataset_name = CommandBuilder.record(manifest, **self._record_kwargs(kwargs))
            await self.board.update(
                target_episodes=kwargs.get("num_episodes", 10),
                dataset=self._dataset_name,
            )
            await self.start(argv)
            from roboclaw.embodied.toolkit.tty import TtySession

            return await TtySession(tty_handoff).run(self)
        return "This action requires a local terminal."

    def _record_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract CommandBuilder.record() keyword args from raw kwargs."""
        return {
            k: v
            for k, v in kwargs.items()
            if k in ("task", "dataset_name", "num_episodes", "fps",
                     "episode_time_s", "reset_time_s", "arms", "use_cameras")
        }

    # -- CLI protocol ------------------------------------------------------

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="lerobot-record")

    def status_line(self) -> str:
        s = self.board.state
        state = s.get("state", "idle")
        if state in (SessionState.IDLE, SessionState.PREPARING):
            return f"  {state}..."
        phase = s.get("episode_phase", "")
        current = s.get("current_episode", 0)
        target = s.get("target_episodes", 0)
        saved = s.get("saved_episodes", 0)
        return f"  Episode {current}/{target} | Saved: {saved} | {phase or state}"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()
        elif key == "right":
            self.board.post_command(Command.SAVE_EPISODE)
        elif key == "left":
            self.board.post_command(Command.DISCARD_EPISODE)

    def result(self) -> str:
        s = self.board.state
        error = s.get("error", "")
        if error:
            return f"Recording failed: {error}"
        saved = s.get("saved_episodes", 0)
        dataset = s.get("dataset")
        if dataset:
            return f"Recording finished. {saved} episodes saved to {dataset}."
        return f"Recording finished. {saved} episodes saved."

    async def stop(self) -> None:
        if self._parent.busy:
            await super().stop()
