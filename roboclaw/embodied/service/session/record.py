"""RecordSession — dataset recording with episode lifecycle tracking."""

from __future__ import annotations

import asyncio
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board import Board, Command, InputConsumer, OutputConsumer, SessionState
from roboclaw.embodied.command import CommandBuilder
from roboclaw.embodied.service.session.base import Session

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService

_RE_RECORDING_EP = re.compile(r"Recording episode (\d+)")


class RecordPhase(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    SAVE_REQUESTED = "save_requested"
    DISCARD_REQUESTED = "discard_requested"
    RESETTING = "resetting"
    SKIP_RESET_REQUESTED = "skip_reset_requested"
    STOPPING = "stopping"
    ERROR = "error"


class RecordPhaseController:
    """Owns record-specific phase transitions and command validation."""

    def __init__(self, board: Board) -> None:
        self.board = board
        self._phase_before_stop: RecordPhase | None = None

    @property
    def phase(self) -> RecordPhase:
        value = self.board.get("record_phase", RecordPhase.IDLE)
        return RecordPhase(value or RecordPhase.IDLE)

    @property
    def pending_command(self) -> str:
        return self.board.get("record_pending_command", "")

    async def request_save(self) -> None:
        self._require_no_pending()
        self._require_phase(Command.SAVE_EPISODE, RecordPhase.RECORDING)
        await self._set(RecordPhase.SAVE_REQUESTED, Command.SAVE_EPISODE)

    async def request_discard(self) -> None:
        self._require_no_pending()
        self._require_phase(Command.DISCARD_EPISODE, RecordPhase.RECORDING)
        await self._set(RecordPhase.DISCARD_REQUESTED, Command.DISCARD_EPISODE)

    async def request_skip_reset(self) -> None:
        self._require_no_pending()
        self._require_phase(Command.SKIP_RESET, RecordPhase.RESETTING)
        await self._set(RecordPhase.SKIP_RESET_REQUESTED, Command.SKIP_RESET)

    async def request_stop(self) -> None:
        self._phase_before_stop = self.phase
        await self._set(RecordPhase.STOPPING, "")

    async def observe_recording_episode(self, episode: int) -> None:
        saved = self.board.get("saved_episodes", 0)
        self._phase_before_stop = None
        if self.phase in (
            RecordPhase.SAVE_REQUESTED,
            RecordPhase.RESETTING,
            RecordPhase.SKIP_RESET_REQUESTED,
        ):
            saved += 1
        await self.board.update(
            state=SessionState.RECORDING,
            current_episode=episode,
            saved_episodes=saved,
            record_phase=RecordPhase.RECORDING,
            record_pending_command="",
        )

    async def observe_reset_prompt(self) -> None:
        await self._set(RecordPhase.RESETTING, "")

    async def observe_save_key_ack(self) -> None:
        if self.phase in (RecordPhase.RECORDING, RecordPhase.SAVE_REQUESTED):
            await self._set(RecordPhase.SAVE_REQUESTED, "")

    async def observe_discard_key_ack(self) -> None:
        if self.phase in (RecordPhase.RECORDING, RecordPhase.DISCARD_REQUESTED):
            await self._set(RecordPhase.DISCARD_REQUESTED, "")

    async def observe_skip_reset_key_ack(self) -> None:
        if self.phase in (RecordPhase.RESETTING, RecordPhase.SKIP_RESET_REQUESTED):
            await self._set(RecordPhase.SKIP_RESET_REQUESTED, "")

    async def observe_rerecord(self) -> None:
        self._phase_before_stop = None
        await self._set(RecordPhase.RECORDING, "")

    async def observe_stop(self) -> None:
        saved = self.board.get("saved_episodes", 0)
        previous_phase = self._phase_before_stop or self.phase
        if previous_phase in (
            RecordPhase.SAVE_REQUESTED,
            RecordPhase.RESETTING,
            RecordPhase.SKIP_RESET_REQUESTED,
        ):
            saved += 1
        self._phase_before_stop = None
        await self.board.update(
            saved_episodes=saved,
            record_phase=RecordPhase.STOPPING,
            record_pending_command="",
        )

    async def _set(self, phase: RecordPhase, pending_command: str | Command) -> None:
        await self.board.update(
            record_phase=phase,
            record_pending_command=str(pending_command),
        )

    def _require_no_pending(self) -> None:
        if self.pending_command:
            raise RuntimeError(
                f"Cannot send record command while waiting for {self.pending_command}."
            )

    def _require_phase(self, command: Command, *allowed: RecordPhase) -> None:
        if self.phase not in allowed:
            expected = ", ".join(phase.value for phase in allowed)
            raise RuntimeError(
                f"Cannot {command.value} while record phase is {self.phase.value}; expected {expected}."
            )


class RecordInputConsumer(InputConsumer):
    """Record-specific command bytes for LeRobot's control listener."""

    _RECORD_KEYMAP: dict[str, bytes] = {
        Command.SAVE_EPISODE: b"\x1b[C",
        Command.DISCARD_EPISODE: b"\x1b[D",
        Command.SKIP_RESET: b"p",
        Command.STOP: b"\x1b",
        Command.CONFIRM: b"\n",
    }

    def translate(self, command: str) -> bytes | None:
        return self._RECORD_KEYMAP.get(command)


class RecordOutputConsumer(OutputConsumer):
    """Parses lerobot record stdout for episode lifecycle."""

    def __init__(
        self,
        board: Board,
        stdout: asyncio.StreamReader,
        phase: RecordPhaseController | None = None,
    ) -> None:
        super().__init__(board, stdout)
        self._phase = phase or RecordPhaseController(board)

    async def parse_line(self, line: str) -> None:
        m = _RE_RECORDING_EP.search(line)
        if m:
            await self._phase.observe_recording_episode(int(m.group(1)))
            return

        if "Right arrow key pressed" in line:
            await self._phase.observe_save_key_ack()
            return

        if "P key pressed" in line or "Skipping reset" in line:
            await self._phase.observe_skip_reset_key_ack()
            return

        if "Reset the environment" in line:
            await self._phase.observe_reset_prompt()
            return

        if "Re-record" in line or "Left arrow key pressed" in line:
            if "Left arrow key pressed" in line:
                await self._phase.observe_discard_key_ack()
                return
            await self._phase.observe_rerecord()
            return

        if "Stop recording" in line or "Stopping data recording" in line:
            await self._phase.observe_stop()
            return


class RecordSession(Session):
    """Dataset recording session.

    CLI entry: record(manifest, kwargs, tty_handoff)
    Web entry: EmbodiedService.start_recording() -> start(argv)
    """

    def __init__(self, parent: EmbodiedService) -> None:
        super().__init__(board=parent.board, manifest=parent.manifest)
        self._parent = parent
        self._phase = RecordPhaseController(self.board)
        self._kwargs: dict[str, Any] = {}
        self._dataset_name: str = ""

    def _initial_board_fields(self) -> dict[str, Any]:
        return {
            "record_phase": RecordPhase.PREPARING,
            "record_pending_command": "",
        }

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        return RecordOutputConsumer(board, stdout, self._phase)

    def _make_input_consumer(self, board: Board, stdin: asyncio.StreamWriter) -> InputConsumer:
        return RecordInputConsumer(board, stdin)

    async def request_save_episode(self) -> None:
        await self._phase.request_save()
        self.board.post_command(Command.SAVE_EPISODE)

    async def request_discard_episode(self) -> None:
        await self._phase.request_discard()
        self.board.post_command(Command.DISCARD_EPISODE)

    async def request_skip_reset(self) -> None:
        await self._phase.request_skip_reset()
        self.board.post_command(Command.SKIP_RESET)

    # -- CLI entry point ---------------------------------------------------

    async def record(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        self._kwargs = kwargs
        if tty_handoff:
            self._parent.acquire_embodiment("recording")
            try:
                dataset = self._parent.datasets.prepare_recording_dataset(
                    kwargs.get("dataset_name", ""),
                    prefix="rec",
                )
                argv = CommandBuilder.record(
                    manifest,
                    dataset=dataset.runtime,
                    **self._record_kwargs(kwargs),
                )
                self._dataset_name = dataset.runtime.name
                await self.start(argv)
                await self.board.update(
                    target_episodes=kwargs.get("num_episodes", 10),
                    dataset=self._dataset_name,
                )
                from roboclaw.embodied.toolkit.tty import TtySession

                return await TtySession(tty_handoff).run(self)
            finally:
                self._parent.release_embodiment()
        return "This action requires a local terminal."

    def _record_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract CommandBuilder.record() keyword args from raw kwargs."""
        return {
            k: v
            for k, v in kwargs.items()
            if k in ("task", "num_episodes", "fps",
                     "episode_time_s", "reset_time_s", "arms", "use_cameras")
        }

    async def _wait_process(self) -> None:
        """Release embodiment lock on natural subprocess exit (web path)."""
        await super()._wait_process()
        if self.board.get("state") == SessionState.ERROR:
            await self.board.update(
                record_phase=RecordPhase.ERROR,
                record_pending_command="",
            )
        self._parent.release_embodiment()

    # -- CLI protocol ------------------------------------------------------

    def interaction_spec(self):
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="lerobot-record")

    def status_line(self) -> str:
        s = self.board.state
        state = s.get("state", "idle")
        if state in (SessionState.IDLE, SessionState.PREPARING):
            return f"  {state}..."
        phase = s.get("record_phase", "")
        current = s.get("current_episode", 0)
        target = s.get("target_episodes", 0)
        saved = s.get("saved_episodes", 0)
        return f"  Episode {current}/{target} | Saved: {saved} | {phase or state}"

    async def on_key(self, key: str) -> None:
        if key in ("ctrl_c", "esc"):
            await self.stop()
        elif key == "right":
            if self._phase.phase == RecordPhase.RESETTING:
                await self.request_skip_reset()
            else:
                await self.request_save_episode()
        elif key == "left":
            await self.request_discard_episode()

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
            await self._phase.request_stop()
            await super().stop()
