"""Session — subprocess lifecycle manager.

Holds Board + InputConsumer + OutputConsumer.
Manages: start, stop, drain delay, serial flush, process wait.
Provides CLI protocol methods for TtySession adapter.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from roboclaw.embodied.board import Board, EpisodePhase, InputConsumer, OutputConsumer, SessionState
from roboclaw.embodied.executor import SubprocessExecutor

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest


class Session:
    """Base class for all operation sessions.

    Lifecycle: init -> start(argv) -> running -> stop() -> idle
    """

    DRAIN_DELAY_S: float = 5.0

    def __init__(
        self,
        board: Board,
        manifest: Manifest | None = None,
    ) -> None:
        self._manifest = manifest
        self.board = board
        self._process: asyncio.subprocess.Process | None = None
        self._output_consumer: OutputConsumer | None = None
        self._input_consumer: InputConsumer | None = None
        self._wait_task: asyncio.Task | None = None
        self._runner = SubprocessExecutor()
        self._stopped = False
        self._exit_callback: Callable[["Session"], Any] | None = None

    # -- Subclass hooks ----------------------------------------------------

    def _make_output_consumer(self, board: Board, stdout: asyncio.StreamReader) -> OutputConsumer:
        """Override to provide operation-specific output parsing."""
        return OutputConsumer(board, stdout)

    def _make_input_consumer(self, board: Board, stdin: asyncio.StreamWriter) -> InputConsumer:
        """Override to provide operation-specific key mappings."""
        return InputConsumer(board, stdin)

    # -- Lifecycle ---------------------------------------------------------

    async def start(
        self, argv: list[str], *,
        initial_state: str = SessionState.PREPARING,
        auto_confirm: bool = True,
    ) -> None:
        """Start subprocess and wire consumers."""
        self._stopped = False
        owner = self.board.get("embodiment_owner", "")
        self.board.reset()
        await self.board.update(state=initial_state, embodiment_owner=owner)
        try:
            # Launch interactive subprocess (stdin piped, stderr merged into stdout)
            self._process = await self._runner.run_streaming_interactive(argv)

            # Auto-confirm calibration prompts for non-calibration sessions
            if auto_confirm and self._process.stdin:
                self._process.stdin.write(b"\n\n\n\n")
                await self._process.stdin.drain()

            # Wire consumers
            self._output_consumer = self._make_output_consumer(self.board, self._process.stdout)
            if self._process.stdin:
                self._input_consumer = self._make_input_consumer(self.board, self._process.stdin)
                self.board._input_consumer_notify = self._input_consumer._on_command_posted

            self.board.start_timer()
            await self._output_consumer.start()
            if self._input_consumer:
                await self._input_consumer.start()

            # Monitor process exit
            self._wait_task = asyncio.create_task(self._wait_process(), name="session-wait")
            logger.info("Session started pid={}: {}", self._process.pid, " ".join(argv[:5]))
        except Exception as exc:
            await self._rollback_start(exc)
            raise

    async def stop(self) -> None:
        """Graceful stop: ESC -> wait -> SIGINT -> wait -> kill."""
        if self._process is None or self._process.returncode is not None:
            await self.board.update(**self._idle_fields())
            return
        self._stopped = True
        await self._enter_stopping_state()

        if self._process.stdin:
            try:
                self._process.stdin.write(b"\x1b\n")
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        try:
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
            await self._cleanup()
            return
        except asyncio.TimeoutError:
            pass

        try:
            self._process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            await self._cleanup()
            return

        try:
            await asyncio.wait_for(self._process.wait(), timeout=3.0)
            await self._cleanup()
            return
        except asyncio.TimeoutError:
            pass

        self._process.kill()
        await self._process.wait()
        await self._cleanup()

    async def _teardown(self) -> None:
        """Stop consumers, close stdin, clear process reference."""
        if self._output_consumer:
            await self._output_consumer.stop()
            self._output_consumer = None
        if self._input_consumer:
            await self._input_consumer.stop()
            self._input_consumer = None
        self.board._input_consumer_notify = None
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        self._process = None

    async def _rollback_start(self, exc: Exception) -> None:
        """Abort a partially started session and surface a startup error."""
        process = self._process
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        await self._teardown()
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
            try:
                await self._wait_task
            except asyncio.CancelledError:
                pass
        self._wait_task = None
        await self.board.update(state=SessionState.ERROR, error=str(exc))

    async def _wait_process(self) -> None:
        """Wait for subprocess to exit naturally, update Board."""
        assert self._process is not None
        rc = await self._process.wait()
        logger.info("Session subprocess exited with code {}", rc)
        await self._teardown()

        if not self._stopped:
            if rc in (0, None, -2, 130, -15):
                await self.board.update(**self._idle_fields())
            else:
                await self.board.update(state=SessionState.ERROR, error=self._format_exit_error(rc))
            # Release embodiment lock — subprocess is dead, hardware is free
            if self._exit_callback:
                self._exit_callback(self)

    def _format_exit_error(self, rc: int) -> str:
        """Extract error info from Board logs after a non-zero exit."""
        logs = self.board.recent_logs(5)
        keywords = ("Error", "Traceback", "Exception")
        error_lines = [line for line in logs if any(kw in line for kw in keywords)]
        tail = "\n".join(error_lines[-3:]) if error_lines else "\n".join(logs[-3:])
        error = f"Process exited with code {rc}"
        if tail:
            error += f"\n{tail}"
        return error

    async def _cleanup(self) -> None:
        """Stop consumers and update board to idle."""
        await self._teardown()
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
            try:
                await self._wait_task
            except asyncio.CancelledError:
                pass
        self._wait_task = None
        await self.board.update(**self._idle_fields())

    async def _enter_stopping_state(self) -> None:
        state = self.board.get("state", SessionState.IDLE)
        if state in (SessionState.IDLE, SessionState.ERROR, SessionState.STOPPING):
            return

        fields: dict[str, Any] = {"state": SessionState.STOPPING}
        phase = self.board.get("episode_phase", "")
        if state == SessionState.RECORDING and phase not in (EpisodePhase.SAVING, EpisodePhase.RESETTING):
            fields["episode_phase"] = EpisodePhase.STOPPING
        await self.board.update(**fields)

    def _idle_fields(self) -> dict[str, Any]:
        return {
            "state": SessionState.IDLE,
            "episode_phase": "",
            "prepare_stage": "",
        }

    async def wait(self) -> None:
        """Wait for the natural-exit monitor to finish."""
        task = self._wait_task
        if task is not None:
            await asyncio.shield(task)

    @property
    def busy(self) -> bool:
        return self._process is not None and self._process.returncode is None

    # -- CLI protocol methods (TtySession adapter uses these) --------------

    def interaction_spec(self):
        """Override in subclasses."""
        from roboclaw.embodied.toolkit.protocol import PollingSpec

        return PollingSpec(label="session")

    def status_line(self) -> str:
        """Read from Board, format for terminal. Override in subclasses."""
        s = self.board.state
        return f"  {s.get('state', 'idle')}"

    async def on_key(self, key: str) -> None:
        """Map key to Board command. Override in subclasses."""
        if key in ("ctrl_c", "esc"):
            await self.stop()

    def is_done(self) -> bool:
        return self.board.state.get("state") in (SessionState.IDLE, SessionState.ERROR)

    def result(self) -> str:
        """Override in subclasses."""
        s = self.board.state
        if s.get("error"):
            return f"Failed: {s['error']}"
        return "Done."
