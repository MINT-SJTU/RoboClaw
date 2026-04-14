"""Board — per-embodiment unified state + pub/sub hub (看板).

Writers: OutputConsumer writes parsed subprocess output.
Readers: Agent and Frontend read current state and logs.
Commands: Agent and Frontend post commands (save, discard, stop).
Pub/Sub: Any component can subscribe to channels; Board broadcasts.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections import deque
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.embodied.board.channels import CH_SESSION
from roboclaw.embodied.board.constants import SessionState

IDLE_STATE: dict[str, Any] = {
    "state": SessionState.IDLE,
    "episode_phase": "",
    "saved_episodes": 0,
    "current_episode": 0,
    "target_episodes": 0,
    "total_frames": 0,
    "elapsed_seconds": 0.0,
    "dataset": None,
    "rerun_web_port": 0,
    "error": "",
    "embodiment_owner": "",
    "prepare_stage": "",
}

Subscriber = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class Board:
    def __init__(self, max_log_lines: int | None = None) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = dict(IDLE_STATE)
        self._commands: deque[str] = deque()
        self._max_log_lines = max_log_lines
        self._log: list[str] = []
        self._start_time: float = 0.0
        self._input_consumer_notify: Any = None
        self._subscribers: dict[str | None, list[Subscriber]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Pub/Sub ──

    def on(self, channel: str | None, handler: Subscriber) -> None:
        """Subscribe *handler* to *channel* (``None`` = all channels)."""
        with self._lock:
            self._subscribers.setdefault(channel, []).append(handler)

    def off(self, channel: str | None, handler: Subscriber) -> None:
        """Remove *handler* from *channel*."""
        with self._lock:
            handlers = self._subscribers.get(channel, [])
            if handler in handlers:
                handlers.remove(handler)

    async def emit(self, channel: str, data: dict[str, Any]) -> None:
        """Publish *data* to *channel*. Notifies matching + wildcard subscribers."""
        # Board is constructed before the event loop starts; capture on first async call.
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        with self._lock:
            handlers = [
                *self._subscribers.get(channel, []),
                *self._subscribers.get(None, []),
            ]
        for handler in handlers:
            try:
                result = handler(channel, data)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Board subscriber error on channel '{}'", channel)

    def emit_sync(self, channel: str, data: dict[str, Any]) -> None:
        """Fire-and-forget emit for synchronous contexts (e.g. Manifest).

        Safe to call from any thread. Uses the captured event loop
        reference to schedule the async emit.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(channel, data))
            return
        except RuntimeError:
            pass
        loop = self._loop
        if loop is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.emit(channel, data), loop)
        else:
            logger.debug("emit_sync(): no event loop bound, event on '{}' dropped", channel)

    # ── State (OutputConsumer writes, Agent/Frontend reads) ──

    @property
    def state(self) -> dict[str, Any]:
        """Thread-safe state snapshot."""
        with self._lock:
            s = dict(self._state)
            if self._start_time and s["state"] not in (SessionState.IDLE, SessionState.ERROR):
                s["elapsed_seconds"] = round(time.monotonic() - self._start_time, 1)
            return s

    async def update(self, **fields: Any) -> None:
        """Update state fields and emit to CH_SESSION.

        Only emits if at least one field actually changed.
        """
        with self._lock:
            changed = any(self._state.get(k) != v for k, v in fields.items())
            if not changed:
                return
            self._state.update(fields)
            snapshot = dict(self._state)
            if self._start_time and snapshot["state"] not in (SessionState.IDLE, SessionState.ERROR):
                snapshot["elapsed_seconds"] = round(time.monotonic() - self._start_time, 1)
        snapshot["timestamp"] = time.time()
        await self.emit(CH_SESSION, snapshot)

    def start_timer(self) -> None:
        with self._lock:
            self._start_time = time.monotonic()

    def reset(self) -> None:
        """Reset session state to idle. Preserves subscriptions."""
        with self._lock:
            self._state = dict(IDLE_STATE)
            self._commands.clear()
            self._start_time = 0.0

    def set_field(self, key: str, value: Any) -> None:
        """Set a single state field without emitting. Thread-safe."""
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Read a single state field without full snapshot copy."""
        with self._lock:
            return self._state.get(key, default)

    # ── Commands (Agent/Frontend writes, InputConsumer reads) ──

    def post_command(self, command: str) -> None:
        """Post a command. Thread-safe. Called by Agent/Frontend."""
        with self._lock:
            self._commands.append(command)
        if self._input_consumer_notify:
            self._input_consumer_notify()

    def poll_command(self) -> str | None:
        """Take one command. Called by InputConsumer."""
        with self._lock:
            return self._commands.popleft() if self._commands else None

    # ── Logs (OutputConsumer writes, Agent/Frontend reads) ──

    def log(self, line: str) -> None:
        with self._lock:
            self._log.append(line)
            if self._max_log_lines is not None and len(self._log) > self._max_log_lines:
                overflow = len(self._log) - self._max_log_lines
                if overflow > 0:
                    del self._log[:overflow]

    def recent_logs(self, n: int = 20) -> list[str]:
        with self._lock:
            return self._log[-n:]

    def all_logs(self) -> list[str]:
        with self._lock:
            return list(self._log)

    def clear_logs(self) -> None:
        with self._lock:
            self._log.clear()
