"""TtySession — generic terminal driver for session protocol objects.

Reads ``session.interaction_spec()`` and dispatches to the appropriate
I/O loop (passthrough, polling, or prompting).  Knows nothing about
teleop, recording, or setup — only about PollingSpec, PassthroughSpec,
and PromptingSpec.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from roboclaw.embodied.toolkit.protocol import (
    PassthroughSpec,
    PollStep,
    PollingSpec,
    PromptStep,
    PromptingSpec,
)


class TtySession:
    """Drive any session that implements the interaction protocol."""

    def __init__(self, tty_handoff: Any) -> None:
        self._tty_handoff = tty_handoff

    async def run(self, session: Any) -> str:
        spec = session.interaction_spec()
        if isinstance(spec, PassthroughSpec):
            return await self._run_passthrough(session, spec)
        if isinstance(spec, PollingSpec):
            return await self._run_polling(session, spec)
        if isinstance(spec, PromptingSpec):
            return await self._run_prompting(session, spec)
        raise ValueError(f"Unknown interaction spec: {type(spec)}")

    # -- Passthrough -----------------------------------------------------------

    async def _run_passthrough(self, session: Any, spec: PassthroughSpec) -> str:
        from roboclaw.embodied.executor import SubprocessExecutor

        await self._tty_handoff(start=True, label=spec.label)
        try:
            runner = SubprocessExecutor()
            rc, stderr_text = await runner.run_interactive(spec.argv)
            session.set_exit_code(rc, stderr_text)
            return session.result()
        finally:
            await self._tty_handoff(start=False, label=spec.label)

    # -- Polling ---------------------------------------------------------------

    async def _run_polling(self, session: Any, spec: PollingSpec) -> str:
        from roboclaw.embodied.toolkit.terminal import raw_terminal, read_key_nonblocking

        await self._tty_handoff(start=True, label=spec.label)
        try:
            await session.start()
            with raw_terminal():
                while not session.is_done():
                    key = read_key_nonblocking()
                    if key:
                        await session.on_key(key)
                    print(f"\r{session.status_line()}", end="", flush=True)
                    await asyncio.sleep(spec.poll_interval_s)
            print()  # newline after status line
            return session.result()
        finally:
            try:
                await session.stop()
            finally:
                await self._tty_handoff(start=False, label=spec.label)

    # -- Prompting -------------------------------------------------------------

    async def _run_prompting(self, session: Any, spec: PromptingSpec) -> str:
        await self._tty_handoff(start=True, label=spec.label)
        try:
            result = await asyncio.to_thread(self._prompting_loop, session)
            return result
        finally:
            await self._tty_handoff(start=False, label=spec.label)

    @staticmethod
    def _prompting_loop(session: Any) -> str:
        def _flush() -> None:
            for msg in session.drain_messages():
                print(msg)

        while True:
            step = session.next_step()
            _flush()
            if step is None:
                return session.result()
            if isinstance(step, PollStep):
                answer = _run_poll_step(step)
                if answer is None:
                    retry = input(step.retry_prompt).strip().lower()
                    if retry != "y":
                        if hasattr(session, 'on_timeout'):
                            session.on_timeout()
                        return session.result()
                    continue
                session.submit_answer(step.prompt_id, answer)
                _flush()
            elif isinstance(step, PromptStep):
                answer = _run_prompt_step(step)
                session.submit_answer(step.prompt_id, answer)
                _flush()


def _run_prompt_step(step: PromptStep) -> str:
    """Display a prompt and return user input."""
    if step.options:
        for i, opt in enumerate(step.options, 1):
            print(f"  [{i}] {opt}")
        while True:
            raw = input(step.message + " ").strip()
            try:
                idx = int(raw)
                if 1 <= idx <= len(step.options):
                    return raw
            except ValueError:
                pass
            print(f"  Enter 1-{len(step.options)}")
    return input(step.message + " ").strip()


def _run_poll_step(step: PollStep) -> str | None:
    """Poll until poll_fn returns a value or timeout expires."""
    print(step.message)
    start = time.monotonic()
    while time.monotonic() - start < step.timeout_s:
        result = step.poll_fn()
        if result is not None:
            return result
        time.sleep(0.1)
    print(step.timeout_message)
    return None
