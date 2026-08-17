from __future__ import annotations

import asyncio

import pytest

from roboclaw.embodied.board import Board, SessionState
from roboclaw.embodied.service.session.base import Session


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = FakeStdin()
        self.stdout = None
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.killed = False
        self._done = asyncio.Event()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def finish(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._done.set()

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def kill(self) -> None:
        self.killed = True
        self.finish(-9)


@pytest.mark.asyncio
async def test_stop_marks_board_stopping_before_process_exits() -> None:
    session = Session(Board())
    process = FakeProcess()
    session._process = process
    await session.board.update(state=SessionState.TELEOPERATING)

    stop_task = asyncio.create_task(session.stop())
    await asyncio.sleep(0)

    assert session.board.state["state"] == SessionState.STOPPING
    assert process.stdin.writes == [b"\x1b\n"]

    process.finish()
    await asyncio.wait_for(stop_task, timeout=1)

    state = session.board.state
    assert state["state"] == SessionState.IDLE
    assert state["record_phase"] == "idle"
    assert state["record_pending_command"] == ""
    assert state["prepare_stage"] == ""


@pytest.mark.asyncio
async def test_stop_preserves_record_phase_until_cleanup() -> None:
    session = Session(Board())
    process = FakeProcess()
    session._process = process
    await session.board.update(
        state=SessionState.RECORDING,
        record_phase="recording",
        prepare_stage="warmup",
    )

    stop_task = asyncio.create_task(session.stop())
    await asyncio.sleep(0)

    state = session.board.state
    assert state["state"] == SessionState.STOPPING
    assert state["record_phase"] == "recording"
    assert state["prepare_stage"] == "warmup"

    process.finish()
    await asyncio.wait_for(stop_task, timeout=1)

    idle_state = session.board.state
    assert idle_state["state"] == SessionState.IDLE
    assert idle_state["record_phase"] == "idle"
    assert idle_state["record_pending_command"] == ""
    assert idle_state["prepare_stage"] == ""


@pytest.mark.asyncio
async def test_stop_preserves_requested_record_phase_until_cleanup() -> None:
    session = Session(Board())
    process = FakeProcess()
    session._process = process
    await session.board.update(
        state=SessionState.RECORDING,
        record_phase="save_requested",
    )

    stop_task = asyncio.create_task(session.stop())
    await asyncio.sleep(0)

    state = session.board.state
    assert state["state"] == SessionState.STOPPING
    assert state["record_phase"] == "save_requested"

    process.finish()
    await asyncio.wait_for(stop_task, timeout=1)
