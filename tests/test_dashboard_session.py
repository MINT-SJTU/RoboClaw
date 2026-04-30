"""Tests for current session consumers and CLI adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from roboclaw.embodied.board import Board, Command, SessionState
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.helpers import save_manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.service.session.infer import InferOutputConsumer
from roboclaw.embodied.service.session.record import (
    RecordOutputConsumer,
    RecordPhase,
    RecordPhaseController,
)
from roboclaw.embodied.service.session.replay import ReplayOutputConsumer


def _service(tmp_path: Path) -> EmbodiedService:
    manifest_path = tmp_path / "manifest.json"
    save_manifest(
        {
            "version": 2,
            "arms": [],
            "hands": [],
            "cameras": [],
            "datasets": {"root": "/data"},
            "policies": {"root": "/policies"},
        },
        manifest_path,
    )
    return EmbodiedService(manifest=Manifest(path=manifest_path))


def test_record_output_consumer_tracks_episode_lifecycle() -> None:
    board = Board()
    consumer = RecordOutputConsumer(board, stdout=None)

    async def _run() -> None:
        await board.update(state=SessionState.PREPARING)
        await consumer.parse_line("[lerobot] Recording episode 0")
        await consumer.parse_line("Right arrow key pressed.")
        await consumer.parse_line("[lerobot] Reset the environment")
        await consumer.parse_line("[lerobot] Recording episode 1")
        await consumer.parse_line("Stopping data recording")

    asyncio.run(_run())

    state = board.state
    assert state["state"] == SessionState.RECORDING
    assert state["current_episode"] == 1
    assert state["saved_episodes"] == 1
    assert state["record_phase"] == RecordPhase.STOPPING


def test_record_output_consumer_rerecord_restores_recording_phase() -> None:
    board = Board()
    consumer = RecordOutputConsumer(board, stdout=None)

    async def _run() -> None:
        await board.update(state=SessionState.RECORDING, record_phase=RecordPhase.DISCARD_REQUESTED)
        await consumer.parse_line("[lerobot] Re-record episode")

    asyncio.run(_run())
    assert board.state["record_phase"] == RecordPhase.RECORDING


def test_record_session_status_line_and_keys(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.record
    service.board.set_field("state", SessionState.RECORDING)
    service.board.set_field("current_episode", 3)
    service.board.set_field("target_episodes", 10)
    service.board.set_field("saved_episodes", 2)
    service.board.set_field("record_phase", RecordPhase.RESETTING)

    assert session.status_line() == "  Episode 3/10 | Saved: 2 | resetting"

    asyncio.run(session.on_key("right"))
    assert service.board.poll_command() == Command.SKIP_RESET
    service.board.set_field("record_phase", RecordPhase.RECORDING)
    service.board.set_field("record_pending_command", "")
    asyncio.run(session.on_key("right"))
    assert service.board.poll_command() == Command.SAVE_EPISODE
    service.board.set_field("record_phase", RecordPhase.RECORDING)
    service.board.set_field("record_pending_command", "")
    asyncio.run(session.on_key("left"))
    assert service.board.poll_command() == Command.DISCARD_EPISODE


def test_record_phase_controller_rejects_invalid_commands() -> None:
    board = Board()
    controller = RecordPhaseController(board)

    async def _run() -> None:
        await board.update(record_phase=RecordPhase.RESETTING)
        with pytest.raises(RuntimeError, match="save_episode"):
            await controller.request_save()
        await controller.request_skip_reset()
        with pytest.raises(RuntimeError, match="waiting for skip_reset"):
            await controller.request_skip_reset()

    asyncio.run(_run())


def test_record_phase_controller_stop_keeps_output_verified_save_count() -> None:
    board = Board()
    controller = RecordPhaseController(board)

    async def _run() -> None:
        await board.update(
            record_phase=RecordPhase.RESETTING,
            saved_episodes=2,
        )
        await controller.request_stop()
        assert board.state["record_phase"] == RecordPhase.STOPPING
        await controller.observe_stop()
        assert board.state["saved_episodes"] == 3

    asyncio.run(_run())


def test_replay_output_consumer_updates_prepare_stage_and_state() -> None:
    board = Board()
    consumer = ReplayOutputConsumer(board, stdout=None)

    async def _run() -> None:
        await board.update(state=SessionState.PREPARING)
        await consumer.parse_line("loading checkpoint")
        await consumer.parse_line("[lerobot] replaying episode 2")

    asyncio.run(_run())

    state = board.state
    assert state["prepare_stage"] == ""
    assert state["state"] == SessionState.REPLAYING


def test_infer_output_consumer_updates_prepare_stage_and_state() -> None:
    board = Board()
    consumer = InferOutputConsumer(board, stdout=None)

    async def _run() -> None:
        await board.update(state=SessionState.PREPARING)
        await consumer.parse_line("make_policy")
        await consumer.parse_line("running policy")

    asyncio.run(_run())

    state = board.state
    assert state["prepare_stage"] == ""
    assert state["state"] == SessionState.INFERRING


def test_record_result_uses_dataset_and_error(tmp_path: Path) -> None:
    service = _service(tmp_path)
    session = service.record

    service.board.set_field("saved_episodes", 4)
    service.board.set_field("dataset", "demo_set")
    assert session.result() == "Recording finished. 4 episodes saved to demo_set."

    service.board.set_field("error", "serial timeout")
    assert session.result() == "Recording failed: serial timeout"
