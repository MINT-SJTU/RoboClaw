"""Regression tests for dashboard recording serial-control races."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from roboclaw.embodied.service import EmbodiedService


def test_lerobot_record_loop_exit_events_are_phase_specific() -> None:
    record_module = pytest.importorskip("lerobot.scripts.lerobot_record")
    if not hasattr(record_module, "_consume_loop_exit"):
        pytest.skip("installed lerobot package does not expose local record-loop helper")

    episode_events = {
        "exit_early": True,
        "rerecord_episode": False,
        "stop_recording": False,
        "skip_reset": False,
    }
    assert record_module._consume_loop_exit(episode_events, is_reset_loop=False) is True
    assert episode_events["exit_early"] is False

    reset_right_events = {
        "exit_early": True,
        "rerecord_episode": False,
        "stop_recording": False,
        "skip_reset": False,
    }
    assert record_module._consume_loop_exit(reset_right_events, is_reset_loop=True) is False
    assert reset_right_events["exit_early"] is False

    reset_skip_events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "skip_reset": True,
    }
    assert record_module._consume_loop_exit(reset_skip_events, is_reset_loop=True) is True
    assert reset_skip_events["skip_reset"] is False

    stale_skip_events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "skip_reset": True,
    }
    assert record_module._consume_loop_exit(stale_skip_events, is_reset_loop=False) is False
    assert stale_skip_events["skip_reset"] is False


def test_servo_positions_are_blocked_while_operation_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EmbodiedService()
    service._active_operation = SimpleNamespace(busy=True)

    def fail_if_called() -> bool:
        raise AssertionError("servo polling must not touch the hardware lock while busy")

    monkeypatch.setattr(service._file_lock, "try_shared", fail_if_called)

    assert service.read_servo_positions() == {"error": "busy", "arms": {}}


def test_servo_positions_hold_service_lock_while_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = EmbodiedService()
    calls = []

    monkeypatch.setattr(service._file_lock, "try_shared", lambda: True)
    monkeypatch.setattr(service._file_lock, "release_shared", lambda: calls.append("release"))

    def fake_read_servo_positions(arms):
        assert service._lock.locked()
        calls.append("read")
        return {"error": None, "arms": {}}

    import roboclaw.embodied.embodiment.hardware.motors as motors

    monkeypatch.setattr(motors, "read_servo_positions", fake_read_servo_positions)

    assert service.read_servo_positions() == {"error": None, "arms": {}}
    assert calls == ["read", "release"]
