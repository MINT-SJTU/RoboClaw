"""Tests for arm identification flow."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.helpers import save_manifest
from roboclaw.embodied.toolkit.tools import create_embodied_tools, EmbodiedToolGroup


_MOCK_SETUP = {
    "version": 2,
    "arms": [],
    "cameras": [],
    "datasets": {"root": "/data"},
    "policies": {"root": "/policies"},
}


def _hw_tool(tty_handoff=None) -> EmbodiedToolGroup:
    tools = create_embodied_tools(tty_handoff=tty_handoff)
    return next(t for t in tools if t.name == "setup")


def _manifest_from_data(tmp_path: Path, data: dict) -> Manifest:
    path = tmp_path / "manifest.json"
    save_manifest(data, path)
    return Manifest(path=path)


@pytest.mark.asyncio
async def test_identify_no_tty_returns_json(tmp_path: Path) -> None:
    """Identify without TTY should return JSON for conversational agents."""
    tool = _hw_tool()  # no tty_handoff
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    result = await tool.execute(action="identify")
    # Conversational mode returns JSON with model options
    assert "so101" in result or "koch" in result


@pytest.mark.asyncio
async def test_identify_with_tty_uses_tty_session(tmp_path: Path) -> None:
    """Identify with TTY should delegate to TtySession."""
    mock_handoff = AsyncMock()
    tool = _hw_tool(tty_handoff=mock_handoff)
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    async def fake_tty_run(self, session):
        return "Setup complete. 2 binding(s) committed to manifest."

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(action="identify")
    assert "Setup complete" in result


# ── Unit tests for hardware module helpers ─────────────────────────────

from roboclaw.embodied.embodiment.hardware.motion import (
    detect_motion,
    keep_strongest_motion,
    resolve_active_motion,
)
from roboclaw.embodied.embodiment.interface.serial import SerialInterface


def test_detect_motion_above_threshold() -> None:
    baseline = {1: 100, 2: 200, 3: 300}
    current = {1: 130, 2: 230, 3: 330}
    assert detect_motion(baseline, current) == 90


def test_detect_motion_below_threshold() -> None:
    baseline = {1: 100, 2: 200}
    current = {1: 110, 2: 205}
    delta = detect_motion(baseline, current)
    assert delta == 15
    assert delta < 50  # below default threshold


def test_detect_motion_missing_ids() -> None:
    """Missing motor IDs in current should be skipped."""
    baseline = {1: 100, 2: 200, 3: 300}
    current = {1: 150}
    assert detect_motion(baseline, current) == 50


def test_keep_strongest_motion_keeps_only_one_active_port() -> None:
    results = [
        {"stable_id": "arm-a", "delta": 120, "moved": True},
        {"stable_id": "arm-b", "delta": 180, "moved": True},
        {"stable_id": "arm-c", "delta": 20, "moved": False},
    ]

    assert keep_strongest_motion(results) == [
        {"stable_id": "arm-a", "delta": 120, "moved": False},
        {"stable_id": "arm-b", "delta": 180, "moved": True},
        {"stable_id": "arm-c", "delta": 20, "moved": False},
    ]


def test_resolve_active_motion_latches_previous_port_until_new_motion() -> None:
    results = [
        {"stable_id": "arm-a", "delta": 0, "moved": False},
        {"stable_id": "arm-b", "delta": 0, "moved": False},
    ]

    normalized, active_id = resolve_active_motion(results, active_id="arm-a")

    assert active_id == "arm-a"
    assert normalized == [
        {"stable_id": "arm-a", "delta": 0, "moved": True},
        {"stable_id": "arm-b", "delta": 0, "moved": False},
    ]


def test_motion_detector_tracks_recent_motion_instead_of_sticky_displacement() -> None:
    port = SerialInterface(dev="/dev/ttyACM0", motor_ids=(1, 2))
    detector = port.motion_detector

    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        side_effect=[
            {1: 100, 2: 200},
            {1: 160, 2: 200},
            {1: 160, 2: 200},
            {1: 160, 2: 280},
        ],
    ):
        detector.capture_baseline()
        first = detector.poll()
        second = detector.poll()
        third = detector.poll()

    assert first.delta == 60
    assert first.moved is True
    assert second.delta == 0
    assert second.moved is False
    assert third.delta == 80
    assert third.moved is True
