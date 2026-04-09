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

from roboclaw.embodied.embodiment.hardware.motion import detect_motion
from roboclaw.embodied.embodiment.hardware.scan import port_candidates


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


def test_port_candidates_adds_cu_variant_on_macos() -> None:
    import roboclaw.embodied.embodiment.hardware.scan as scan_module
    with patch.object(scan_module.sys, "platform", "darwin"):
        assert port_candidates("/dev/tty.usbmodem123") == [
            "/dev/tty.usbmodem123",
            "/dev/cu.usbmodem123",
        ]
