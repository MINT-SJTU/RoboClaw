"""Tests for arm identification flow."""

from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.tool import create_embodied_tools, EmbodiedToolGroup


_MOCK_SETUP = {
    "version": 2,
    "arms": [],
    "cameras": [],
    "datasets": {"root": "/data"},
    "policies": {"root": "/policies"},
}

_MOCK_PORTS = [
    {"by_path": "/dev/serial/by-path/pci-0:2.1", "by_id": "/dev/serial/by-id/usb-ABC-if00", "dev": "/dev/ttyACM0"},
]


def _hw_tool(tty_handoff=None) -> EmbodiedToolGroup:
    tools = create_embodied_tools(tty_handoff=tty_handoff)
    return next(t for t in tools if t.name == "embodied_hardware")


@pytest.mark.asyncio
async def test_identify_no_tty() -> None:
    """Identify without TTY handoff should return the no-TTY message."""
    tool = _hw_tool()  # no tty_handoff
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="identify")
    assert "local terminal" in result.lower()


@pytest.mark.asyncio
async def test_identify_no_ports() -> None:
    """Identify with empty scanned_ports should return an error message."""
    tool = _hw_tool(tty_handoff=AsyncMock())
    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]),
    ):
        result = await tool.execute(action="identify")
    assert result == "No serial ports detected."


@pytest.mark.asyncio
async def test_identify_success() -> None:
    """Identify with TTY and ports should run the subprocess and report success."""
    mock_handoff = AsyncMock()
    tool = _hw_tool(tty_handoff=mock_handoff)
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_PORTS),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="identify")

    assert result == "Arm identification complete."
    assert mock_handoff.call_count == 2  # start + stop
    argv = mock_runner.run_interactive.call_args[0][0]
    assert "roboclaw.embodied.identify" in " ".join(argv)


@pytest.mark.asyncio
async def test_identify_failure() -> None:
    """Identify subprocess failure should report the exit code."""
    tool = _hw_tool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (1, "identify subprocess error")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_PORTS),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="identify")

    assert "failed" in result.lower()
    assert "exit 1" in result


# ── Unit tests for identify.py helpers ───────────────────────────────


from roboclaw.embodied.identify import _filter_feetech_ports, _port_candidates, detect_motion


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
    import roboclaw.embodied.scan as scan_module
    with patch.object(scan_module.sys, "platform", "darwin"):
        assert _port_candidates("/dev/tty.usbmodem123") == [
            "/dev/tty.usbmodem123",
            "/dev/cu.usbmodem123",
        ]


def test_filter_feetech_ports_skips_probe_exceptions() -> None:
    ports = [
        {"dev": "/dev/tty.bad", "by_id": "", "by_path": ""},
        {"dev": "/dev/tty.good", "by_id": "", "by_path": ""},
    ]
    with patch("roboclaw.embodied.identify.probe_port", side_effect=[OSError("boom"), [1, 2, 3, 4, 5, 6]]):
        result = _filter_feetech_ports(ports)

    assert result == [{"dev": "/dev/tty.good", "by_id": "", "by_path": "", "motor_ids": [1, 2, 3, 4, 5, 6]}]


def test_filter_feetech_ports_prefers_usb_like_ports_before_generic_tty() -> None:
    ports = [
        {"dev": "/dev/tty", "by_id": "", "by_path": ""},
        {"dev": "/dev/tty.usbmodem123", "by_id": "", "by_path": ""},
    ]
    seen = []

    def _fake_probe(port):
        seen.append(port["dev"])
        if "usbmodem" in port["dev"]:
            return {**port, "motor_ids": [1, 2, 3, 4, 5, 6]}
        return None

    with patch("roboclaw.embodied.identify._probe_single_port", side_effect=_fake_probe):
        result = _filter_feetech_ports(ports)

    assert seen == ["/dev/tty.usbmodem123"]
    assert result == [{"dev": "/dev/tty.usbmodem123", "by_id": "", "by_path": "", "motor_ids": [1, 2, 3, 4, 5, 6]}]
