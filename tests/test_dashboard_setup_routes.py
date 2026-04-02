"""Tests for setup wizard dashboard routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.web.dashboard_setup import register_setup_routes


_MOCK_PORTS = [
    {
        "by_path": "/dev/serial/by-path/pci-0:2.1",
        "by_id": "/dev/serial/by-id/usb-ABC-if00",
        "dev": "/dev/ttyACM0",
        "motor_ids": [1, 2, 3, 4, 5, 6],
    },
]

_MOCK_CAMERAS = [
    {"by_path": "/dev/v4l/by-path/cam0", "by_id": "", "dev": "/dev/video0", "width": 640, "height": 480},
]


def _make_app(session_busy: bool = False) -> FastAPI:
    """Create a minimal FastAPI app with setup routes registered."""
    app = FastAPI()
    session = MagicMock()
    session.busy = session_busy
    app.state.dashboard_session = session
    register_setup_routes(app)
    return app


def test_scan_returns_ports_and_cameras() -> None:
    app = _make_app()
    client = TestClient(app)
    with (
        patch(
            "roboclaw.web.dashboard_setup.scan_serial_ports",
            return_value=[{"by_path": "", "by_id": "/dev/serial/by-id/usb-ABC-if00", "dev": "/dev/ttyACM0"}],
        ),
        patch(
            "roboclaw.web.dashboard_setup._filter_feetech_ports",
            return_value=_MOCK_PORTS,
        ),
        patch(
            "roboclaw.web.dashboard_setup.scan_cameras",
            return_value=_MOCK_CAMERAS,
        ),
    ):
        resp = client.post("/api/dashboard/setup/scan")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ports"]) == 1
    assert data["ports"][0]["motor_ids"] == [1, 2, 3, 4, 5, 6]
    assert len(data["cameras"]) == 1


def test_motion_start_after_scan() -> None:
    app = _make_app()
    client = TestClient(app)
    # Populate wizard state via scan
    with (
        patch("roboclaw.web.dashboard_setup.scan_serial_ports", return_value=[]),
        patch("roboclaw.web.dashboard_setup._filter_feetech_ports", return_value=_MOCK_PORTS),
        patch("roboclaw.web.dashboard_setup.scan_cameras", return_value=[]),
    ):
        client.post("/api/dashboard/setup/scan")

    with patch("roboclaw.web.dashboard_setup.read_positions", return_value={1: 100, 2: 200}):
        resp = client.post("/api/dashboard/setup/motion/start")

    assert resp.status_code == 200
    assert resp.json()["status"] == "watching"
    assert resp.json()["port_count"] == 1


def test_motion_start_without_scan_returns_400() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/api/dashboard/setup/motion/start")
    assert resp.status_code == 400


def test_motion_poll_returns_deltas() -> None:
    app = _make_app()
    client = TestClient(app)
    # Scan first
    with (
        patch("roboclaw.web.dashboard_setup.scan_serial_ports", return_value=[]),
        patch("roboclaw.web.dashboard_setup._filter_feetech_ports", return_value=_MOCK_PORTS),
        patch("roboclaw.web.dashboard_setup.scan_cameras", return_value=[]),
    ):
        client.post("/api/dashboard/setup/scan")
    # Start motion
    with patch("roboclaw.web.dashboard_setup.read_positions", return_value={1: 100, 2: 200}):
        client.post("/api/dashboard/setup/motion/start")
    # Poll with changed positions
    with patch("roboclaw.web.dashboard_setup.read_positions", return_value={1: 200, 2: 300}):
        resp = client.get("/api/dashboard/setup/motion/poll")

    assert resp.status_code == 200
    ports = resp.json()["ports"]
    assert len(ports) == 1
    assert ports[0]["delta"] == 200  # |200-100| + |300-200|
    assert ports[0]["moved"] is True


def test_motion_poll_without_start_returns_400() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.get("/api/dashboard/setup/motion/poll")
    assert resp.status_code == 400


def test_motion_stop_clears_state() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/api/dashboard/setup/motion/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert app.state.setup_wizard.active is False


def test_add_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    with patch("roboclaw.web.dashboard_setup.set_arm") as mock_set:
        resp = client.post(
            "/api/dashboard/setup/arm",
            json={"alias": "left", "arm_type": "so101_follower", "port_id": "/dev/serial/by-id/usb-ABC"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    mock_set.assert_called_once_with("left", "so101_follower", "/dev/serial/by-id/usb-ABC")


def test_remove_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    with patch("roboclaw.web.dashboard_setup.remove_arm") as mock_rm:
        resp = client.delete("/api/dashboard/setup/arm/left")
    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"
    mock_rm.assert_called_once_with("left")


def test_rename_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    with patch("roboclaw.web.dashboard_setup.rename_arm") as mock_rn:
        resp = client.patch(
            "/api/dashboard/setup/arm/left/rename",
            json={"new_alias": "right"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"status": "renamed", "old": "left", "new": "right"}
    mock_rn.assert_called_once_with("left", "right")


def test_add_camera() -> None:
    app = _make_app()
    client = TestClient(app)
    with patch("roboclaw.web.dashboard_setup.set_camera") as mock_set:
        resp = client.post(
            "/api/dashboard/setup/camera",
            json={"alias": "top", "camera_index": 0},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    mock_set.assert_called_once_with("top", 0)


def test_remove_camera() -> None:
    app = _make_app()
    client = TestClient(app)
    with patch("roboclaw.web.dashboard_setup.remove_camera") as mock_rm:
        resp = client.delete("/api/dashboard/setup/camera/top")
    assert resp.status_code == 200
    mock_rm.assert_called_once_with("top")


def test_current_setup() -> None:
    app = _make_app()
    client = TestClient(app)
    mock_setup = {
        "arms": [{"alias": "left", "type": "so101_follower"}],
        "cameras": [{"alias": "top"}],
        "hands": [],
    }
    with patch("roboclaw.web.dashboard_setup.load_setup", return_value=mock_setup):
        resp = client.get("/api/dashboard/setup/current")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["arms"]) == 1
    assert data["arms"][0]["alias"] == "left"


def test_scan_returns_409_when_recording() -> None:
    app = _make_app(session_busy=True)
    client = TestClient(app)
    resp = client.post("/api/dashboard/setup/scan")
    assert resp.status_code == 409
