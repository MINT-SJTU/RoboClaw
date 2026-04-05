"""Tests for setup wizard dashboard routes."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.interface.serial import SerialInterface
from roboclaw.embodied.interface.video import VideoInterface
from roboclaw.embodied.manifest import Manifest
from roboclaw.http.routes.setup import register_setup_routes


_RAW_PORTS = [
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.1",
        by_id="/dev/serial/by-id/usb-ABC-if00",
        dev="/dev/ttyACM0",
    ),
]

_MOCK_CAMERAS = [
    VideoInterface(by_path="/dev/v4l/by-path/cam0", by_id="", dev="/dev/video0", width=640, height=480),
]


def _scan_context(cameras: list | None = None):
    """Context manager that patches discover_all internals so the real method
    runs, populates _scanned_ports, and returns motor-bearing ports."""
    cam_list = cameras if cameras is not None else []
    return contextlib.ExitStack(), [
        patch("roboclaw.embodied.hardware.discovery.scan_serial_ports", return_value=_RAW_PORTS),
        patch("roboclaw.embodied.hardware.discovery.scan_cameras", return_value=cam_list),
        patch("roboclaw.embodied.hardware.probers.feetech.FeetechProber.probe", return_value=[1, 2, 3, 4, 5, 6]),
        patch("roboclaw.embodied.hardware.discovery.suppress_stderr", return_value=99),
        patch("roboclaw.embodied.hardware.discovery.restore_stderr"),
    ]


@contextlib.contextmanager
def _patched_scan(cameras: list | None = None):
    """Convenience wrapper: enter all scan patches at once."""
    cam_list = cameras if cameras is not None else []
    with (
        patch("roboclaw.embodied.hardware.discovery.scan_serial_ports", return_value=_RAW_PORTS),
        patch("roboclaw.embodied.hardware.discovery.scan_cameras", return_value=cam_list),
        patch("roboclaw.embodied.hardware.probers.feetech.FeetechProber.probe", return_value=[1, 2, 3, 4, 5, 6]),
        patch("roboclaw.embodied.hardware.discovery.suppress_stderr", return_value=99),
        patch("roboclaw.embodied.hardware.discovery.restore_stderr"),
    ):
        yield


def _make_app(session_busy: bool = False) -> FastAPI:
    """Create a minimal FastAPI app with setup routes registered."""
    from roboclaw.embodied.hardware.discovery import HardwareDiscovery

    app = FastAPI()
    svc = MagicMock()
    svc.busy = session_busy
    svc.embodiment_busy = session_busy
    svc.busy_reason = "recording"

    scanner = HardwareDiscovery()
    setup = MagicMock()

    def _run_full_scan(model=""):
        return {
            "ports": scanner.discover_all(),
            "cameras": scanner.discover_cameras(),
        }

    setup.run_full_scan = _run_full_scan
    setup.capture_previews = scanner.capture_camera_previews
    setup.start_motion_detection = scanner.start_motion_detection
    setup.poll_motion = scanner.poll_motion
    setup.stop_motion_detection = MagicMock(side_effect=lambda: scanner.stop_motion_detection())
    setup.to_dict = MagicMock(return_value={"phase": "idle"})

    if session_busy:
        from roboclaw.embodied.service import EmbodimentBusyError
        setup.run_full_scan = MagicMock(
            side_effect=EmbodimentBusyError("Embodiment busy: recording"),
        )

    svc.setup = setup
    svc.manifest = MagicMock(spec=Manifest)
    svc.manifest.snapshot = {
        "arms": [{"alias": "left", "type": "so101_follower"}],
        "cameras": [{"alias": "top"}],
        "hands": [],
    }

    app.state.embodied_service = svc
    app.state.setup_wizard = scanner
    register_setup_routes(app, svc)
    return app


def test_scan_returns_ports_and_cameras() -> None:
    app = _make_app()
    client = TestClient(app)
    with _patched_scan(cameras=_MOCK_CAMERAS):
        resp = client.post("/api/dashboard/setup/scan")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ports"]) == 1
    assert data["ports"][0]["motor_ids"] == [1, 2, 3, 4, 5, 6]
    assert len(data["cameras"]) == 1


def test_motion_start_after_scan() -> None:
    app = _make_app()
    client = TestClient(app)
    with _patched_scan():
        client.post("/api/dashboard/setup/scan")

    with patch("roboclaw.embodied.hardware.motion_detector.MotionDetector._read_positions", return_value={1: 100, 2: 200}):
        resp = client.post("/api/dashboard/setup/motion/start")

    assert resp.status_code == 200
    assert resp.json()["status"] == "watching"
    assert resp.json()["port_count"] == 1


def test_motion_start_without_scan_returns_400() -> None:
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/dashboard/setup/motion/start")
    assert resp.status_code == 400


def test_motion_poll_returns_deltas() -> None:
    app = _make_app()
    client = TestClient(app)
    with _patched_scan():
        client.post("/api/dashboard/setup/scan")
    with patch("roboclaw.embodied.hardware.motion_detector.MotionDetector._read_positions", return_value={1: 100, 2: 200}):
        client.post("/api/dashboard/setup/motion/start")
    with patch("roboclaw.embodied.hardware.motion_detector.MotionDetector._read_positions", return_value={1: 200, 2: 300}):
        resp = client.get("/api/dashboard/setup/motion/poll")

    assert resp.status_code == 200
    ports = resp.json()["ports"]
    assert len(ports) == 1
    assert ports[0]["delta"] == 200  # |200-100| + |300-200|
    assert ports[0]["moved"] is True


def test_motion_poll_without_start_returns_400() -> None:
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/dashboard/setup/motion/poll")
    assert resp.status_code == 400


def test_motion_stop_clears_state() -> None:
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/api/dashboard/setup/motion/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert app.state.setup_wizard.motion_active is False
    assert app.state.embodied_service.setup.stop_motion_detection.called


def test_add_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    with patch("roboclaw.http.routes.setup._resolve_serial_interface", return_value="iface"):
        resp = client.post(
            "/api/dashboard/setup/arm",
            json={"alias": "left", "arm_type": "so101_follower", "port_id": "/dev/serial/by-id/usb-ABC"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    svc.bind_arm.assert_called_once_with("left", "so101_follower", "iface")


def test_remove_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    resp = client.delete("/api/dashboard/setup/arm/left")
    assert resp.status_code == 200
    assert resp.json()["status"] == "removed"
    svc.unbind_arm.assert_called_once_with("left")


def test_rename_arm() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    resp = client.patch(
        "/api/dashboard/setup/arm/left/rename",
        json={"new_alias": "right"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "renamed", "old": "left", "new": "right"}
    svc.rename_arm.assert_called_once_with("left", "right")


def test_add_camera() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    with patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=_MOCK_CAMERAS):
        resp = client.post(
            "/api/dashboard/setup/camera",
            json={"alias": "top", "camera_index": 0},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "added"
    svc.bind_camera.assert_called_once_with("top", _MOCK_CAMERAS[0])


def test_remove_camera() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    resp = client.delete("/api/dashboard/setup/camera/top")
    assert resp.status_code == 200
    svc.unbind_camera.assert_called_once_with("top")


def test_current_setup() -> None:
    app = _make_app()
    client = TestClient(app)
    svc = app.state.embodied_service
    resp = client.get("/api/dashboard/setup/current")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["arms"]) == 1
    assert data["arms"][0]["alias"] == "left"
    assert data == svc.manifest.snapshot


def test_scan_returns_409_when_recording() -> None:
    app = _make_app(session_busy=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/dashboard/setup/scan")
    assert resp.status_code == 409
