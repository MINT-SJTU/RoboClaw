"""Tests for current setup wizard and device routes."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.board import Board
from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.http.routes import setup as setup_routes
from roboclaw.http.routes.devices import register_device_routes
from roboclaw.http.routes.setup import register_setup_routes


_RAW_PORTS = [
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.1",
        by_id="/dev/serial/by-id/usb-ABC-if00",
        dev="/dev/ttyACM0",
        motor_ids=(1, 2, 3, 4, 5, 6),
    ),
]

_RAW_PORTS_PAIR = [
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.1",
        by_id="/dev/serial/by-id/usb-ABC-if00",
        dev="/dev/ttyACM0",
        motor_ids=(1, 2, 3, 4, 5, 6),
    ),
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.2",
        by_id="/dev/serial/by-id/usb-XYZ-if00",
        dev="/dev/ttyACM1",
        motor_ids=(1, 2, 3, 4, 5, 6),
    ),
]

_MOCK_CAMERAS = [
    VideoInterface(
        by_path="/dev/v4l/by-path/cam0",
        by_id="",
        dev="/dev/video0",
        width=640,
        height=480,
    ),
]


@pytest.fixture(autouse=True)
def isolated_roboclaw_home(tmp_path: Path):
    with patch(
        "roboclaw.embodied.embodiment.lock.get_roboclaw_home",
        return_value=tmp_path,
    ):
        yield


@contextlib.contextmanager
def _patched_scan(ports: list | None = None, cameras: list | None = None):
    port_list = ports if ports is not None else _RAW_PORTS
    cam_list = cameras if cameras is not None else []
    with (
        patch(
            "roboclaw.embodied.embodiment.hardware.discovery.scan_serial_ports",
            return_value=port_list,
        ),
        patch(
            "roboclaw.embodied.embodiment.hardware.discovery.scan_cameras",
            return_value=cam_list,
        ),
        patch(
            "roboclaw.embodied.embodiment.hardware.discovery.HardwareDiscovery._ensure_serial_access",
            return_value=None,
        ),
        patch(
            "roboclaw.embodied.embodiment.hardware.probers.feetech.FeetechProber.probe",
            return_value=[1, 2, 3, 4, 5, 6],
        ),
        patch(
            "roboclaw.embodied.embodiment.hardware.discovery.suppress_stderr",
            return_value=99,
        ),
        patch("roboclaw.embodied.embodiment.hardware.discovery.restore_stderr"),
    ):
        yield


def _make_app(tmp_path: Path, session_busy: bool = False) -> FastAPI:
    app = FastAPI()
    board = Board()
    manifest = Manifest(path=tmp_path / "manifest.json", board=board)
    monitor = HardwareMonitor(board=board, manifest=manifest)
    service = EmbodiedService(hardware_monitor=monitor, board=board, manifest=manifest)
    if session_busy:
        service.acquire_embodiment("recording")
    app.state.embodied_service = service
    register_setup_routes(app, service)
    register_device_routes(app, service)
    return app


def test_scan_returns_ports_and_cameras(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan(cameras=_MOCK_CAMERAS):
        resp = client.post("/api/setup/scan")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ports"]) == 1
    assert data["ports"][0]["motor_ids"] == [1, 2, 3, 4, 5, 6]
    assert len(data["cameras"]) == 1
    assert data["cameras"][0]["stable_id"] == "/dev/v4l/by-path/cam0"


def test_devices_catalog_returns_arm_models(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/api/devices/catalog")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["categories"][0]["id"] == "arm"
    assert payload["categories"][0]["supported"] is True
    assert payload["models"]["arm"] == [
        {"name": "so101", "roles": ["follower", "leader"]},
        {"name": "koch", "roles": ["follower", "leader"]},
    ]


def test_setup_previews_return_keyed_urls(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    service = app.state.embodied_service
    service.setup.capture_previews = lambda output_dir: [
        {
            "stable_id": "/dev/v4l/by-path/cam0",
            "preview_key": "00-preview",
            "image_path": f"{output_dir}/00-preview.jpg",
        },
    ]

    resp = client.post("/api/setup/previews")

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "stable_id": "/dev/v4l/by-path/cam0",
            "preview_key": "00-preview",
            "image_path": "/tmp/roboclaw-camera-previews/setup/00-preview.jpg",
            "preview_url": "/api/setup/previews/by-key/00-preview",
        },
    ]


def test_setup_preview_image_lookup_uses_exact_key(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    setup_routes.SETUP_PREVIEW_DIR = tmp_path
    (tmp_path / "00-preview.jpg").write_bytes(b"jpeg-data")
    (tmp_path / "00-stale.jpg").write_bytes(b"stale-data")

    resp = client.get("/api/setup/previews/by-key/00-preview")

    assert resp.status_code == 200
    assert resp.content == b"jpeg-data"
    assert resp.headers["cache-control"] == "no-store"


def test_motion_start_after_scan(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan():
        client.post("/api/setup/scan")
    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        return_value={1: 100, 2: 200},
    ):
        resp = client.post("/api/setup/motion/start")

    assert resp.status_code == 200
    assert resp.json() == {"status": "watching", "port_count": 1}


def test_motion_start_without_scan_returns_400(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/setup/motion/start")
    assert resp.status_code == 400


def test_motion_poll_returns_deltas(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan():
        client.post("/api/setup/scan")
    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        return_value={1: 100, 2: 200},
    ):
        client.post("/api/setup/motion/start")
    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        return_value={1: 200, 2: 300},
    ):
        resp = client.get("/api/setup/motion/poll")

    assert resp.status_code == 200
    assert resp.json()["ports"] == [
        {
            "stable_id": "/dev/serial/by-id/usb-ABC-if00",
            "dev": "/dev/ttyACM0",
            "by_id": "/dev/serial/by-id/usb-ABC-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 200,
            "moved": True,
        }
    ]


def test_motion_poll_latches_last_active_port_until_another_port_moves(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan(ports=_RAW_PORTS_PAIR):
        client.post("/api/setup/scan")
    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        side_effect=[
            {1: 100, 2: 200},
            {1: 100, 2: 200},
        ],
    ):
        client.post("/api/setup/motion/start")

    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        side_effect=[
            {1: 200, 2: 300},
            {1: 100, 2: 200},
        ],
    ):
        first = client.get("/api/setup/motion/poll")

    assert first.status_code == 200
    assert first.json()["ports"] == [
        {
            "stable_id": "/dev/serial/by-id/usb-ABC-if00",
            "dev": "/dev/ttyACM0",
            "by_id": "/dev/serial/by-id/usb-ABC-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 200,
            "moved": True,
        },
        {
            "stable_id": "/dev/serial/by-id/usb-XYZ-if00",
            "dev": "/dev/ttyACM1",
            "by_id": "/dev/serial/by-id/usb-XYZ-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 0,
            "moved": False,
        },
    ]

    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        side_effect=[
            {1: 200, 2: 300},
            {1: 100, 2: 200},
        ],
    ):
        second = client.get("/api/setup/motion/poll")

    assert second.status_code == 200
    assert second.json()["ports"] == [
        {
            "stable_id": "/dev/serial/by-id/usb-ABC-if00",
            "dev": "/dev/ttyACM0",
            "by_id": "/dev/serial/by-id/usb-ABC-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 0,
            "moved": True,
        },
        {
            "stable_id": "/dev/serial/by-id/usb-XYZ-if00",
            "dev": "/dev/ttyACM1",
            "by_id": "/dev/serial/by-id/usb-XYZ-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 0,
            "moved": False,
        },
    ]

    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        side_effect=[
            {1: 200, 2: 300},
            {1: 220, 2: 320},
        ],
    ):
        third = client.get("/api/setup/motion/poll")

    assert third.status_code == 200
    assert third.json()["ports"] == [
        {
            "stable_id": "/dev/serial/by-id/usb-ABC-if00",
            "dev": "/dev/ttyACM0",
            "by_id": "/dev/serial/by-id/usb-ABC-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 0,
            "moved": False,
        },
        {
            "stable_id": "/dev/serial/by-id/usb-XYZ-if00",
            "dev": "/dev/ttyACM1",
            "by_id": "/dev/serial/by-id/usb-XYZ-if00",
            "motor_ids": [1, 2, 3, 4, 5, 6],
            "delta": 240,
            "moved": True,
        },
    ]


def test_motion_poll_without_start_returns_400(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/setup/motion/poll")
    assert resp.status_code == 400


def test_motion_stop_clears_state(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    resp = client.post("/api/setup/motion/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


def test_assign_commit_and_devices_list(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan(cameras=_MOCK_CAMERAS):
        scan = client.post("/api/setup/scan")
    port_id = scan.json()["ports"][0]["stable_id"]
    cam_id = scan.json()["cameras"][0]["stable_id"]

    arm_resp = client.post(
        "/api/setup/session/assign",
        json={
            "interface_stable_id": port_id,
            "alias": "follower",
            "spec_name": "so101_follower",
            "side": "",
        },
    )
    camera_resp = client.post(
        "/api/setup/session/assign",
        json={
            "interface_stable_id": cam_id,
            "alias": "top",
            "spec_name": "opencv",
            "side": "",
        },
    )
    commit = client.post("/api/setup/session/commit")
    devices = client.get("/api/devices")

    assert arm_resp.status_code == 200
    assert camera_resp.status_code == 200
    assert commit.status_code == 200
    assert commit.json() == {"status": "committed", "bindings_created": 2}
    payload = devices.json()
    assert [arm["alias"] for arm in payload["arms"]] == ["follower"]
    assert [camera["alias"] for camera in payload["cameras"]] == ["top"]


def test_dismiss_pending_camera_allows_commit_of_existing_assignments(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan(cameras=_MOCK_CAMERAS):
        scan = client.post("/api/setup/scan")
    port_id = scan.json()["ports"][0]["stable_id"]
    cam_id = scan.json()["cameras"][0]["stable_id"]

    arm_resp = client.post(
        "/api/setup/session/assign",
        json={
            "interface_stable_id": port_id,
            "alias": "follower",
            "spec_name": "so101_follower",
            "side": "",
        },
    )
    dismiss = client.post(
        "/api/setup/session/dismiss",
        json={"interface_stable_id": cam_id},
    )
    session = client.get("/api/setup/session")
    commit = client.post("/api/setup/session/commit")
    devices = client.get("/api/devices")

    assert arm_resp.status_code == 200
    assert dismiss.status_code == 200
    assert dismiss.json() == {"status": "dismissed", "interface_stable_id": cam_id}
    assert all(candidate["stable_id"] != cam_id for candidate in session.json()["candidates"])
    assert session.json()["unassigned"] == []
    assert commit.status_code == 200
    assert commit.json() == {"status": "committed", "bindings_created": 1}
    assert [arm["alias"] for arm in devices.json()["arms"]] == ["follower"]
    assert devices.json()["cameras"] == []


def test_unassign_removes_pending_assignment(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan():
        scan = client.post("/api/setup/scan")
    port_id = scan.json()["ports"][0]["stable_id"]

    client.post(
        "/api/setup/session/assign",
        json={
            "interface_stable_id": port_id,
            "alias": "follower",
            "spec_name": "so101_follower",
        },
    )
    resp = client.delete("/api/setup/session/assign/follower")
    session = client.get("/api/setup/session")

    assert resp.status_code == 200
    assert resp.json() == {"status": "unassigned", "alias": "follower"}
    assert session.json()["assignments"] == []


def test_dismiss_last_pending_serial_stops_motion_detection(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with _patched_scan():
        scan = client.post("/api/setup/scan")
    port_id = scan.json()["ports"][0]["stable_id"]

    with patch(
        "roboclaw.embodied.embodiment.hardware.motion_detector.MotionDetector._read_positions",
        return_value={1: 100, 2: 200},
    ):
        motion = client.post("/api/setup/motion/start")

    dismiss = client.post(
        "/api/setup/session/dismiss",
        json={"interface_stable_id": port_id},
    )
    session = client.get("/api/setup/session")

    assert motion.status_code == 200
    assert dismiss.status_code == 200
    assert session.status_code == 200
    assert session.json()["phase"] == "assigning"
    assert session.json()["unassigned"] == []


def test_device_rename_and_remove_routes(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    service = app.state.embodied_service
    service.bind_camera("top", VideoInterface(dev="/dev/video0"))

    rename = client.patch("/api/devices/cameras/top", json={"new_alias": "front"})
    remove = client.delete("/api/devices/cameras/front")
    devices = client.get("/api/devices")

    assert rename.status_code == 200
    assert rename.json() == {"status": "renamed", "old": "top", "new": "front"}
    assert remove.status_code == 200
    assert remove.json() == {"status": "removed", "alias": "front"}
    assert devices.json()["cameras"] == []

def test_setup_session_returns_busy_fields(tmp_path: Path) -> None:
    app = _make_app(tmp_path, session_busy=True)
    client = TestClient(app)
    resp = client.get("/api/setup/session")
    assert resp.status_code == 200
    assert resp.json()["busy"] is True
    assert resp.json()["busy_reason"] == "recording"


def test_setup_reset_calls_service_reset(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)
    with patch.object(app.state.embodied_service.setup, "reset") as reset:
        resp = client.post("/api/setup/session/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "reset"}
    reset.assert_called_once_with()


def test_scan_returns_409_when_recording(tmp_path: Path) -> None:
    app = _make_app(tmp_path, session_busy=True)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/setup/scan")
    assert resp.status_code == 409
