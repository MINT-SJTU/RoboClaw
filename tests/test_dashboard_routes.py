"""Tests for the consolidated dashboard API routes."""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import patch, PropertyMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.embodiment.hardware.monitor import (
    ArmStatus, CameraStatus, FaultType, HardwareFault,
)
from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.http.routes import hardware as hardware_routes
from roboclaw.http.routes import recovery as recovery_routes
from roboclaw.http.routes import register_all_routes


@pytest.fixture(autouse=True)
def isolated_roboclaw_home(tmp_path):
    with patch(
        "roboclaw.embodied.embodiment.lock.get_roboclaw_home",
        return_value=tmp_path,
    ), patch(
        "roboclaw.embodied.embodiment.manifest.helpers.get_roboclaw_home",
        return_value=tmp_path,
    ):
        yield


@pytest.fixture()
def app(tmp_path):
    """Minimal FastAPI app with dashboard routes registered."""
    app = FastAPI()

    class FakeChannel:
        async def broadcast(self, event):
            pass

    from roboclaw.embodied.board import Board
    from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
    from roboclaw.embodied.embodiment.manifest import Manifest

    manifest_path = tmp_path / "manifest.json"
    board = Board()
    manifest = Manifest(path=manifest_path, board=board)
    hw_monitor = HardwareMonitor(board=board, manifest=manifest)
    app.state.hardware_monitor = hw_monitor

    service = EmbodiedService(hardware_monitor=hw_monitor, board=board, manifest=manifest)
    app.state.embodied_service = service

    register_all_routes(app, FakeChannel(), service, get_config=lambda: ("0.0.0.0", 8080))
    return app


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Session status
# ---------------------------------------------------------------------------

class TestSessionStatus:
    def test_idle_status(self, client):
        resp = client.get("/api/session/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["dataset"] is None

    def test_status_fields(self, client):
        resp = client.get("/api/session/status")
        data = resp.json()
        for field in ("state", "episode_phase", "saved_episodes", "target_episodes", "dataset"):
            assert field in data


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_teleop_stop_from_idle(self, client):
        """Stopping teleop from idle is a no-op (returns idle)."""
        resp = client.post("/api/teleop/stop")
        assert resp.status_code == 200

    def test_record_stop_from_idle(self, client):
        resp = client.post("/api/record/stop")
        assert resp.status_code == 200

    def test_save_episode_no_subprocess(self, client):
        resp = client.post("/api/record/episode/save")
        assert resp.status_code == 200
        assert resp.json() == {"status": "episode_saved"}

    def test_discard_episode_no_subprocess(self, client):
        resp = client.post("/api/record/episode/discard")
        assert resp.status_code == 200
        assert resp.json() == {"status": "episode_discarded"}


# ---------------------------------------------------------------------------
# Hardware status
# ---------------------------------------------------------------------------

class TestHardwareStatus:
    def test_hardware_status(self, client):
        resp = client.get("/api/hardware/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "arms" in data
        assert "cameras" in data
        assert "capabilities" in data

    def test_capabilities_allow_replay_when_camera_is_disconnected(self, client, app):
        service = app.state.embodied_service
        service.bind_arm("follower", "so101_follower", SerialInterface(dev="/tmp/follower"))
        service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))
        service.bind_camera("wrist", VideoInterface(dev="/tmp/wrist"))

        with patch("roboclaw.embodied.service.check_arm_status") as arm_status, patch(
            "roboclaw.embodied.service.check_camera_status",
        ) as camera_status:
            arm_status.side_effect = [
                ArmStatus("follower", "so101_follower", "follower", True, True),
                ArmStatus("leader", "so101_leader", "leader", True, True),
            ]
            camera_status.return_value = CameraStatus("wrist", False, 640, 480)
            resp = client.get("/api/hardware/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False
        assert data["capabilities"]["teleop"]["ready"] is True
        assert data["capabilities"]["record"]["ready"] is False
        assert data["capabilities"]["record_without_cameras"]["ready"] is True
        assert data["capabilities"]["replay"]["ready"] is True
        assert data["capabilities"]["infer"]["ready"] is False

    def test_record_start_returns_400_when_cameras_are_required(self, client, app):
        service = app.state.embodied_service
        service.bind_arm("follower", "so101_follower", SerialInterface(dev="/tmp/follower"))
        service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))

        with patch("roboclaw.embodied.service.check_arm_status") as arm_status:
            arm_status.side_effect = [
                ArmStatus("follower", "so101_follower", "follower", True, True),
                ArmStatus("leader", "so101_leader", "leader", True, True),
            ]
            resp = client.post(
                "/api/record/start",
                json={"task": "pick", "num_episodes": 1, "episode_time_s": 10, "reset_time_s": 1},
            )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "No cameras configured"

    def test_recovery_faults_return_current_faults(self, client, app):
        fault = HardwareFault(
            fault_type=FaultType.CAMERA_DISCONNECTED,
            device_alias="wrist",
            message="Camera 'wrist' device not found",
            timestamp=time.time(),
        )
        app.state.hardware_monitor._active_faults = {"camera_disconnected:wrist": fault}

        resp = client.get("/api/recovery/faults")

        assert resp.status_code == 200
        assert resp.json() == {
            "faults": [{
                "fault_type": "camera_disconnected",
                "device_alias": "wrist",
                "message": "Camera 'wrist' device not found",
                "timestamp": fault.timestamp,
            }],
        }

    def test_recovery_restart_dashboard_schedules_process_restart(self, client, app):
        with patch.object(recovery_routes, "schedule_dashboard_restart") as restart:
            resp = client.post("/api/recovery/restart-dashboard")

        assert resp.status_code == 200
        assert resp.json() == {"status": "restarting"}
        restart.assert_called_once_with(app)

    def test_hardware_previews_return_alias_keyed_urls(self, client, app):
        app.state.embodied_service.bind_camera("top", VideoInterface(dev="/dev/video0"))
        with patch(
            "roboclaw.http.routes.hardware.capture_named_camera_frames",
            return_value=[
                {
                    "alias": "top",
                    "preview_key": "00-top",
                    "image_path": "/tmp/roboclaw-camera-previews/hardware/00-top.jpg",
                },
            ],
        ):
            resp = client.post("/api/hardware/previews")

        assert resp.status_code == 200
        assert resp.json() == [
            {
                "alias": "top",
                "preview_key": "00-top",
                "image_path": "/tmp/roboclaw-camera-previews/hardware/00-top.jpg",
                "preview_url": "/api/hardware/previews/by-key/00-top",
            },
        ]

    def test_hardware_preview_image_lookup_uses_exact_key(self, client, tmp_path: Path):
        hardware_routes.HARDWARE_PREVIEW_DIR = tmp_path
        (tmp_path / "00-top.jpg").write_bytes(b"jpeg-data")
        (tmp_path / "00-stale.jpg").write_bytes(b"stale-data")

        resp = client.get("/api/hardware/previews/by-key/00-top")

        assert resp.status_code == 200
        assert resp.content == b"jpeg-data"
        assert resp.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_list_datasets(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_datasets_no_root_uses_default(self, client):
        """When no datasets root is configured, falls back to default path."""
        resp = client.get("/api/datasets")
        assert resp.status_code == 200

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/datasets/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Servo positions
# ---------------------------------------------------------------------------

class TestServoPositions:
    def test_servo_when_idle(self, client):
        resp = client.get("/api/hardware/servos")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None

    def test_servo_when_busy(self, client, app):
        with patch.object(
            app.state.embodied_service._file_lock,
            "try_shared",
            return_value=False,
        ):
            resp = client.get("/api/hardware/servos")
        assert resp.status_code == 200
        assert resp.json()["error"] == "busy"


# ---------------------------------------------------------------------------
# Network info
# ---------------------------------------------------------------------------

class TestNetworkInfo:
    def test_network_info(self, client):
        resp = client.get("/api/system/network")
        assert resp.status_code == 200
        data = resp.json()
        assert data["port"] == 8080
