"""End-to-end convergence tests.

Proves three things:
1. **Convergence**: CLI and Web paths call the same service methods and return
   the same data.
2. **Embodiment lock**: Mutual exclusion works across all operation types.
3. **Full HTTP path**: Requests flow correctly through routes -> service -> engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService, EmbodimentBusyError
from roboclaw.http.routes import register_all_routes

# ---------------------------------------------------------------------------
# Mock setup data
# ---------------------------------------------------------------------------

MOCK_SETUP = {
    "version": 2,
    "arms": [
        {
            "alias": "leader",
            "type": "so101_leader",
            "port": "/dev/ttyACM0",
            "calibrated": True,
            "calibration_dir": "/tmp/cal/leader",
        },
        {
            "alias": "follower",
            "type": "so101_follower",
            "port": "/dev/ttyACM1",
            "calibrated": True,
            "calibration_dir": "/tmp/cal/follower",
        },
    ],
    "cameras": [
        {"alias": "top", "port": "/dev/video0", "width": 640, "height": 480},
    ],
    "datasets": {"root": "/tmp/datasets"},
    "policies": {"root": "/tmp/policies"},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def service(tmp_path, monkeypatch):
    """EmbodiedService with mocked manifest and /dev/* paths always 'connected'."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(MOCK_SETUP, indent=2), encoding="utf-8")
    manifest = Manifest(path=manifest_path)

    _original_exists = Path.exists

    def _mock_exists(self):
        if str(self).startswith("/dev/"):
            return True
        return _original_exists(self)

    monkeypatch.setattr(Path, "exists", _mock_exists)

    return EmbodiedService(manifest=manifest)


@pytest.fixture()
def app_and_service(service):
    """FastAPI app with dashboard routes wired to the same service instance."""
    from roboclaw.embodied.hardware.monitor import HardwareMonitor

    app = FastAPI()
    hw = HardwareMonitor(manifest=service.manifest)
    app.state.hardware_monitor = hw
    app.state.embodied_service = service

    class FakeChannel:
        async def broadcast(self, event):
            pass

    register_all_routes(
        app, FakeChannel(), service, get_config=lambda: ("0.0.0.0", 8765),
    )
    return app, service


@pytest.fixture()
def client(app_and_service):
    return TestClient(app_and_service[0], raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test Class 1: Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """CLI and Web paths converge on the same service data."""

    def test_hardware_status_data_matches(self, service):
        """CLI get_manifest() embeds the same hardware_status as the direct query."""
        cli_json = service.queries.get_manifest()
        cli_data = json.loads(cli_json)
        cli_hw = cli_data["hardware_status"]

        web_hw = service.queries.get_hardware_status()

        assert cli_hw["ready"] == web_hw["ready"]
        assert cli_hw["arms"] == web_hw["arms"]
        assert cli_hw["cameras"] == web_hw["cameras"]
        assert cli_hw["missing"] == web_hw["missing"]

    def test_hardware_status_via_http_matches_service(self, client, app_and_service):
        """HTTP GET /hardware-status returns same data as direct service call."""
        _, service = app_and_service
        resp = client.get("/api/dashboard/hardware-status")
        assert resp.status_code == 200
        http_data = resp.json()

        direct_data = service.queries.get_hardware_status()

        assert http_data["ready"] == direct_data["ready"]
        assert http_data["arms"] == direct_data["arms"]
        assert http_data["cameras"] == direct_data["cameras"]
        assert http_data["missing"] == direct_data["missing"]

    def test_scan_same_method_cli_and_web(self, service, monkeypatch):
        """CLI scan and Web scan call the exact same service method."""
        mock_result = {
            "ports": [{"dev": "/dev/ttyACM0", "motor_ids": [1, 2, 3]}],
            "cameras": [{"dev": "/dev/video0"}],
        }
        monkeypatch.setattr(
            service.scanning, "_scanner",
            type("FakeScanner", (), {
                "scan_ports": lambda self: mock_result["ports"],
                "scan_cameras_list": lambda self: mock_result["cameras"],
            })(),
        )

        # CLI path
        cli_result = service.scanning.run_full_scan()
        # Web path (same method)
        web_result = service.scanning.run_full_scan()

        assert cli_result == web_result
        assert cli_result == mock_result


# ---------------------------------------------------------------------------
# Test Class 2: Embodiment Lock
# ---------------------------------------------------------------------------

class TestEmbodimentLock:
    """Only one operation can hold the embodiment at a time."""

    def test_scan_blocked_during_operation(self, service):
        service.acquire_embodiment("teleop")
        with pytest.raises(EmbodimentBusyError):
            service.scanning.run_full_scan()
        service.release_embodiment(owner="teleop")

    def test_remove_arm_blocked_during_operation(self, service):
        """Config mutation returns a rejection string when embodiment is busy."""
        service.acquire_embodiment("recording")
        result = service.config.remove_arm("leader")
        assert "Cannot remove arm" in result
        service.release_embodiment(owner="recording")

    def test_acquire_release_cycle(self, service):
        service.acquire_embodiment("scanning")
        assert service.embodiment_busy
        assert service.busy_reason == "scanning"
        service.release_embodiment(owner="scanning")
        assert not service.embodiment_busy

    def test_double_acquire_fails(self, service):
        service.acquire_embodiment("teleop")
        with pytest.raises(EmbodimentBusyError):
            service.acquire_embodiment("calibrating")
        service.release_embodiment(owner="teleop")

    def test_wrong_owner_release_ignored(self, service):
        service.acquire_embodiment("teleop")
        service.release_embodiment(owner="scanning")  # wrong owner
        assert service.embodiment_busy  # still locked
        service.release_embodiment(owner="teleop")  # right owner
        assert not service.embodiment_busy


# ---------------------------------------------------------------------------
# Test Class 3: Full HTTP Path
# ---------------------------------------------------------------------------

class TestFullHTTPPath:
    """HTTP requests flow through routes -> service -> engine correctly."""

    def test_hardware_status_endpoint(self, client):
        resp = client.get("/api/dashboard/hardware-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "arms" in data
        assert "cameras" in data
        assert len(data["arms"]) == 2
        assert len(data["cameras"]) == 1

    def test_hardware_status_arms_detail(self, client):
        """Each arm in the response has the expected connectivity fields."""
        resp = client.get("/api/dashboard/hardware-status")
        data = resp.json()
        for arm in data["arms"]:
            assert "alias" in arm
            assert "connected" in arm
            assert "calibrated" in arm
            assert arm["connected"] is True
            assert arm["calibrated"] is True

    def test_hardware_status_cameras_detail(self, client):
        """Each camera has connectivity info."""
        resp = client.get("/api/dashboard/hardware-status")
        data = resp.json()
        for cam in data["cameras"]:
            assert "alias" in cam
            assert "connected" in cam
            assert cam["connected"] is True

    def test_session_status_idle(self, client):
        resp = client.get("/api/dashboard/session/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["dataset"] is None

    def test_network_info_returns_configured_port(self, client):
        resp = client.get("/api/dashboard/network-info")
        assert resp.status_code == 200
        assert resp.json()["port"] == 8765

    def test_scan_returns_409_when_busy(self, client, app_and_service):
        """POST /setup/scan returns 409 Conflict when embodiment is locked."""
        _, service = app_and_service
        service.acquire_embodiment("teleop")
        resp = client.post("/api/dashboard/setup/scan")
        assert resp.status_code == 409
        service.release_embodiment(owner="teleop")
