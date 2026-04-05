"""Tests for the consolidated dashboard API routes."""

from __future__ import annotations

from unittest.mock import patch, PropertyMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.service import EmbodiedService
from roboclaw.http.routes import register_all_routes


@pytest.fixture()
def app(tmp_path):
    """Minimal FastAPI app with dashboard routes registered."""
    app = FastAPI()

    class FakeChannel:
        async def broadcast(self, event):
            pass

    from roboclaw.embodied.events import EventBus
    from roboclaw.embodied.hardware.monitor import HardwareMonitor
    from roboclaw.embodied.manifest import Manifest

    manifest_path = tmp_path / "manifest.json"
    event_bus = EventBus()
    manifest = Manifest(path=manifest_path, event_bus=event_bus)
    hw_monitor = HardwareMonitor(event_bus=event_bus, manifest=manifest)
    app.state.hardware_monitor = hw_monitor

    service = EmbodiedService(hardware_monitor=hw_monitor, event_bus=event_bus, manifest=manifest)
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
        resp = client.get("/api/dashboard/session/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["dataset"] is None

    def test_status_fields(self, client):
        resp = client.get("/api/dashboard/session/status")
        data = resp.json()
        for field in ("state", "episode_phase", "saved_episodes", "target_episodes", "dataset"):
            assert field in data


# ---------------------------------------------------------------------------
# Session lifecycle (wrong state -> 500)
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_teleop_stop_from_idle(self, client):
        """Stopping teleop from idle is a no-op (returns idle)."""
        resp = client.post("/api/dashboard/session/teleop/stop")
        assert resp.status_code == 200

    def test_record_stop_from_idle(self, client):
        resp = client.post("/api/dashboard/session/record/stop")
        assert resp.status_code == 200

    def test_save_episode_no_subprocess(self, client):
        resp = client.post("/api/dashboard/session/episode/save")
        assert resp.status_code == 400

    def test_discard_episode_no_subprocess(self, client):
        resp = client.post("/api/dashboard/session/episode/discard")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Hardware status
# ---------------------------------------------------------------------------

class TestHardwareStatus:
    def test_hardware_status(self, client):
        resp = client.get("/api/dashboard/hardware-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "ready" in data
        assert "arms" in data
        assert "cameras" in data


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_list_datasets(self, client):
        resp = client.get("/api/dashboard/datasets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_datasets_no_root_uses_default(self, client):
        """When no datasets root is configured, falls back to default path."""
        resp = client.get("/api/dashboard/datasets")
        assert resp.status_code == 200

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/dashboard/datasets/nope")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Servo positions
# ---------------------------------------------------------------------------

class TestServoPositions:
    def test_servo_when_idle(self, client):
        resp = client.get("/api/dashboard/servo-positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None

    def test_servo_when_busy(self, client, app):
        app.state.embodied_service._engine._state = "recording"
        resp = client.get("/api/dashboard/servo-positions")
        assert resp.status_code == 200
        assert resp.json()["error"] == "busy"
        app.state.embodied_service._engine._state = "idle"


# ---------------------------------------------------------------------------
# Network info
# ---------------------------------------------------------------------------

class TestNetworkInfo:
    def test_network_info(self, client):
        resp = client.get("/api/dashboard/network-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["port"] == 8080
