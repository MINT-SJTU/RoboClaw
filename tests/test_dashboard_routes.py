"""Tests for the consolidated dashboard API routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.web.dashboard import register_dashboard_routes
from roboclaw.web.dashboard_session import DashboardSession


@pytest.fixture()
def app():
    """Minimal FastAPI app with dashboard routes registered."""
    app = FastAPI()

    async def _noop_broadcast(event):
        pass

    class FakeChannel:
        async def broadcast(self, event):
            pass

    from roboclaw.embodied.hardware_monitor import HardwareMonitor
    app.state.hardware_monitor = HardwareMonitor(
        on_fault=lambda f: None,
        on_fault_resolved=lambda f: None,
    )
    register_dashboard_routes(app, FakeChannel(), get_config=lambda: ("0.0.0.0", 8080))
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
# Session lifecycle (wrong state → 500)
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
        assert resp.status_code == 500

    def test_discard_episode_no_subprocess(self, client):
        resp = client.post("/api/dashboard/session/episode/discard")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Hardware status
# ---------------------------------------------------------------------------

class TestHardwareStatus:
    def test_hardware_status(self, client, monkeypatch):
        monkeypatch.setattr(
            "roboclaw.web.dashboard.load_setup",
            lambda: {"arms": [], "cameras": []},
        )
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
    def test_list_datasets(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "roboclaw.web.dashboard.load_setup",
            lambda: {"datasets": {"root": str(tmp_path)}},
        )
        resp = client.get("/api/dashboard/datasets")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_datasets_no_root_uses_default(self, client, monkeypatch):
        """When no datasets root is configured, falls back to default path."""
        monkeypatch.setattr(
            "roboclaw.web.dashboard.load_setup",
            lambda: {"datasets": {}},
        )
        resp = client.get("/api/dashboard/datasets")
        assert resp.status_code == 200

    def test_delete_nonexistent(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "roboclaw.web.dashboard.load_setup",
            lambda: {"datasets": {"root": str(tmp_path)}},
        )
        resp = client.delete("/api/dashboard/datasets/nope")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Servo positions
# ---------------------------------------------------------------------------

class TestServoPositions:
    def test_servo_when_idle(self, client, monkeypatch):
        monkeypatch.setattr(
            "roboclaw.embodied.motors.load_setup",
            lambda: {"arms": []},
        )
        resp = client.get("/api/dashboard/servo-positions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None

    def test_servo_when_busy(self, client, app):
        app.state.dashboard_session._state = "recording"
        resp = client.get("/api/dashboard/servo-positions")
        assert resp.status_code == 200
        assert resp.json()["error"] == "busy"
        app.state.dashboard_session._state = "idle"


# ---------------------------------------------------------------------------
# Network info
# ---------------------------------------------------------------------------

class TestNetworkInfo:
    def test_network_info(self, client):
        resp = client.get("/api/dashboard/network-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["port"] == 8080
