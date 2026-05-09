from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
from roboclaw.http.routes.workflows import register_workflow_routes


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
    app = FastAPI()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "version": 2,
        "arms": [],
        "hands": [],
        "cameras": [],
        "datasets": {"root": str(tmp_path / "datasets")},
        "policies": {"root": str(tmp_path / "policies")},
    }), encoding="utf-8")
    board = Board()
    manifest = Manifest(path=manifest_path, board=board)
    hw_monitor = HardwareMonitor(board=board, manifest=manifest)
    service = EmbodiedService(hardware_monitor=hw_monitor, board=board, manifest=manifest)
    service.bind_arm("follower", "so101_follower", SerialInterface(dev="/tmp/follower"))
    service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))
    service.bind_camera("wrist", VideoInterface(dev="/tmp/wrist"))
    register_workflow_routes(app, service)
    app.state.embodied_service = service
    return app


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_workflow_plan_route_returns_compiled_stages(client):
    resp = client.post("/api/workflows/plan", json={
        "name": "pick-cube-pipeline",
        "hardware": {"useCameras": True},
        "record": {
            "enabled": True,
            "task": "pick cube",
            "datasetName": "pick_cube_v1",
        },
        "train": {
            "enabled": True,
            "policyType": "act",
        },
        "infer": {
            "enabled": True,
            "datasetName": "eval_pick_cube_v1",
        },
    })

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert [stage["stage"] for stage in payload["stages"]] == ["record", "train", "infer"]
    assert payload["stages"][0]["ready"] is True
    assert payload["stages"][1]["ready"] is False
    assert payload["stages"][1]["blockedBy"] == ["record"]
    assert payload["stages"][1]["datasetName"] == "pick_cube_v1"
    assert payload["stages"][2]["ready"] is False
    assert payload["stages"][2]["blockedBy"] == ["train"]
    assert payload["stages"][2]["checkpointPath"].endswith(
        "pick_cube_v1/checkpoints/last/pretrained_model"
    )


def test_workflow_run_route_delegates_to_service(client, app):
    service = app.state.embodied_service
    service.start_workflow_phase = AsyncMock(return_value={"status": "recording", "dataset_name": "demo"})

    resp = client.post("/api/workflows/run/record", json={
        "record": {
            "enabled": True,
            "task": "pick cube",
            "datasetName": "demo",
        },
    })

    assert resp.status_code == 200
    assert resp.json() == {"status": "recording", "dataset_name": "demo"}
    service.start_workflow_phase.assert_awaited_once()


def test_workflow_run_route_returns_400_for_blocked_stage(client, app):
    service = app.state.embodied_service
    service.start_workflow_phase = AsyncMock(side_effect=RuntimeError("Workflow phase 'train' is waiting on: record."))

    resp = client.post("/api/workflows/run/train", json={
        "record": {
            "enabled": True,
            "task": "pick cube",
            "datasetName": "pick_cube_v1",
        },
        "train": {
            "enabled": True,
        },
    })

    assert resp.status_code == 400
    assert resp.json() == {"detail": "Workflow phase 'train' is waiting on: record."}
