from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.board import Board
from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.cloud.evo_train import EvoTrainBridge, EvoTrainSettings
from roboclaw.account import AccountLedger
from roboclaw.http.routes.policies import register_policy_routes
from roboclaw.http.routes.train import register_train_routes
from roboclaw.http.routes import train_cloud as train_cloud_routes
from roboclaw.http.routes.train_cloud import register_train_cloud_routes
from roboclaw.training import TrainingJobStatus


class StubBridge:
    def __init__(self) -> None:
        self.enabled = True
        self.settings = SimpleNamespace(username="default-user")
        self.start_calls: list[dict[str, object]] = []
        self.stop_calls: list[dict[str, object]] = []
        self.status_calls: list[dict[str, object]] = []
        self.policy_calls: list[dict[str, object]] = []
        self.plan_calls: list[dict[str, object]] = []
        self.sku_calls: list[dict[str, object]] = []
        self.image_calls: list[dict[str, object]] = []
        self.runtime_match_calls: list[dict[str, object]] = []
        self.fail_next_plan = False
        self.status_result: dict[str, object] = {
            "job_id": "cloud-job-1",
            "status": "Running",
            "running": True,
            "message": "status: Running\nrunning: True",
            "task_name": "demo-act-1",
            "checkpoint_path": "/mnt/cloud/checkpoints/demo/checkpoints/last/pretrained_model",
            "dataset_path": "/mnt/cloud/datasets/demo",
            "provider": "aliyun",
        }
        self.policy_entries: list[dict[str, object]] = [
            {
                "name": "demo-act-1",
                "checkpoint": "/mnt/cloud/checkpoints/demo/checkpoints/last/pretrained_model",
                "dataset": "/mnt/cloud/datasets/demo",
                "source": "cloud",
                "deployable": False,
                "provider": "aliyun",
                "status": "Running",
                "job_id": "cloud-job-1",
                "task_name": "demo-act-1",
                "updated_at": "2026-05-04 13:39:00",
            }
        ]

    def start_training(self, service: EmbodiedService, **kwargs: object) -> dict[str, object]:
        self.start_calls.append(kwargs)
        return {
            "job_id": "cloud-job-1",
            "status": "Submitted",
            "running": True,
            "message": "status: Submitted\nrunning: True",
            "task_name": "demo-act-1",
            "checkpoint_path": "/mnt/cloud/checkpoints/demo",
            "dataset_path": "/mnt/cloud/datasets/demo",
            "provider": "aliyun",
        }

    def stop_training(self, **kwargs: object) -> dict[str, object]:
        self.stop_calls.append(kwargs)
        return {
            "job_id": str(kwargs.get("job_id") or ""),
            "status": "STOPPED",
            "running": False,
            "message": "status: STOPPED\nrunning: False",
            "task_name": "demo-act-1",
            "checkpoint_path": "/mnt/cloud/checkpoints/demo",
            "dataset_path": "/mnt/cloud/datasets/demo",
            "provider": "aliyun",
        }

    def current_task(self, **kwargs: object) -> dict[str, object]:
        return self.status_result

    def task_status(self, **kwargs: object) -> dict[str, object]:
        self.status_calls.append(kwargs)
        return self.status_result

    def list_policy_entries(self, **kwargs: object) -> list[dict[str, object]]:
        self.policy_calls.append(kwargs)
        return self.policy_entries

    def training_plan(self, **kwargs: object) -> dict[str, object]:
        if self.fail_next_plan:
            raise RuntimeError("EVO_Train upstream failed")
        self.plan_calls.append(kwargs)
        return {
            "message": "plan generated",
            "plan": {
                "workflow": kwargs.get("workflow") or "evf_libero",
                "readyToStart": True,
                "missingFields": [],
            },
        }

    def gpu_skus(self, **kwargs: object) -> dict[str, object]:
        self.sku_calls.append(kwargs)
        return {
            "message": "gpu sku query success",
            "skus": [{"skuId": "autodl-4090d", "readyToStart": True}],
        }

    def images(self, **kwargs: object) -> dict[str, object]:
        self.image_calls.append(kwargs)
        return {
            "message": "autodl image query success",
            "images": [{"imageId": "robotics-cu121", "readyToStart": True}],
        }

    def runtime_match(self, **kwargs: object) -> dict[str, object]:
        self.runtime_match_calls.append(kwargs)
        return {
            "message": "runtime match success",
            "readyToStart": True,
            "matches": [
                {
                    "compatible": True,
                    "score": 100,
                    "sku": {"skuId": "autodl-4090d", "gpuMemoryGb": "24"},
                    "image": {"imageId": "vla-rlinf-cu121"},
                    "reasons": ["gpu memory ok: 24GB"],
                    "blockingReasons": [],
                    "risks": [],
                }
            ],
        }


@pytest.fixture()
def isolated_roboclaw_home(tmp_path: Path):
    with patch(
        "roboclaw.embodied.embodiment.lock.get_roboclaw_home",
        return_value=tmp_path,
    ), patch(
        "roboclaw.embodied.embodiment.manifest.helpers.get_roboclaw_home",
        return_value=tmp_path,
    ):
        yield tmp_path


@pytest.fixture()
def route_app(tmp_path: Path, isolated_roboclaw_home: Path):
    app = FastAPI()
    board = Board()
    manifest = Manifest(path=tmp_path / "manifest.json", board=board)
    hw_monitor = HardwareMonitor(board=board, manifest=manifest)
    service = EmbodiedService(hardware_monitor=hw_monitor, board=board, manifest=manifest)
    app.state.embodied_service = service
    return app, service, isolated_roboclaw_home


def _write_local_policy(root: Path, name: str = "local_demo") -> Path:
    checkpoint = root / "workspace" / "embodied" / "policies" / name / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "local/demo"}, "steps": 5000}),
        encoding="utf-8",
    )
    return checkpoint


def test_train_start_uses_cloud_bridge_when_enabled(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={"dataset_name": "demo", "policy_type": "act", "steps": 5000, "username": "13800138000"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "cloud"
    assert data["job_id"] == "cloud-job-1"
    assert bridge.start_calls[0]["username"] == "13800138000"


def test_train_start_freezes_first_hour_credit_when_cost_is_declared(route_app, tmp_path):
    app, _, _ = route_app
    bridge = StubBridge()
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("13800138000", 2_000)
    train_cloud_routes.set_ledger_for_tests(ledger)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "dataset_name": "demo",
            "policy_type": "act",
            "steps": 5000,
            "username": "13800138000",
            "hourly_cost_cents": 900,
            "service_fee_bps": 1000,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["billing"]["holdCents"] == 990
    assert data["billing"]["record"]["jobId"] == "cloud-job-1"
    wallet = ledger.wallet("13800138000")
    assert wallet.balance_cents == 2_000
    assert wallet.frozen_cents == 990
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_start_rejects_when_credit_is_insufficient(route_app, tmp_path):
    app, _, _ = route_app
    bridge = StubBridge()
    train_cloud_routes.set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={"dataset_name": "demo", "username": "13800138000", "hourly_cost_cents": 900},
    )

    assert resp.status_code == 409
    assert "insufficient" in resp.json()["detail"]
    assert bridge.start_calls == []
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_cloud_billing_settle_charges_service_fee_and_releases_remainder(route_app, tmp_path):
    app, _, _ = route_app
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("13800138000", 2_000)
    ledger.freeze("13800138000", 990, job_id="cloud-job-1", task_name="demo")
    train_cloud_routes.set_ledger_for_tests(ledger)

    register_train_cloud_routes(app, app.state.embodied_service)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/billing/settle",
        json={
            "username": "13800138000",
            "job_id": "cloud-job-1",
            "provider_cost_cents": 500,
            "service_fee_bps": 1000,
            "task_name": "demo",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["chargeCents"] == 550
    assert data["releaseRecord"]["amountCents"] == 440
    assert data["wallet"]["availableCreditCents"] == 1_450
    assert data["wallet"]["frozenCreditCents"] == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_plan_forwards_skill_request_to_evo_train(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/plan",
        json={
            "username": "13800138000",
            "message": "帮我跑 LIBERO object smoke test",
            "workflow": "evf_libero",
            "params": {"suite": "libero_object_task", "epochs": 1},
            "provider": "autodl",
            "sku_id": "autodl-4090d",
            "image_id": "robotics-cu121",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "plan generated"
    assert bridge.plan_calls[0]["username"] == "13800138000"
    assert bridge.plan_calls[0]["workflow"] == "evf_libero"
    assert bridge.plan_calls[0]["sku_id"] == "autodl-4090d"


def test_train_plan_returns_503_when_evo_train_bridge_is_disabled(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.enabled = False

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/train/plan", json={"username": "13800138000", "workflow": "evf_libero"})

    assert resp.status_code == 503
    assert "bridge is not enabled" in resp.json()["detail"]


def test_train_plan_returns_502_when_evo_train_request_fails(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.fail_next_plan = True

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/train/plan", json={"username": "13800138000", "workflow": "evf_libero"})

    assert resp.status_code == 502
    assert "upstream failed" in resp.json()["detail"]


def test_evo_train_bridge_applies_timeout_to_socket_reads_and_writes():
    class FakeSocket:
        def __init__(self) -> None:
            self.timeout: float | None = None
            self.sent = b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, value: float) -> None:
            self.timeout = value

        def sendall(self, payload: bytes) -> None:
            self.sent += payload

        def recv(self, _size: int) -> bytes:
            return b'{"message":"ok"}\n'

    fake_socket = FakeSocket()
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="autodl",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
    )

    with patch("socket.create_connection", return_value=fake_socket) as create_connection:
        response = EvoTrainBridge(settings)._request({"action": "ping"})

    assert response == {"message": "ok"}
    create_connection.assert_called_once_with(("127.0.0.1", 9000), timeout=3.5)
    assert fake_socket.timeout == 3.5


def test_train_environment_catalog_routes_use_evo_train_bridge(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    skus = client.get("/api/train/gpu-skus?provider=autodl")
    images = client.get("/api/train/images")

    assert skus.status_code == 200
    assert images.status_code == 200
    assert skus.json()["skus"][0]["skuId"] == "autodl-4090d"
    assert images.json()["images"][0]["imageId"] == "robotics-cu121"
    assert bridge.sku_calls[0]["provider"] == "autodl"


def test_train_runtime_match_forwards_task_requirements(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/runtime-match",
        json={
            "username": "13800138000",
            "provider": "autodl",
            "params": {
                "backendKind": "rlinf",
                "modelFamily": "pi0",
                "benchmark": "libero",
                "minGpuMemoryGb": 24,
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["readyToStart"] is True
    assert data["matches"][0]["sku"]["skuId"] == "autodl-4090d"
    assert bridge.runtime_match_calls[0]["username"] == "13800138000"
    assert bridge.runtime_match_calls[0]["params"]["benchmark"] == "libero"


def test_train_status_falls_back_to_local_when_cloud_task_missing(route_app):
    app, service, _ = route_app
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "missing-job",
        "status": "missing",
        "running": False,
        "message": "status: missing",
        "task_name": "",
        "checkpoint_path": "",
        "dataset_path": "",
        "provider": "aliyun",
    }

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, service)

    service.train.job_status_state = AsyncMock(
        return_value=TrainingJobStatus(
            job_id="local-1",
            status="running",
            running=True,
            message="job_id: local-1\nstatus: running\nrunning: True",
            mode="local",
        )
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/status/local-1?username=13800138000")

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "local"
    assert data["running"] is True
    assert data["status"] == "running"


def test_policies_route_merges_local_and_cloud_entries(route_app):
    app, service, roboclaw_home = route_app
    bridge = StubBridge()
    _write_local_policy(roboclaw_home)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_policy_routes(app, service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/policies?username=13800138000")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    local_entry = next(item for item in data if item["source"] == "local")
    cloud_entry = next(item for item in data if item["source"] == "cloud")

    assert local_entry["deployable"] is True
    assert local_entry["dataset"] == "local/demo"
    assert cloud_entry["deployable"] is False
    assert cloud_entry["job_id"] == "cloud-job-1"
    assert bridge.policy_calls[0]["username"] == "13800138000"


def test_train_start_local_preserves_upstream_payload(route_app):
    app, service, _ = route_app
    register_train_routes(app, service)
    service.train.start_job = AsyncMock(
        return_value=TrainingJobStatus(
            job_id="local-1",
            status="running",
            running=True,
            message="Training started. Job ID: local-1",
            mode="local",
        )
    )

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/train/start", json={"dataset_name": "demo"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "local-1"
    assert data["message"] == "Training started. Job ID: local-1"
