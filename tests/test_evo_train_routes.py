from __future__ import annotations

import asyncio
import json
import time
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
from roboclaw.http.routes import cloud_repair_agent
from roboclaw.http.routes import cloud_supervisor as cloud_supervisor_state
from roboclaw.http.routes.cloud_supervisor import _set_cloud_supervisor_state
from roboclaw.http.routes.train_cloud import register_train_cloud_routes
from roboclaw.http.routes.agent_consult import register_agent_consult_routes
from roboclaw.providers.base import LLMResponse
from roboclaw.training import TrainingJobStatus


class StubBridge:
    def __init__(self) -> None:
        self.enabled = True
        self.settings = SimpleNamespace(username="default-user", provider="autodl")
        self.start_calls: list[dict[str, object]] = []
        self.stop_calls: list[dict[str, object]] = []
        self.status_calls: list[dict[str, object]] = []
        self.policy_calls: list[dict[str, object]] = []
        self.plan_calls: list[dict[str, object]] = []
        self.sku_calls: list[dict[str, object]] = []
        self.image_calls: list[dict[str, object]] = []
        self.runtime_match_calls: list[dict[str, object]] = []
        self.source_preflight_calls: list[dict[str, object]] = []
        self.provider_balance_calls: list[dict[str, object]] = []
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
        params = dict(kwargs.get("params") or {}) if isinstance(kwargs.get("params"), dict) else {}
        return {
            "message": "plan generated",
            "wallet": {
                "username": str(kwargs.get("username") or ""),
                "balanceCents": "10000",
                "availableBalanceCents": "10000",
            },
            "plan": {
                "workflow": kwargs.get("workflow") or "evf_libero",
                "params": params,
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

    def configuration_check(self, **kwargs: object) -> dict[str, object]:
        return {
            "message": "configuration check success",
            "provider": kwargs.get("provider") or "autodl",
            "ready": True,
            "mode": "managed",
            "skuCount": 2,
            "readySkuCount": 1,
            "imageCount": 2,
            "readyImageCount": 1,
            "warnings": [],
        }

    def provider_balance(self, **kwargs: object) -> dict[str, object]:
        self.provider_balance_calls.append(kwargs)
        return {
            "message": "platform balance query success",
            "provider": kwargs.get("provider") or "autodl",
            "balance": {"assets": "3660", "accumulate": "1869340", "voucherBalance": "0"},
            "minimumAssets": str(kwargs.get("minimum_assets") or 0),
            "lowBalance": False,
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

    def source_preflight(self, **kwargs: object) -> dict[str, object]:
        self.source_preflight_calls.append(kwargs)
        return {
            "message": "source preflight success",
            "source": {
                "uri": (kwargs.get("source") or {}).get("uri") if isinstance(kwargs.get("source"), dict) else "",
                "sizeKnown": False,
                "estimatedSize": "unknown",
                "requiresUserConfirmation": True,
                "risks": ["public_source_download", "cloud_cost", "license_responsibility", "unknown_size"],
            },
        }


class AutoRepairSshBridge(StubBridge):
    def __init__(self, *, failure_payloads: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self.failure_payloads = failure_payloads
        self.started_job_ids: list[str] = []

    def configuration_check(self, **kwargs: object) -> dict[str, object]:
        result = dict(super().configuration_check(**kwargs))
        result["mode"] = "ssh"
        return result

    def start_training(self, service: EmbodiedService, **kwargs: object) -> dict[str, object]:
        self.start_calls.append(kwargs)
        job_id = f"cloud-job-{len(self.start_calls)}"
        self.started_job_ids.append(job_id)
        return {
            "job_id": job_id,
            "status": "Submitted",
            "running": True,
            "message": "status: Submitted\nrunning: True",
            "task_name": str(kwargs.get("task_name") or f"cloud-task-{len(self.start_calls)}"),
            "provider": "autodl",
        }

    def task_status(self, **kwargs: object) -> dict[str, object]:
        self.status_calls.append(kwargs)
        job_id = str(kwargs.get("job_id") or "")
        if job_id in self.failure_payloads:
            return dict(self.failure_payloads[job_id])
        return {
            "job_id": job_id,
            "status": "Succeeded",
            "running": False,
            "task_name": "cloud-task",
            "provider": "autodl",
            "message": "status: Succeeded\nrunning: False",
        }


def _wait_for(predicate, *, timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


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


@pytest.fixture(autouse=True)
def reset_cloud_supervisor_snapshots():
    train_cloud_routes.clear_cloud_supervisor_snapshots_for_tests()
    yield
    train_cloud_routes.clear_cloud_supervisor_snapshots_for_tests()


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
        json={
            "dataset_name": "demo",
            "policy_type": "act",
            "steps": 5000,
            "username": "13800138000",
            "provider": "aliyun",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "cloud"
    assert data["job_id"] == "cloud-job-1"
    assert bridge.start_calls[0]["username"] == "13800138000"
    assert bridge.start_calls[0]["provider"] == "aliyun"


def test_train_cloud_bridge_status_reports_deployment_managed_connection(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/bridge")

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["managedBy"] == "Evo Studio deployment"
    assert data["userActionRequired"] is False
    assert data["deploymentMode"] == "managed"
    assert data["configurationReady"] is True
    assert data["resourceCatalog"]["readySkuCount"] == 1


def test_train_cloud_bridge_translates_stale_ssh_binding_warning(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    def stale_ssh_configuration_check(**kwargs: object) -> dict[str, object]:
        return {
            "message": "configuration check failed",
            "provider": kwargs.get("provider") or "autodl",
            "ready": False,
            "mode": "ssh",
            "sshConnectionReady": False,
            "sshGpuReady": False,
            "warnings": [
                "configured SSH instance is not reachable at the SSH protocol layer; restart or rebind the cloud instance before submitting jobs"
            ],
        }

    bridge.configuration_check = stale_ssh_configuration_check  # type: ignore[method-assign]

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/bridge")

    assert resp.status_code == 200
    data = resp.json()
    assert data["configurationReady"] is False
    assert "最新 SSH 命令" in data["configurationWarnings"][0]
    assert "configured SSH instance" in data["rawConfigurationWarnings"][0]


def test_train_cloud_bridge_auto_unbinds_stale_ssh_runtime(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    def stale_ssh_configuration_check(**kwargs: object) -> dict[str, object]:
        return {
            "message": "configuration check failed",
            "provider": kwargs.get("provider") or "autodl",
            "ready": False,
            "mode": "ssh",
            "sshConnectionReady": False,
            "sshGpuReady": False,
            "warnings": [
                "configured SSH instance is not reachable at the SSH protocol layer; restart or rebind the cloud instance before submitting jobs"
            ],
        }

    bridge.configuration_check = stale_ssh_configuration_check  # type: ignore[method-assign]

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    endpoint = {
        "endpoint": "root@connect.cqa1.seetacloud.com:30671",
        "host": "connect.cqa1.seetacloud.com",
        "port": "30671",
        "user": "root",
        "envPath": "/tmp/evo_train_env.sh",
    }
    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("roboclaw.http.routes.train_cloud._read_ssh_runtime_endpoint", return_value=endpoint),
        patch("roboclaw.http.routes.train_cloud._clear_ssh_runtime_env", return_value=endpoint) as clear_env,
        patch("roboclaw.http.routes.train_cloud._ssh_runtime_env_path", return_value=Path("/tmp/evo_train_env.sh")),
        patch(
            "roboclaw.http.routes.train_cloud._restart_local_evo_train_bridge",
            return_value={"restarted": True, "listening": True},
        ),
    ):
        resp = client.get("/api/train/cloud/bridge")

    assert resp.status_code == 200
    data = resp.json()
    assert data["autoUnboundRuntime"] is True
    assert data["previousRuntimeEndpoint"] == endpoint["endpoint"]
    assert data["runtimeEndpoint"] == ""
    assert "自动解绑" in data["configurationWarnings"][0]
    clear_env.assert_called_once()


def test_train_cloud_provider_balance_is_provider_pool_not_user_wallet(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/provider-balance", params={"provider": "autodl", "minimum_assets": 1000})

    assert resp.status_code == 200
    data = resp.json()
    assert data["balanceScope"] == "provider_pool"
    assert data["balance"]["assets"] == "3660"
    assert "User spend" in data["description"]
    assert bridge.provider_balance_calls[0]["provider"] == "autodl"
    assert bridge.provider_balance_calls[0]["minimum_assets"] == 1000


def test_train_cloud_dev_rebind_ssh_updates_local_runtime_env(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    env_file = tmp_path / "evo_train_seetacloud_env.sh"
    env_file.write_text(
        "\n".join([
            "#!/usr/bin/env zsh",
            "export AUTODL_HOST='old-host'",
            "export AUTODL_PORT='1111'",
            "export AUTODL_USER='old-user'",
            "export AUTODL_PASSWORD='old-password'",
            "export AUTODL_KEY_PATH='old-key'",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVO_TRAIN_SEETACLOUD_ENV_FILE", str(env_file))

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("roboclaw.http.routes.cloud_runtime_binding._probe_ssh_banner", return_value=(True, "")),
        patch(
            "roboclaw.http.routes.cloud_runtime_binding._restart_local_evo_train_bridge",
            return_value={"restarted": True, "pid": 1234, "listening": True},
        ),
    ):
        resp = client.post(
            "/api/train/cloud/dev/rebind-ssh",
            json={
                "sshCommand": "ssh -p 42552 root@connect.cqa1.seetacloud.com",
                "password": "new-password",
                "restartBridge": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["runtimeReady"] is True
    assert payload["host"] == "connect.cqa1.seetacloud.com"
    assert payload["port"] == "42552"
    assert payload["user"] == "root"
    text = env_file.read_text(encoding="utf-8")
    assert "export AUTODL_HOST=connect.cqa1.seetacloud.com" in text
    assert "export AUTODL_PORT=42552" in text
    assert "export AUTODL_USER=root" in text
    assert "new-password" in text
    assert "old-host" not in text


def test_train_cloud_dev_rebind_ssh_rejects_bad_banner_without_saving(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    env_file = tmp_path / "evo_train_seetacloud_env.sh"
    original_text = "\n".join([
        "#!/usr/bin/env zsh",
        "export AUTODL_HOST='old-host'",
        "export AUTODL_PORT='1111'",
        "export AUTODL_USER='old-user'",
        "export AUTODL_PASSWORD='old-password'",
        "export AUTODL_KEY_PATH='old-key'",
    ])
    env_file.write_text(original_text, encoding="utf-8")
    monkeypatch.setenv("EVO_TRAIN_SEETACLOUD_ENV_FILE", str(env_file))

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("roboclaw.http.routes.cloud_runtime_binding._probe_ssh_banner", return_value=(False, "端口已连接，但远端没有返回 SSH 登录协议。")),
        patch("roboclaw.http.routes.cloud_runtime_binding._restart_local_evo_train_bridge") as restart_mock,
    ):
        resp = client.post(
            "/api/train/cloud/dev/rebind-ssh",
            json={
                "sshCommand": "ssh -p 42552 root@connect.cqa1.seetacloud.com",
                "password": "new-password",
                "restartBridge": True,
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["saved"] is False
    assert payload["activeEndpoint"] == "old-user@old-host:1111"
    assert "没有返回 SSH 登录协议" in payload["validationError"]
    assert env_file.read_text(encoding="utf-8") == original_text
    restart_mock.assert_not_called()


def test_train_cloud_auth_connections_list_public_metadata_only(route_app, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    monkeypatch.setenv(
        "EVO_STUDIO_AUTH_REFS_JSON",
        json.dumps(
            {
                "connections": [
                    {
                        "id": "hf-team-models",
                        "kind": "model",
                        "provider": "huggingface",
                        "label": "HF team models",
                        "scope": "read-only model checkpoints",
                        "env": {"HF_TOKEN": "PRIVATE_HF_TOKEN"},
                    }
                ]
            }
        ),
    )
    monkeypatch.setenv("PRIVATE_HF_TOKEN", "hf_secret_value")

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/auth-connections?kind=model")

    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["connections"][0]["id"] == "hf-team-models"
    assert data["connections"][0]["configured"] is True
    assert "hf_secret_value" not in json.dumps(data)
    assert "PRIVATE_HF_TOKEN" not in json.dumps(data)


def test_train_cloud_can_save_private_auth_connection(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    monkeypatch.setenv("EVO_STUDIO_AUTH_REFS_FILE", str(tmp_path / "auth_refs.json"))

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    save_resp = client.post(
        "/api/train/cloud/auth-connections",
        json={
            "username": "pearl",
            "id": "hf-private-models",
            "kind": "model",
            "provider": "huggingface",
            "label": "HF private models",
            "visibility": "user",
            "secrets": {"token": "hf_private_token"},
        },
    )

    assert save_resp.status_code == 200
    saved = save_resp.json()["connection"]
    assert saved["id"] == "hf-private-models"
    assert saved["visibility"] == "user"
    assert saved["configured"] is True
    assert "hf_private_token" not in json.dumps(saved)

    list_resp = client.get("/api/train/cloud/auth-connections?kind=model&username=pearl")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["connections"][0]["id"] == "hf-private-models"
    assert "hf_private_token" not in json.dumps(data)


def test_train_cloud_start_rejects_unknown_auth_ref(route_app, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    monkeypatch.setenv("EVO_STUDIO_AUTH_REFS_JSON", json.dumps({"connections": []}))

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "dataset_name": "demo",
            "username": "13800138000",
            "params": {
                "modelSource": {
                    "sourceType": "user_object_storage",
                    "uri": "s3://private-bucket/model",
                    "authRef": "missing-model-connection",
                }
            },
        },
    )

    assert resp.status_code == 400
    assert "missing-model-connection" in json.dumps(resp.json(), ensure_ascii=False)
    assert bridge.start_calls == []


def test_train_cloud_start_accepts_configured_auth_ref(route_app, monkeypatch):
    app, _, _ = route_app
    bridge = StubBridge()
    monkeypatch.setenv(
        "EVO_STUDIO_AUTH_REFS_JSON",
        json.dumps(
            {
                "connections": [
                    {
                        "id": "team-s3-data",
                        "kind": "data",
                        "provider": "s3",
                        "env": {"AWS_ACCESS_KEY_ID": "TEAM_S3_ACCESS_KEY"},
                    }
                ]
            }
        ),
    )
    monkeypatch.setenv("TEAM_S3_ACCESS_KEY", "key")

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "dataset_name": "demo",
            "username": "13800138000",
            "params": {
                "datasetSource": {
                    "sourceType": "user_object_storage",
                    "uri": "s3://private-bucket/dataset",
                    "authRef": "team-s3-data",
                }
            },
        },
    )

    assert resp.status_code == 200
    assert bridge.start_calls[0]["params"]["datasetSource"]["authRef"] == "team-s3-data"


def test_train_cloud_start_maps_official_libero_to_public_source(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "dataset_name": "local/libero_full",
            "username": "13800138000",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "params": {
                "benchmark": "libero",
                "datasetSource": {
                    "sourceType": "public_reference",
                    "datasetId": "local/libero_full",
                },
            },
        },
    )

    assert resp.status_code == 200
    params = bridge.start_calls[0]["params"]
    assert params["datasetPath"] == "hf://HuggingFaceVLA/libero"
    assert params["datasetFormat"] == "lerobot"
    assert params["datasetSource"]["uri"] == "hf://HuggingFaceVLA/libero"
    assert params["datasetSource"]["datasetId"] == "libero"


def test_train_cloud_start_keeps_explicit_public_hf_url_over_legacy_dataset_name(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "dataset_name": "local/libero_full",
            "username": "13800138000",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "params": {
                "datasetPath": "https://huggingface.co/datasets/HuggingFaceVLA/libero",
                "datasetSource": {
                    "sourceType": "public_reference",
                    "datasetId": "libero",
                    "uri": "https://huggingface.co/datasets/HuggingFaceVLA/libero",
                    "format": "lerobot",
                },
            },
        },
    )

    assert resp.status_code == 200
    params = bridge.start_calls[0]["params"]
    assert params["datasetPath"] == "https://huggingface.co/datasets/HuggingFaceVLA/libero"
    assert params["datasetSource"]["sourceType"] == "public_reference"
    assert params["datasetSource"]["uri"] == "https://huggingface.co/datasets/HuggingFaceVLA/libero"

def test_train_cloud_start_backfills_missing_model_source_type(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "policy_type": "openvla",
            "params": {
                "datasetSource": {
                    "sourceType": "public_reference",
                    "uri": "hf://HuggingFaceVLA/libero",
                    "format": "libero",
                },
                "modelSource": {
                    "modelFamily": "openvla",
                    "uri": "hf://moojink/openvla-7b-oft-finetuned-libero-spatial",
                    "format": "huggingface_transformers",
                },
            },
        },
    )

    assert resp.status_code == 200
    params = bridge.start_calls[0]["params"]
    assert params["modelSource"]["sourceType"] == "public_model_repo"
    assert params["modelSourceKind"] == "public_model_repo"


def test_train_cloud_start_rejects_maniskill_with_libero_dataset(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "13800138000",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "policy_type": "openvla",
            "params": {
                "benchmark": "maniskill",
                "datasetSource": {
                    "sourceType": "public_reference",
                    "datasetId": "libero",
                    "uri": "hf://HuggingFaceVLA/libero",
                },
                "modelSource": {
                    "sourceType": "public_model_repo",
                    "modelFamily": "openvla",
                    "uri": "hf://openvla/openvla-7b",
                },
            },
        },
    )

    assert resp.status_code == 400
    assert "benchmark=maniskill" in resp.json()["detail"]
    assert bridge.start_calls == []


def test_train_cloud_start_rejects_unresolved_auto_model(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "13800138000",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "policy_type": "auto",
            "params": {
                "datasetSource": {
                    "sourceType": "public_reference",
                    "datasetId": "libero",
                    "uri": "hf://HuggingFaceVLA/libero",
                },
                "modelSource": {
                    "sourceType": "builtin_policy",
                    "modelFamily": "auto",
                },
            },
        },
    )

    assert resp.status_code == 400
    assert "model source is unresolved" in resp.json()["detail"]
    assert bridge.start_calls == []


def test_train_cloud_start_allows_rlinf_config_to_resolve_auto_model(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "13800138000",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "policy_type": "auto",
            "params": {
                "configName": "maniskill_ppo_openvlaoft_quickstart",
                "benchmark": "maniskill",
                "datasetSource": {
                    "sourceType": "public_reference",
                    "datasetId": "maniskill",
                    "uri": "hf://some/public-maniskill-dataset",
                },
                "modelSource": {
                    "sourceType": "builtin_policy",
                    "modelFamily": "auto",
                },
            },
        },
    )

    assert resp.status_code == 200
    params = bridge.start_calls[0]["params"]
    assert params["configName"] == "maniskill_ppo_openvlaoft_quickstart"
    assert params["repoUrl"] == "https://github.com/RLinf/RLinf.git"
    assert params["modelSource"] == {
        "sourceType": "rlinf_config_default",
        "modelFamily": "openvla-oft",
        "format": "rlinf_config",
    }
    assert not str(params.get("checkpointPath") or "").startswith("hf://moojink/openvla-7b-oft-finetuned-libero")


def test_train_start_freezes_first_hour_balance_when_cost_is_declared(route_app, tmp_path):
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


def test_train_start_prepare_only_does_not_freeze_declared_gpu_cost(route_app, tmp_path):
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
            "params": {"prepareOnly": True, "executionPhase": "prepare_only"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "billing" not in data
    assert bridge.start_calls[0]["params"]["prepareOnly"] is True
    wallet = ledger.wallet("13800138000")
    assert wallet.balance_cents == 2_000
    assert wallet.frozen_cents == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_start_releases_hold_when_bridge_start_fails(route_app, tmp_path):
    app, _, _ = route_app

    class FailingBridge(StubBridge):
        def start_training(self, service: EmbodiedService, **kwargs: object) -> dict[str, object]:
            self.start_calls.append(kwargs)
            raise RuntimeError("create task failed: unknown or disabled AutoDL skuId")

    bridge = FailingBridge()
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

    assert resp.status_code == 502
    assert "unknown or disabled AutoDL skuId" in resp.json()["detail"]
    wallet = ledger.wallet("13800138000")
    assert wallet.balance_cents == 2_000
    assert wallet.frozen_cents == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_start_rejects_when_balance_is_insufficient(route_app, tmp_path):
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


def test_train_status_releases_missing_task_hold_with_stable_task_id(route_app, tmp_path):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "task:pearl:cloud-openvla-smoke",
        "status": "missing",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
    }
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 2_000)
    ledger.freeze(
        "pearl",
        1_100,
        reason="cloud training first-hour hold",
        task_name="cloud-openvla-smoke",
        job_id="cloud-openvla-smoke",
    )
    train_cloud_routes.set_ledger_for_tests(ledger)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/status/task%3Apearl%3Acloud-openvla-smoke?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["billingRelease"]["amountCents"] == 1100
    assert data["wallet"]["frozenBalanceCents"] == 0
    assert ledger.wallet("pearl").frozen_cents == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_status_releases_successful_external_billing_hold(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("ROBOCLAW_EVO_TRAIN_BILLING_MODE", "external")
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "task:pearl:cloud-openvla-smoke",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_RLINF_METRICS_CAPTURED__=/workspace/outputs/metrics.json",
    }
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 2_000)
    ledger.freeze(
        "pearl",
        1_100,
        reason="cloud training first-hour hold",
        task_name="cloud-openvla-smoke",
        job_id="cloud-openvla-smoke",
    )
    train_cloud_routes.set_ledger_for_tests(ledger)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/status/task%3Apearl%3Acloud-openvla-smoke?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Succeeded"
    assert data["billingRelease"]["amountCents"] == 1100
    assert data["wallet"]["frozenBalanceCents"] == 0
    assert ledger.wallet("pearl").frozen_cents == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_status_releases_successful_external_sibling_repair_holds(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("ROBOCLAW_EVO_TRAIN_BILLING_MODE", "external")
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "task:pearl:cloud-openvla-smoke-intervention-222-repair-333",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke-intervention-222-repair-333",
        "provider": "autodl",
        "log_tail": "__EVO_RLINF_METRICS_CAPTURED__=/workspace/outputs/metrics.json",
    }
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 4_000)
    ledger.freeze(
        "pearl",
        1_100,
        reason="cloud training first-hour hold",
        task_name="cloud-openvla-smoke-repair-111",
        job_id="task:pearl:cloud-openvla-smoke-repair-111",
    )
    ledger.freeze(
        "pearl",
        1_100,
        reason="cloud training first-hour hold",
        task_name="cloud-openvla-smoke-intervention-222-repair-333",
        job_id="task:pearl:cloud-openvla-smoke-intervention-222-repair-333",
    )
    train_cloud_routes.set_ledger_for_tests(ledger)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(
        "/api/train/cloud/status/task%3Apearl%3Acloud-openvla-smoke-intervention-222-repair-333?username=pearl"
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Succeeded"
    assert len(data["billingReleases"]) == 2
    assert data["wallet"]["frozenBalanceCents"] == 0
    assert ledger.wallet("pearl").frozen_cents == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_status_exposes_cloud_artifact_paths(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_RLINF_METRICS_CAPTURED__=/root/autodl-tmp/RLinf/outputs/run/metrics.json",
        "log_path": "/root/autodl-tmp/evo_studio/runs/current.log",
    }

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["metricsPath"] == "/root/autodl-tmp/RLinf/outputs/run/metrics.json"
    assert data["artifacts"][0]["kind"] == "metrics"
    assert data["artifacts"][0]["path"] == "/root/autodl-tmp/RLinf/outputs/run/metrics.json"
    assert any(item["kind"] == "log" for item in data["artifacts"])


def test_train_cloud_artifacts_reads_metrics_json(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_RLINF_METRICS_CAPTURED__=/root/autodl-tmp/RLinf/outputs/run/metrics.json",
    }

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge), patch(
        "roboclaw.http.routes.train_cloud._read_remote_text_file",
        return_value='{"eval/success_once": 0.0, "eval/num_trajectories": 4}',
    ):
        register_train_cloud_routes(app, app.state.embodied_service)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/train/cloud/artifacts?username=pearl&job_id=cloud-job-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["metricsPath"] == "/root/autodl-tmp/RLinf/outputs/run/metrics.json"
    assert data["metrics"]["eval/success_once"] == 0.0
    assert data["metrics"]["eval/num_trajectories"] == 4


def test_train_cloud_artifacts_falls_back_to_log_metrics(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": (
            "[INFO] {'eval/success_at_end': array(0., dtype=float32), "
            "'eval/episode_len': array(512., dtype=float32), "
            "'eval/num_trajectories': 4}\n"
            "__EVO_RLINF_METRICS_CAPTURED__=/root/autodl-tmp/RLinf/outputs/run/metrics.json"
        ),
    }

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge), patch(
        "roboclaw.http.routes.train_cloud._read_remote_text_file",
        side_effect=RuntimeError("cloud SSH runtime is not bound"),
    ):
        register_train_cloud_routes(app, app.state.embodied_service)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/train/cloud/artifacts?username=pearl&job_id=cloud-job-1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["metricsReadError"] == "cloud SSH runtime is not bound"
    assert data["metricsSource"] == "log_tail"
    assert data["metrics"]["eval/success_at_end"] == 0
    assert data["metrics"]["eval/episode_len"] == 512
    assert data["metrics"]["eval/num_trajectories"] == 4


def test_train_current_prefers_completed_supervisor_over_stale_ssh_failure(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_FILE", str(tmp_path / "cloud_supervisor.json"))

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "error": "SSH status check failed: SSH connection failed after 4 attempt(s): Error reading SSH protocol banner",
        "log_tail": "SSH status check failed: SSH connection failed after 4 attempt(s): Error reading SSH protocol banner",
        "failureRemediation": {
            "code": "CLOUD_INSTANCE_UNREACHABLE",
            "autoRepair": {
                "safe": False,
                "strategy": "rebind_ssh_runtime_before_retry",
            },
        },
    }

    _set_cloud_supervisor_state("pearl", "cloud-job-1", {
        "state": "completed",
        "rootJobId": "cloud-job-1",
        "currentJobId": "cloud-job-1",
        "status": "Succeeded",
        "message": "任务已结束，后端总控停止观察。",
    })
    cloud_supervisor_state._cloud_supervisor_states.clear()
    cloud_supervisor_state._cloud_supervisor_states_loaded = False

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/current?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Succeeded"
    assert data["running"] is False
    assert data["error"] == ""
    assert data["message"] == "任务已结束，后端总控停止观察。"
    assert "failureRemediation" not in data
    assert data["supervisor"]["state"] == "completed"
    assert data["supervisor"]["runtime"]["state"] == "completed"
    assert data["supervisor"]["runtime"]["status"] == "Succeeded"


def test_train_current_success_overrides_stale_runtime_rebind_state(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_FILE", str(tmp_path / "cloud_supervisor.json"))

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1-repair-123456",
        "status": "Succeeded",
        "running": False,
        "task_name": "cloud-openvla-smoke-repair-123456",
        "provider": "autodl",
        "message": "status: Succeeded\nerror: Missing required value: AUTODL_HOST or AutoDL snapshot proxy_host",
        "error": "Missing required value: AUTODL_HOST or AutoDL snapshot proxy_host",
        "log_tail": "__EVO_RLINF_METRICS_CAPTURED__=/workspace/outputs/metrics.json",
    }

    _set_cloud_supervisor_state("pearl", "cloud-job-1", {
        "state": "needs_rebind",
        "rootJobId": "cloud-job-1",
        "currentJobId": "cloud-job-1-repair-123456",
        "status": "failed",
        "failureRemediation": {"code": "CLOUD_INSTANCE_UNREACHABLE"},
        "message": "当前云端实例不可达，已停止自动重试。",
    })
    cloud_supervisor_state._cloud_supervisor_states.clear()
    cloud_supervisor_state._cloud_supervisor_states_loaded = False

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/current?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Succeeded"
    assert data["running"] is False
    assert data["error"] == ""
    assert data["message"] == "任务已成功完成。"
    assert "failureRemediation" not in data
    assert data["supervisor"]["state"] == "completed"
    assert data["supervisor"]["runtime"]["state"] == "completed"
    assert data["supervisor"]["runtime"]["message"] == "任务已成功完成。"
    assert "failureRemediation" not in data["supervisor"]["runtime"]
    assert data["supervisor"]["requiresConfirmation"] is False
    assert data["autonomy"]["phase"] == "completed"
    assert data["autonomy"]["humanActionRequired"] is False


def test_train_status_marks_failed_ssh_job_as_same_runtime_repairable(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-smoke",
            "params": {
                "modelFamily": "openvla",
                "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
            },
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": "ModuleNotFoundError: No module named 'peft'",
        "failureRemediation": {
            "code": "PYTHON_IMPORT_MISSING",
            "autoRepair": {
                "safe": True,
                "strategy": "install_missing_dependency_and_retry",
            },
        },
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    supervisor = resp.json()["supervisor"]
    assert supervisor["state"] == "repairable_same_runtime"
    assert supervisor["canRetrySameRuntime"] is True
    assert supervisor["sameRuntimeAvailable"] is True


def test_cloud_watch_loop_auto_submits_safe_repair_by_default(route_app, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INTERVAL_SECONDS", "0.01")
    bridge = AutoRepairSshBridge(
        failure_payloads={
            "cloud-job-1": {
                "job_id": "cloud-job-1",
                "status": "Failed",
                "running": False,
                "task_name": "cloud-openvla-smoke",
                "provider": "autodl",
                "log_tail": "__EVO_STAGE_FAILED__=setup_env\nModuleNotFoundError: No module named 'peft'",
                "failureRemediation": {
                    "code": "PYTHON_IMPORT_MISSING",
                    "autoRepair": {"safe": True, "strategy": "install_missing_dependency_and_retry"},
                    "requiresUserConfirmationBeforeStart": False,
                },
            }
        }
    )

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/train/cloud/start",
            json={
                "username": "pearl",
                "provider": "autodl",
                "workflow": "vla_rl_backend",
                "task_name": "cloud-openvla-smoke",
                "params": {
                    "modelFamily": "openvla",
                    "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["supervisor"]["nextAction"] == "auto_retry_same_runtime"
        assert _wait_for(lambda: len(bridge.start_calls) >= 2)

    assert len(bridge.start_calls) == 2
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["repairOfJobId"] == "cloud-job-1"
    assert retry_params["forceSkipStageCache"] is True
    assert "peft" in " ".join(retry_params["repairBootstrapCommands"])


def test_cloud_watch_loop_does_not_repair_confirmation_required_failure(route_app, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INTERVAL_SECONDS", "0.01")
    bridge = AutoRepairSshBridge(
        failure_payloads={
            "cloud-job-1": {
                "job_id": "cloud-job-1",
                "status": "Failed",
                "running": False,
                "task_name": "cloud-openvla-smoke",
                "provider": "autodl",
                "log_tail": "__EVO_STAGE_FAILED__=preflight\n__EVO_GPU_UNAVAILABLE__=cuda",
                "failureRemediation": {
                    "code": "CLOUD_GPU_UNAVAILABLE",
                    "autoRepair": {"safe": False, "strategy": "rebind_ssh_runtime_before_retry"},
                    "requiresUserConfirmationBeforeStart": True,
                },
            }
        }
    )

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/train/cloud/start",
            json={
                "username": "pearl",
                "provider": "autodl",
                "workflow": "vla_rl_backend",
                "task_name": "cloud-openvla-smoke",
                "params": {
                    "modelFamily": "openvla",
                    "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
                },
            },
        )
        assert resp.status_code == 200
        assert _wait_for(lambda: bool(cloud_supervisor_state._cloud_supervisor_states), timeout=0.4)

    assert len(bridge.start_calls) == 1


def test_cloud_watch_loop_caps_safe_auto_repairs_at_three(route_app, monkeypatch):
    app, _, _ = route_app
    monkeypatch.delenv("EVO_STUDIO_CLOUD_SUPERVISOR_MAX_REPAIRS", raising=False)
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INTERVAL_SECONDS", "0.01")
    bridge = AutoRepairSshBridge(
        failure_payloads={
            f"cloud-job-{idx}": {
                "job_id": f"cloud-job-{idx}",
                "status": "Failed",
                "running": False,
                "task_name": "cloud-openvla-smoke",
                "provider": "autodl",
                "log_tail": f"__EVO_STAGE_FAILED__=setup-env-{idx}\nModuleNotFoundError: No module named 'peft'",
                "failureRemediation": {
                    "code": "PYTHON_IMPORT_MISSING",
                    "autoRepair": {"safe": True, "strategy": "install_missing_dependency_and_retry"},
                    "requiresUserConfirmationBeforeStart": False,
                },
            }
            for idx in range(1, 5)
        }
    )

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/train/cloud/start",
            json={
                "username": "pearl",
                "provider": "autodl",
                "workflow": "vla_rl_backend",
                "task_name": "cloud-openvla-smoke",
                "params": {
                    "modelFamily": "openvla",
                    "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
                },
            },
        )
        assert resp.status_code == 200
        assert _wait_for(lambda: len(bridge.start_calls) >= 4, timeout=4.5)
        time.sleep(0.08)

    assert len(bridge.start_calls) == 4
    runtime_state = next(iter(cloud_supervisor_state._cloud_supervisor_states.values()))
    assert runtime_state["repairAttempts"] == 3
    assert runtime_state["maxRepairs"] == 3


def test_cloud_watch_loop_oom_intervention_halves_batch_size(route_app, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INITIAL_DELAY_SECONDS", "0")
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_INTERVAL_SECONDS", "0.01")
    bridge = AutoRepairSshBridge(
        failure_payloads={
            "cloud-job-1": {
                "job_id": "cloud-job-1",
                "status": "Running",
                "running": True,
                "task_name": "cloud-openvla-smoke",
                "provider": "autodl",
                "log_tail": "RuntimeError: CUDA out of memory. Tried to allocate 8.00 GiB",
            }
        }
    )

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/train/cloud/start",
            json={
                "username": "pearl",
                "provider": "autodl",
                "workflow": "vla_rl_backend",
                "task_name": "cloud-openvla-smoke",
                "params": {
                    "modelFamily": "openvla",
                    "batchSize": 8,
                    "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
                },
            },
        )
        assert resp.status_code == 200
        assert _wait_for(lambda: len(bridge.start_calls) >= 2)

    intervention_params = bridge.start_calls[-1]["params"]
    assert bridge.stop_calls[-1]["job_id"] == "cloud-job-1"
    assert intervention_params["batchSize"] == 4
    assert intervention_params["resumeFromCheckpoint"] is True
    assert intervention_params["repairStrategy"] == "halve_batch_size_and_resume"


def test_train_status_does_not_show_stale_watching_runtime_for_failed_current_job(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()
    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "error": "SSH status check failed: Error reading SSH protocol banner",
        "failureRemediation": {
            "code": "CLOUD_INSTANCE_UNREACHABLE",
            "autoRepair": {
                "safe": False,
                "strategy": "rebind_ssh_runtime_before_retry",
            },
        },
    }
    _set_cloud_supervisor_state("pearl", "cloud-job-1", {
        "state": "watching",
        "rootJobId": "cloud-job-1",
        "currentJobId": "cloud-job-1",
        "status": "Running",
        "message": "stale runtime state",
    })

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Failed"
    assert data["supervisor"]["state"] != "watching"
    assert "runtime" not in data["supervisor"]
    assert data["autonomy"]["phase"] == "blocked"
    assert data["autonomy"]["loop"] == "rebind_runtime"
    assert data["autonomy"]["blockerCode"] == "CLOUD_INSTANCE_UNREACHABLE"
    assert data["autonomy"]["humanActionRequired"] is True


def test_cloud_supervisor_injects_repair_commands_for_known_remediations():
    cases = [
        (
            {"code": "PYTHON_IMPORT_MISSING"},
            "__EVO_OPENVLA_OFT_RUNTIME_UNAVAILABLE__=ModuleNotFoundError: Could not import module 'PreTrainedModel'",
            "__EVO_OPENVLA_OFT_RUNTIME_REPAIR__",
        ),
        (
            {"code": "PYTHON_IMPORT_MISSING"},
            "ModuleNotFoundError: No module named 'libero'",
            "__EVO_LIBERO_RUNTIME_REPAIR__",
        ),
        (
            {"code": "CLOUD_GPU_UNAVAILABLE"},
            "CUDA driver version is insufficient for CUDA runtime version; torch_cuda=13.0 cuda_version_mismatch",
            "__EVO_TORCH_CUDA_REPAIR__",
        ),
        (
            {"code": "PYTHON_DEPENDENCY_RESOLUTION_FAILED"},
            "ModuleNotFoundError: No module named 'gym'",
            "gym",
        ),
    ]
    for remediation, log_text, expected in cases:
        params = cloud_supervisor_state._inject_repair_commands({}, remediation, log_text)
        joined = " ".join(params["repairBootstrapCommands"])
        assert expected in joined
        assert params["forceRepairBootstrap"] is True
        assert params["forceSkipStageCache"] is True

    terminated = cloud_supervisor_state._inject_repair_commands(
        {},
        {"code": "CLOUD_STAGE_TERMINATED", "stage": "train_rlinf_vla"},
        "__EVO_STAGE_FAILED__=train_rlinf_vla\nKilled",
    )
    assert terminated["resumeFromStage"] == "train_rlinf_vla"
    assert "repairBootstrapCommands" not in terminated


def test_cloud_supervisor_classifies_missing_workdir_as_prepare_retry():
    log_tail = "bash: 第 1 行： cd: /root/autodl-tmp/RLinf: 没有那个文件或目录"

    remediation = cloud_supervisor_state._infer_cloud_failure_remediation({
        "status": "Failed",
        "log_tail": log_tail,
    })

    assert remediation["code"] == "CLOUD_WORKDIR_MISSING"
    assert remediation["autoRepair"]["safe"] is True
    assert remediation["autoRepair"]["strategy"] == "rerun_prepare_code_on_same_runtime"

    params = cloud_supervisor_state._inject_repair_commands(
        {"skipPrepareCode": True},
        remediation,
        log_tail,
    )
    assert params["forceSkipStageCache"] is True
    assert params["skipPrepareCode"] is False
    assert params["resumeFromStage"] == "prepare_code"
    assert "repairBootstrapCommands" not in params


def test_cloud_supervisor_defaults_to_safe_same_runtime_repair():
    policy = cloud_supervisor_state._auto_repair_policy({})

    assert policy["mode"] == "safe_auto"
    assert policy["autoRetrySameRuntime"] is True
    assert policy["allowAgentRepairSameRuntime"] is True
    assert policy["paidStartRequiresConfirmation"] is True


def test_cloud_start_policy_defaults_safe_auto_but_full_auto_bypasses_paid_gate():
    safe_policy = train_cloud_routes._resolve_automation_policy(None)
    assert safe_policy["mode"] == "safe_auto"
    assert safe_policy["autoRetrySameRuntime"] is True
    assert safe_policy["allowAgentRepairSameRuntime"] is True
    assert safe_policy["paidStartRequiresConfirmation"] is True

    full_policy = train_cloud_routes._resolve_automation_policy({}, "full_auto")
    assert full_policy["mode"] == "full_auto"
    assert full_policy["autoRetrySameRuntime"] is True
    assert full_policy["allowAgentRepairSameRuntime"] is True
    assert full_policy["paidStartRequiresConfirmation"] is False

    explicit_policy = train_cloud_routes._resolve_automation_policy(
        {"mode": "full_auto", "paidStartRequiresConfirmation": True}
    )
    assert explicit_policy["paidStartRequiresConfirmation"] is True


def test_cloud_supervisor_uses_configured_llm_for_unknown_repair():
    class RepairProvider:
        async def chat_with_retry(self, messages, **kwargs):  # noqa: ANN001, ANN003
            assert "failed ML training log" in messages[0]["content"]
            assert "mystery package exploded" in messages[1]["content"]
            return LLMResponse(content='{"command":"python -m pip install --upgrade gymnasium"}')

    params = asyncio.run(
        cloud_supervisor_state._inject_repair_commands_async(
            {},
            {"code": "UNKNOWN_CLOUD_FAILURE"},
            "__EVO_STAGE_FAILED__=setup_env\nmystery package exploded",
            llm_provider=RepairProvider(),
        )
    )

    assert params["forceRepairBootstrap"] is True
    assert params["forceSkipStageCache"] is True
    assert params["repairBootstrapCommands"] == ["python -m pip install --upgrade gymnasium"]


def test_cloud_repair_agent_rejects_unsafe_llm_command():
    assert cloud_repair_agent.repair_command_from_llm_content('{"command":"rm -rf /"}') == ""
    assert cloud_repair_agent.repair_command_from_llm_content('{"command":"&& python -m pip install gym"}') == ""
    assert cloud_repair_agent.repair_command_from_llm_content('{"command":"python -m pip install gym &&"}') == ""
    assert cloud_repair_agent.repair_command_from_llm_content('{"command":"I would install gym with pip"}') == ""
    assert cloud_repair_agent.repair_command_from_llm_content("I would install gym with pip") == ""


def test_cloud_repair_agent_records_audit_event(tmp_path, monkeypatch):
    audit_path = tmp_path / "repair_audit.jsonl"
    monkeypatch.setenv("EVO_STUDIO_CLOUD_REPAIR_AUDIT_FILE", str(audit_path))

    cloud_repair_agent.record_repair_agent_event({"event": "repair_decided", "failureCode": "UNKNOWN"})

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "evo_studio_cloud_repair_agent_event/v1"
    assert payload["event"] == "repair_decided"
    assert payload["failureCode"] == "UNKNOWN"


def test_cloud_supervisor_oom_intervention_halves_batch_size():
    intervention = {
        "code": "TRAINING_OOM",
        "strategy": "halve_batch_size_and_resume",
        "summary": "CUDA out of memory detected during training.",
    }

    patched = cloud_supervisor_state._apply_training_intervention_params(
        {
            "batchSize": 8,
            "trainingContract": {
                "runner": {"batch_size": 16},
                "algorithm": {"learning_rate": 1e-4},
            },
        },
        intervention,
    )

    assert patched["batchSize"] == 4
    assert patched["trainingContract"]["runner"]["batch_size"] == 16
    assert patched["resumeFromCheckpoint"] is True
    assert patched["failureRemediation"]["code"] == "TRAINING_OOM"


def test_cloud_supervisor_ray_memory_oom_intervention_reduces_rollout_parallelism():
    intervention = {
        "code": "TRAINING_NODE_MEMORY_OOM",
        "strategy": "reduce_rollout_parallelism_and_resume",
        "summary": "Ray killed rollout workers because node memory usage exceeded the safety threshold.",
    }

    patched = cloud_supervisor_state._apply_training_intervention_params(
        {
            "numRolloutWorkers": 8,
            "trainingContract": {"runner": {"num_workers": 6, "num_envs": 4}},
            "overrides": ["algorithm.name=grpo"],
        },
        intervention,
    )

    assert patched["numRolloutWorkers"] == 4
    assert patched["trainingContract"]["runner"]["num_workers"] == 3
    assert patched["trainingContract"]["runner"]["num_envs"] == 2
    assert "runner.num_workers=1" in patched["overrides"]
    assert "rollout.num_envs=1" in patched["overrides"]
    assert patched["env"]["RAY_memory_usage_threshold"] == "0.98"
    assert patched["resumeFromCheckpoint"] is True
    assert patched["failureRemediation"]["code"] == "TRAINING_NODE_MEMORY_OOM"


def test_cloud_supervisor_ray_memory_oom_classifies_as_training_intervention():
    payload = {
        "status": "Running",
        "running": True,
        "log_tail": (
            "Exception occurred while running MultiStepRolloutWorker. "
            "1 worker(s) were killed due to the node running low on memory. "
            "Memory on the node was 87.23GB / 90.00GB, which exceeds the memory usage threshold."
        ),
    }

    intervention = cloud_supervisor_state._training_time_intervention(payload)

    assert intervention["code"] == "TRAINING_NODE_MEMORY_OOM"
    assert intervention["strategy"] == "reduce_rollout_parallelism_and_resume"


def test_cloud_supervisor_truncated_ray_memory_oom_classifies_as_training_intervention():
    payload = {
        "status": "Running",
        "running": True,
        "log_tail": (
            "Memory usage threshold exceeded. To see more information about memory usage on this node, "
            "use `ray logs raylet.out -ip 172.17.0.14`; Top 10 memory users: PID MEM(GB) COMMAND."
        ),
    }

    intervention = cloud_supervisor_state._training_time_intervention(payload)

    assert intervention["code"] == "TRAINING_NODE_MEMORY_OOM"
    assert intervention["strategy"] == "reduce_rollout_parallelism_and_resume"


def test_cloud_supervisor_ray_gcs_timeout_classifies_as_training_intervention():
    payload = {
        "status": "Running",
        "running": True,
        "log_tail": (
            "gcs_client.cc:205: Failed to get cluster ID from GCS server: TimedOut: "
            "Timed out while waiting for GCS to become available.\n"
            "rpc_client.h:153: Failed to connect to GCS at address 172.17.0.17:39441 within 5 seconds."
        ),
    }

    intervention = cloud_supervisor_state._training_time_intervention(payload)

    assert intervention["code"] == "RAY_GCS_UNAVAILABLE"
    assert intervention["strategy"] == "restart_ray_runtime_and_resume"


def test_cloud_supervisor_ray_gcs_intervention_resets_ray_runtime():
    intervention = {
        "code": "RAY_GCS_UNAVAILABLE",
        "strategy": "restart_ray_runtime_and_resume",
        "summary": "Ray GCS did not become available during training startup.",
    }

    patched = cloud_supervisor_state._apply_training_intervention_params(
        {
            "numRolloutWorkers": 4,
            "trainingContract": {"runner": {"num_workers": 4, "num_envs": 2}},
            "overrides": ["algorithm.name=grpo"],
        },
        intervention,
    )

    repair_commands = "\n".join(patched["repairBootstrapCommands"])
    assert "ray" in repair_commands
    assert "stop" in repair_commands
    assert "/tmp/ray" in repair_commands
    assert patched["numRolloutWorkers"] == 2
    assert patched["trainingContract"]["runner"]["num_workers"] == 2
    assert patched["trainingContract"]["runner"]["num_envs"] == 1
    assert "runner.num_workers=1" in patched["overrides"]
    assert patched["env"]["RAY_DEDUP_LOGS"] == "0"
    assert patched["env"]["RAY_memory_usage_threshold"] == "0.98"
    assert patched["forceRepairBootstrap"] is True
    assert patched["forceSkipStageCache"] is True
    assert patched["resumeFromCheckpoint"] is True
    assert patched["failureRemediation"]["code"] == "RAY_GCS_UNAVAILABLE"


def test_train_status_treats_stopped_stage_failure_as_repairable(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-smoke",
            "params": {"modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Stopped",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_STAGE_FAILED__=setup_env\nTerminated",
        "message": "status: Stopped\nrunning: False",
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Failed"
    assert data["failureRemediation"]["code"] == "CLOUD_STAGE_TERMINATED"
    assert data["failureRemediation"]["stage"] == "setup_env"
    assert data["supervisor"]["state"] == "repairable_same_runtime"
    assert data["supervisor"]["canRetrySameRuntime"] is True


def test_train_status_classifies_python_version_mismatch_as_same_runtime_repairable(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-rlinf-smoke",
            "params": {"backendKind": "rlinf", "modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Stopped",
        "running": False,
        "task_name": "cloud-rlinf-smoke",
        "provider": "autodl",
        "log_tail": "ERROR: Package 'rlinf' requires a different Python: 3.12.3 not in '<=3.11.14,>=3.10'\n__EVO_STAGE_FAILED__=setup_env",
        "message": "status: Stopped\nrunning: False",
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Failed"
    assert data["failureRemediation"]["code"] == "PYTHON_VERSION_INCOMPATIBLE"
    assert data["failureRemediation"]["autoRepair"]["strategy"] == "switch_to_python_3_11_runtime_and_retry"
    assert data["supervisor"]["state"] == "repairable_same_runtime"


def test_train_status_classifies_missing_libero_assets_as_same_runtime_repairable(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "task_name": "cloud-rlinf-libero-smoke",
            "params": {"backendKind": "rlinf", "benchmark": "libero"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-rlinf-libero-smoke",
        "provider": "autodl",
        "log_tail": (
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/site-packages/libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml'\n"
            "__EVO_STAGE_FAILED__=train_rlinf_vla"
        ),
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["failureRemediation"]["code"] == "LIBERO_ASSETS_MISSING"
    assert data["failureRemediation"]["autoRepair"]["strategy"] == "repair_libero_assets_and_retry"
    assert data["supervisor"]["state"] == "repairable_same_runtime"


def test_train_status_overrides_stale_unknown_remediation_when_python_mismatch_is_detected(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-rlinf-smoke",
            "params": {"backendKind": "rlinf", "modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-rlinf-smoke",
        "provider": "autodl",
        "log_tail": "ERROR: Package 'rlinf' requires a different Python: 3.12.3 not in '<=3.11.14,>=3.10'\n__EVO_STAGE_FAILED__=setup_env",
        "failureRemediation": {
            "code": "UNKNOWN_CLOUD_FAILURE",
            "summary": "The cloud task failed with an unclassified error.",
        },
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["failureRemediation"]["code"] == "PYTHON_VERSION_INCOMPATIBLE"
    assert data["failureRemediation"]["autoRepair"]["safe"] is True
    assert data["supervisor"]["state"] == "repairable_same_runtime"


def test_train_supervisor_repair_reuses_previous_start_payload_on_ssh_runtime(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-smoke",
            "hourly_cost_cents": 1000,
            "params": {
                "modelFamily": "openvla",
                "bootstrapCommands": ["bad stale command"],
                "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
            },
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "error": "ModuleNotFoundError: No module named 'peft'",
        "failureRemediation": {
            "code": "PYTHON_IMPORT_MISSING",
            "autoRepair": {
                "safe": True,
                "strategy": "install_missing_dependency_and_retry",
            },
        },
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "safe_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["supervisor"]["state"] == "repair_submitted"
    assert data["supervisor"]["autoStarted"] is True
    assert len(bridge.start_calls) == 2
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["repairOfJobId"] == "cloud-job-1"
    assert retry_params["repairStrategy"] == "install_missing_dependency_and_retry"
    assert retry_params["forceRepairBootstrap"] is True
    assert retry_params["forceSkipStageCache"] is True
    assert "peft" in " ".join(retry_params["repairBootstrapCommands"])
    assert "bootstrapCommands" not in retry_params
    assert "sourceResolutions" not in retry_params
    assert str(bridge.start_calls[-1]["task_name"]).startswith("cloud-openvla-smoke-repair-")


def test_train_supervisor_classifies_python_dependency_conflicts(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-oft-smoke",
            "params": {"modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Stopped",
        "running": False,
        "task_name": "cloud-openvla-oft-smoke",
        "provider": "autodl",
        "log_tail": "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
    }

    resp = client.get("/api/train/cloud/status/cloud-job-1?username=pearl")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Failed"
    assert data["failureRemediation"]["code"] == "PYTHON_DEPENDENCY_RESOLUTION_FAILED"
    assert data["failureRemediation"]["autoRepair"]["safe"] is True
    assert "dependency" in data["failureRemediation"]["summary"].lower()
    assert data["supervisor"]["state"] == "repairable_same_runtime"


def test_train_supervisor_repair_patches_known_python_dependency_conflict(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-oft-smoke",
            "params": {
                "modelFamily": "openvla",
                "bootstrapCommands": ["stale setup command"],
            },
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Stopped",
        "running": False,
        "task_name": "cloud-openvla-oft-smoke",
        "provider": "autodl",
        "log_tail": "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'",
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "full_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["repairOfJobId"] == "cloud-job-1"
    assert retry_params["repairStrategy"] == "patch_dependency_constraints_and_retry"
    assert retry_params["forceRepairBootstrap"] is True
    assert retry_params["forceSkipStageCache"] is True
    repair_commands = " ".join(retry_params["repairBootstrapCommands"])
    assert "protobuf>=3.20.3,<5" in repair_commands
    assert "__EVO_PYTHON_DEPENDENCY_REPAIR__=protobuf_compat" in repair_commands
    assert "bootstrapCommands" not in retry_params
    assert "stale setup command" not in str(retry_params)


def test_tensorflow_python_dependency_repair_is_version_agnostic():
    command = cloud_repair_agent._python_dependency_repair_command(
        "ERROR: No matching distribution found for tensorflow==2.14.1"
    )

    assert "__EVO_PYTHON_DEPENDENCY_REPAIR__=tensorflow_python_compat" in command
    assert "sys.version_info >= (3, 12)" in command
    assert "tensorflow>=2.16,<2.22" in command
    assert "tensorflow==2.15.0" not in command


def test_train_supervisor_repair_uses_llm_for_unknown_dependency_conflict(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    class RepairProvider:
            async def chat_with_retry(self, *_args, **_kwargs):
                return LLMResponse(
                    content=json.dumps({"command": "python -m pip install --upgrade package-from-llm"}),
                )

    app.state.llm_provider = RepairProvider()
    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=True)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-generic-dependency-smoke",
            "params": {"modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-generic-dependency-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_STAGE_FAILED__=setup_env\nERROR: No matching distribution found for unknown-framework==9.9.9",
        "failureRemediation": {
            "code": "PYTHON_DEPENDENCY_RESOLUTION_FAILED",
            "autoRepair": {
                "safe": True,
                "strategy": "patch_dependency_constraints_and_retry",
            },
        },
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "full_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["forceRepairBootstrap"] is True
    assert retry_params["forceSkipStageCache"] is True
    repair_commands = " ".join(retry_params["repairBootstrapCommands"])
    assert "package-from-llm" in repair_commands


def test_train_supervisor_repair_keeps_metric_collection_retry_lightweight(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "rlinf_vla",
            "task_name": "cloud-openvla-oft-smoke",
            "params": {
                "modelFamily": "openvla",
                "bootstrapProfile": "openvla_oft_libero",
                "bootstrapCommands": ["very long stale bootstrap command"],
                "bootstrapProfileSpec": {"commands": ["very long inline bootstrap spec"]},
                "healthcheckCommands": ["very long healthcheck"],
                "sourceResolutions": {"dataset": {"resolvedPath": "/root/cache/dataset"}},
            },
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-oft-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_STAGE_FAILED__=train_rlinf_vla\n[INFO RLinf] {'eval/success_once': array(0., dtype=float32)}",
        "failureRemediation": {
            "code": "CLOUD_METRICS_MISSING",
            "autoRepair": {
                "safe": True,
                "strategy": "inspect_outputs_and_retry_metric_collection_or_eval",
            },
        },
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "full_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["repairOfJobId"] == "cloud-job-1"
    assert retry_params["repairStrategy"] == "inspect_outputs_and_retry_metric_collection_or_eval"
    assert retry_params["forceRepairBootstrap"] is False
    assert retry_params["setupCommand"] == "true"
    assert retry_params["bootstrapProfile"] == ""
    assert retry_params["bootstrapCommands"] == []
    assert retry_params["healthcheckCommands"] == []
    assert retry_params["skipPrepareCode"] is True
    assert retry_params["skipSetupEnv"] is True
    assert retry_params["skipSourceResolve"] is True
    assert retry_params["skipWriteContract"] is True
    assert "bootstrapProfileSpec" not in retry_params
    assert "sourceResolutions" not in retry_params


def test_train_supervisor_repair_ignores_stale_watching_state_after_runtime_rebind(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            result["deploymentMode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-smoke",
            "hourly_cost_cents": 1000,
            "params": {"modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "error": "SSH status check failed: Error reading SSH protocol banner",
        "failureRemediation": {
            "code": "CLOUD_INSTANCE_UNREACHABLE",
            "summary": "The configured SSH instance is no longer reachable.",
            "autoRepair": {
                "safe": False,
                "strategy": "rebind_ssh_runtime_before_retry",
            },
        },
    }
    _set_cloud_supervisor_state("pearl", "cloud-job-1", {
        "state": "watching",
        "rootJobId": "cloud-job-1",
        "currentJobId": "cloud-job-1",
        "status": "Running",
        "message": "stale watcher from the old SSH endpoint",
    })

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {
                "mode": "full_auto",
                "autoRetrySameRuntime": True,
                "allowAgentRepairSameRuntime": True,
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["supervisor"]["state"] == "repair_submitted"
    assert len(bridge.start_calls) == 2
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["restartOfJobId"] == "cloud-job-1"
    assert retry_params["repairStrategy"] == "restart_after_runtime_rebind"


def test_train_supervisor_repair_pins_transformers_for_openvla_oft_runtime_sentinel(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-oft-smoke",
            "params": {
                "modelFamily": "openvla",
                "bootstrapCommands": ["bad stale command"],
            },
        },
    )
    assert start_resp.status_code == 200

    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-oft-smoke",
        "provider": "autodl",
        "log_tail": "__EVO_OPENVLA_OFT_RUNTIME_UNAVAILABLE__=ModuleNotFoundError: Could not import module 'PreTrainedModel'",
        "failureRemediation": {
            "code": "PYTHON_MODULE_MISSING",
            "autoRepair": {
                "safe": True,
                "strategy": "install_missing_dependency_and_retry",
            },
        },
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "safe_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    retry_params = bridge.start_calls[-1]["params"]
    assert retry_params["repairOfJobId"] == "cloud-job-1"
    assert retry_params["forceRepairBootstrap"] is True
    assert retry_params["forceSkipStageCache"] is True
    repair_commands = " ".join(retry_params["repairBootstrapCommands"])
    assert "__EVO_OPENVLA_OFT_RUNTIME_REPAIR__" in repair_commands
    assert "transformers==4.40.2" in repair_commands
    assert "bootstrapCommands" not in retry_params
    assert "bad stale command" not in str(retry_params)


def test_train_supervisor_repair_loads_start_snapshot_after_backend_restart(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_FILE", str(tmp_path / "cloud_supervisor.json"))

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    start_resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-openvla-smoke",
            "params": {"modelFamily": "openvla"},
        },
    )
    assert start_resp.status_code == 200
    assert (tmp_path / "cloud_supervisor.json").exists()

    train_cloud_routes._cloud_start_snapshots.clear()
    train_cloud_routes._cloud_start_snapshots_loaded = False
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "failureRemediation": {
            "code": "PYTHON_IMPORT_MISSING",
            "autoRepair": {"safe": True, "strategy": "install_missing_dependency_and_retry"},
        },
    }

    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "safe_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 200
    assert len(bridge.start_calls) == 2
    assert bridge.start_calls[-1]["params"]["repairOfJobId"] == "cloud-job-1"


def test_train_supervisor_snapshot_redacts_accidental_secrets(route_app, tmp_path, monkeypatch):
    app, _, _ = route_app
    snapshot_path = tmp_path / "cloud_supervisor.json"
    monkeypatch.setenv("EVO_STUDIO_CLOUD_SUPERVISOR_FILE", str(snapshot_path))
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/start",
        json={
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "task_name": "cloud-secret-smoke",
            "params": {
                "modelFamily": "openvla",
                "modelSource": {
                    "sourceType": "public_model_repo",
                    "uri": "hf://example/model",
                    "token": "hf_should_not_be_written",
                },
                "objectStorage": {
                    "accessKeyId": "ak_should_not_be_written",
                    "secretAccessKey": "sk_should_not_be_written",
                },
            },
        },
    )

    assert resp.status_code == 200
    raw = snapshot_path.read_text(encoding="utf-8")
    assert "hf_should_not_be_written" not in raw
    assert "ak_should_not_be_written" not in raw
    assert "sk_should_not_be_written" not in raw
    assert raw.count("***") >= 3


def test_train_supervisor_repair_requires_snapshot_before_retry(route_app):
    app, _, _ = route_app

    class SshBridge(StubBridge):
        def configuration_check(self, **kwargs: object) -> dict[str, object]:
            result = dict(super().configuration_check(**kwargs))
            result["mode"] = "ssh"
            return result

    bridge = SshBridge()
    bridge.status_result = {
        "job_id": "cloud-job-1",
        "status": "Failed",
        "running": False,
        "task_name": "cloud-openvla-smoke",
        "provider": "autodl",
        "failureRemediation": {
            "code": "PYTHON_IMPORT_MISSING",
            "autoRepair": {"safe": True, "strategy": "install_missing_dependency_and_retry"},
        },
    }

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/cloud/supervisor/repair",
        json={
            "username": "pearl",
            "jobId": "cloud-job-1",
            "automationPolicy": {"mode": "safe_auto", "autoRetrySameRuntime": True},
        },
    )

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "supervisor_repair_requires_review"
    assert detail["supervisor"]["hasStartSnapshot"] is False
    assert bridge.start_calls == []


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
    assert data["wallet"]["availableBalanceCents"] == 1_450
    assert data["wallet"]["frozenBalanceCents"] == 0
    train_cloud_routes.set_ledger_for_tests(None)


def test_train_plan_forwards_skill_request_to_evo_train(route_app):
    app, _, _ = route_app
    bridge = StubBridge()
    ledger = AccountLedger(Path(route_app[2]) / "ledger.json")
    ledger.admin_recharge("13800138000", 2_000)
    train_cloud_routes.set_ledger_for_tests(ledger)

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    try:
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
        assert data["wallet"]["balanceCents"] == 2_000
        assert data["wallet"]["availableBalanceCents"] == 2_000
        assert data["executorWallet"]["balanceCents"] == "10000"
        assert data["billingMode"] == "external"
        assert bridge.plan_calls[0]["username"] == "13800138000"
        assert bridge.plan_calls[0]["workflow"] == "evf_libero"
        assert bridge.plan_calls[0]["sku_id"] == "autodl-4090d"
    finally:
        train_cloud_routes.set_ledger_for_tests(None)


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


def test_evo_train_bridge_adds_external_billing_to_start_payload():
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="mock",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
    )
    bridge = EvoTrainBridge(settings)
    captured: list[dict[str, object]] = []

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        return {
            "message": "training accepted",
            "tasks": [
                {
                    "taskName": "mock-task",
                    "jobId": "task:pearl:mock-task",
                    "status": "Submitting",
                    "provider": "mock",
                }
            ],
        }

    bridge._request = fake_request  # type: ignore[method-assign]

    result = bridge.start_training(
        None,
        provider="aliyun",
        workflow="custom_project",
        params={"repoUrl": "https://example.invalid/repo.git", "trainCommand": "true"},
        task_name="mock-task",
        device="cuda",
        username="pearl",
    )

    assert result["job_id"] == "task:pearl:mock-task"
    assert captured[0]["provider"] == "aliyun"
    assert captured[0]["billingMode"] == "external"
    assert captured[0]["waitForSubmit"] is True


def test_evo_train_bridge_backfills_model_source_contract_before_start():
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="mock",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
    )
    bridge = EvoTrainBridge(settings)
    captured: list[dict[str, object]] = []

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        return {
            "message": "training accepted",
            "tasks": [
                {
                    "taskName": "mock-task",
                    "jobId": "task:pearl:mock-task",
                    "status": "Submitting",
                    "provider": "mock",
                }
            ],
        }

    bridge._request = fake_request  # type: ignore[method-assign]

    bridge.start_training(
        None,
        provider="mock",
        workflow="rlinf_vla",
        params={
            "modelSource": {
                "uri": "hf://moojink/openvla-7b-oft-finetuned-libero-spatial",
                "modelFamily": "openvla",
            },
            "sourceContract": {"modelSource": {"modelFamily": "auto"}},
            "trainingContract": {"sources": {"model": {"modelFamily": "auto"}}},
        },
        task_name="mock-task",
        device="cuda",
        username="pearl",
    )

    params = captured[0]["params"]
    assert isinstance(params, dict)
    assert params["modelSource"]["sourceType"] == "public_model_repo"
    assert params["modelSourceKind"] == "public_model_repo"
    assert params["sourceContract"]["modelSource"]["sourceType"] == "public_model_repo"
    assert params["trainingContract"]["sources"]["model"]["sourceType"] == "public_model_repo"


def test_evo_train_bridge_maps_rlinf_model_source_alias_before_start():
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="mock",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
    )
    bridge = EvoTrainBridge(settings)
    captured: list[dict[str, object]] = []

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        return {
            "message": "training accepted",
            "tasks": [
                {
                    "taskName": "mock-task",
                    "jobId": "task:pearl:mock-task",
                    "status": "Submitting",
                    "provider": "mock",
                }
            ],
        }

    bridge._request = fake_request  # type: ignore[method-assign]

    bridge.start_training(
        None,
        provider="mock",
        workflow="rlinf_vla",
        params={
            "configName": "maniskill_ppo_openvlaoft_quickstart",
            "modelSource": {
                "sourceType": "rlinf_config_default",
                "modelFamily": "openvla-oft",
                "format": "rlinf_config",
            },
        },
        task_name="mock-task",
        device="cuda",
        username="pearl",
    )

    params = captured[0]["params"]
    assert isinstance(params, dict)
    assert params["modelSource"]["sourceType"] == "builtin_policy"
    assert params["modelSourceKind"] == "builtin_policy"


def test_evo_train_bridge_retries_duplicate_task_name():
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="mock",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
    )
    bridge = EvoTrainBridge(settings)
    captured: list[dict[str, object]] = []

    def fake_request(payload: dict[str, object]) -> dict[str, object]:
        captured.append(dict(payload))
        if len(captured) == 1:
            return {
                "ok": False,
                "message": "create task failed: task already exists: mock-task (status=Failed)",
                "tasks": [],
            }
        retry_name = str(payload["taskName"])
        return {
            "ok": True,
            "message": "training accepted",
            "tasks": [
                {
                    "taskName": retry_name,
                    "jobId": f"task:pearl:{retry_name}",
                    "status": "Submitting",
                    "provider": "mock",
                }
            ],
        }

    bridge._request = fake_request  # type: ignore[method-assign]

    result = bridge.start_training(
        None,
        provider="mock",
        workflow="custom_project",
        params={"repoUrl": "https://example.invalid/repo.git", "trainCommand": "true"},
        task_name="mock-task",
        device="cuda",
        username="pearl",
    )

    assert len(captured) == 2
    assert captured[0]["taskName"] == "mock-task"
    assert str(captured[1]["taskName"]).startswith("mock-task-retry-")
    assert result["job_id"] == f"task:pearl:{captured[1]['taskName']}"


def test_evo_train_bridge_adds_client_token_to_requests():
    class FakeSocket:
        def __init__(self) -> None:
            self.sent = b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settimeout(self, _value: float) -> None:
            return None

        def sendall(self, payload: bytes) -> None:
            self.sent += payload

        def recv(self, _size: int) -> bytes:
            return b'{"message":"ok"}\n'

    fake_socket = FakeSocket()
    settings = EvoTrainSettings(
        host="127.0.0.1",
        port=9000,
        timeout_s=3.5,
        provider="mock",
        username="pearl",
        region="cn-hangzhou",
        env_file="",
        dataset_root="",
        checkpoint_root="",
        checkpoint_frequency=1,
        gpu_count=1,
        steps_per_epoch=1000,
        client_token="client-secret",
    )

    with patch("socket.create_connection", return_value=fake_socket):
        response = EvoTrainBridge(settings)._request({"action": "健康检查"})

    assert response == {"message": "ok"}
    sent = json.loads(fake_socket.sent.decode("utf-8"))
    assert sent["apiToken"] == "client-secret"


def test_train_environment_catalog_routes_use_evo_train_bridge(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    skus = client.get("/api/train/gpu-skus?provider=autodl")
    images = client.get("/api/train/images?provider=autodl")

    assert skus.status_code == 200
    assert images.status_code == 200
    assert skus.json()["skus"][0]["skuId"] == "autodl-4090d"
    assert images.json()["images"][0]["imageId"] == "robotics-cu121"
    assert bridge.sku_calls[0]["provider"] == "autodl"
    assert bridge.image_calls[0]["provider"] == "autodl"


def test_train_cloud_resources_returns_platform_catalog(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/train/cloud/resources?provider=aliyun")

    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "aliyun"
    assert data["skus"][0]["skuId"] == "autodl-4090d"
    assert data["images"][0]["imageId"] == "robotics-cu121"
    assert bridge.sku_calls[0]["provider"] == "aliyun"
    assert bridge.image_calls[0]["provider"] == "aliyun"


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


def test_train_source_preflight_forwards_public_dataset_source(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_train_cloud_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/train/source-preflight",
        json={
            "username": "13800138000",
            "provider": "autodl",
            "source": {
                "sourceType": "public_reference",
                "uri": "https://example.edu/datasets/libero.tar.gz",
                "format": "lerobot",
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["source"]["estimatedSize"] == "unknown"
    assert data["source"]["requiresUserConfirmation"] is True
    assert bridge.source_preflight_calls[0]["provider"] == "autodl"
    assert bridge.source_preflight_calls[0]["source"]["uri"] == "https://example.edu/datasets/libero.tar.gz"


def test_evo_studio_agent_consult_delegates_to_cloud_control_plane(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_agent_consult_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/evo-studio/agent-consult",
        json={
            "task": "用 OpenVLA-OFT 评测 LIBERO，先做 smoke test",
            "mode": "plan",
            "username": "pearl",
            "provider": "autodl",
            "workflow": "vla_rl_backend",
            "params": {
                "datasetSource": {
                    "sourceType": "public_reference",
                    "uri": "hf://HuggingFaceVLA/libero",
                    "format": "libero",
                },
                "modelSource": {
                    "sourceType": "public_model_repo",
                    "uri": "hf://openvla/openvla-7b",
                    "format": "huggingface_transformers",
                },
                "modelFamily": "openvla",
            },
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["kind"] == "evo_studio_agent_consult/v1"
    assert payload["consultTool"] == "evo_studio_agent_consult"
    assert payload["readyForConfirmation"] is False
    assert payload["vlaPlan"]["params"]["modelFamily"] == "openvla"
    assert "AI planner did not complete" in payload["vlaPlan"]["missingFields"]
    assert "source_preflight" in payload["actions"]
    assert "runtime_match" in payload["actions"]
    assert bridge.plan_calls
    assert len(bridge.source_preflight_calls) == 2
    assert bridge.runtime_match_calls[0]["params"]["modelFamily"] == "openvla"


def test_evo_studio_agent_consult_uses_configured_ssh_runtime_without_matching(route_app):
    app, _, _ = route_app
    bridge = StubBridge()

    def _ssh_configuration_check(**kwargs: object) -> dict[str, object]:
        return {
            "message": "configuration check success",
            "provider": kwargs.get("provider") or "autodl",
            "ready": True,
            "mode": "ssh",
            "warnings": [],
        }

    bridge.configuration_check = _ssh_configuration_check  # type: ignore[method-assign]

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_agent_consult_routes(app, app.state.embodied_service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/evo-studio/agent-consult",
        json={
            "task": "在 SSH 后端复现 OpenVLA-OFT baseline",
            "mode": "plan",
            "username": "pearl",
            "provider": "ssh",
            "workflow": "vla_rl_backend",
            "context": {"backend": "ssh"},
            "params": {
                "modelFamily": "openvla",
                "datasetSource": {
                    "sourceType": "public_reference",
                    "uri": "hf://HuggingFaceVLA/libero",
                    "format": "libero",
                },
            },
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["runtimeMode"] == "ssh_existing_instance"
    assert payload["provider"] == "autodl"
    assert payload["runtimeMatch"]["skipped"] is True
    assert payload["runtimeMatch"]["readyToStart"] is True
    assert payload["readyForConfirmation"] is False
    assert "AI planner did not complete" in payload["vlaPlan"]["missingFields"]
    assert bridge.runtime_match_calls == []
    assert bridge.plan_calls[0]["provider"] == "autodl"
    assert bridge.source_preflight_calls[0]["provider"] == "autodl"


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
