from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.datasets import DatasetCatalog
from roboclaw.data.storage import build_submission_layout
from roboclaw.data import storage as storage_module
from roboclaw.account import AccountLedger
from roboclaw.http.routes.account import set_ledger_for_tests
from roboclaw.http.routes import datasets as dataset_routes


def _write_dataset(root: Path, dataset_id: str = "demo") -> Path:
    dataset_path = root / dataset_id
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps({
            "total_episodes": 1,
            "total_frames": 30,
            "fps": 30,
            "features": {"action": {}, "observation.state": {}},
        }),
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 30}) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def _make_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    service = SimpleNamespace(datasets=DatasetCatalog(root_resolver=lambda: tmp_path))
    dataset_routes.register_dataset_routes(app, service)  # type: ignore[arg-type]
    return TestClient(app)


def _disable_local_env_file(monkeypatch) -> None:
    monkeypatch.setenv("EVO_STUDIO_DISABLE_ENV_FILE", "1")
    monkeypatch.setattr(storage_module, "_ENV_FILES_LOADED", False)


@pytest.fixture(autouse=True)
def _isolate_storage_env(monkeypatch):
    _disable_local_env_file(monkeypatch)


def test_complete_upload_writes_private_owner_metadata_without_quality(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path)
    calls: list[dict[str, Any]] = []

    class FakeCurationService:
        async def start_quality_run(
            self,
            dataset_path_arg: Path,
            dataset_name: str,
            selected_validators: list[str],
            episode_indices: list[int] | None,
            threshold_overrides: dict[str, float] | None,
            username: str = "",
        ) -> dict[str, str]:
            calls.append({
                "dataset_path": dataset_path_arg,
                "dataset_name": dataset_name,
                "selected_validators": selected_validators,
                "episode_indices": episode_indices,
                "threshold_overrides": threshold_overrides,
                "username": username,
            })
            return {"status": "started"}

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/complete-upload",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "visibility": "public",
            "source_kind": "mounted_path",
            "source_uri": "/mnt/datasets/demo",
            "source_auth_ref": "team-mount-readonly",
            "selected_validators": ["timing"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["ownerUsername"] == "pearl"
    assert payload["quality"] == {"autoTriggered": False, "status": "skipped"}
    assert calls == []

    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["ownerUsername"] == "pearl"
    assert info["contributionSource"] == "self_collected"
    assert info["visibility"] == "private"
    assert info["sourceKind"] == "mounted_path"
    assert info["sourceUri"] == "/mnt/datasets/demo"
    assert info["sourceAuthRef"] == "team-mount-readonly"
    assert info["uploadStatus"] == "uploaded"


def test_storage_status_reports_missing_s3_provider(tmp_path, monkeypatch) -> None:
    for name in [
        "EVO_STUDIO_STORAGE_PROVIDER",
        "EVO_STUDIO_S3_ENDPOINT",
        "EVO_STUDIO_S3_BUCKET",
        "EVO_STUDIO_S3_ACCESS_KEY",
        "EVO_STUDIO_S3_SECRET_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is False
    assert "EVO_STUDIO_S3_BUCKET" in payload["missingFields"]


def test_upload_url_requires_storage_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EVO_STUDIO_STORAGE_PROVIDER", raising=False)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/upload-url",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "filename": "demo.tar",
        },
    )

    assert response.status_code == 501
    assert response.json()["detail"]["message"] == "dataset storage provider is not configured"


def test_storage_status_accepts_aws_compatible_env_aliases(tmp_path, monkeypatch) -> None:
    for name in [
        "EVO_STUDIO_STORAGE_PROVIDER",
        "EVO_STUDIO_S3_ENDPOINT",
        "EVO_STUDIO_S3_ACCESS_KEY",
        "EVO_STUDIO_S3_SECRET_KEY",
        "EVO_STUDIO_S3_REGION",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://example.storage.dev")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("EVO_STUDIO_S3_BUCKET", "evo-studio-data")
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "s3"
    assert payload["configured"] is True
    assert payload["endpoint"] == "https://example.storage.dev"
    assert payload["bucket"] == "evo-studio-data"
    assert payload["missingFields"] == []


def test_storage_status_loads_local_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EVO_STUDIO_DISABLE_ENV_FILE", raising=False)
    for name in [
        "EVO_STUDIO_STORAGE_PROVIDER",
        "EVO_STUDIO_S3_ENDPOINT",
        "EVO_STUDIO_S3_BUCKET",
        "EVO_STUDIO_S3_ACCESS_KEY",
        "EVO_STUDIO_S3_SECRET_KEY",
        "EVO_STUDIO_S3_REGION",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(storage_module, "_ENV_FILES_LOADED", False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text(
        "\n".join([
            "EVO_STUDIO_STORAGE_PROVIDER=s3",
            "AWS_ENDPOINT_URL_S3=https://example.storage.dev",
            "AWS_ACCESS_KEY_ID=test-access-key",
            "AWS_SECRET_ACCESS_KEY=test-secret-key",
            "AWS_REGION=auto",
            "EVO_STUDIO_S3_BUCKET=evo-studio-data",
        ]),
        encoding="utf-8",
    )
    client = _make_client(tmp_path / "catalog")

    response = client.get("/api/datasets/storage/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["endpoint"] == "https://example.storage.dev"
    assert payload["bucket"] == "evo-studio-data"
    assert payload["missingFields"] == []


def test_storage_status_exposes_data_pool_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVO_STUDIO_PENDING_RETENTION_DAYS", "7")
    monkeypatch.setenv("EVO_STUDIO_FREE_PRIVATE_RETENTION_DAYS", "30")
    monkeypatch.setenv("EVO_STUDIO_CONTRIBUTOR_PRIVATE_RETENTION_DAYS", "90")
    monkeypatch.setenv("EVO_STUDIO_TEAM_PRIVATE_RETENTION_DAYS", "180")
    monkeypatch.setenv("EVO_STUDIO_FREE_USER_QUOTA_BYTES", "10")
    monkeypatch.setenv("EVO_STUDIO_CONTRIBUTOR_QUOTA_BYTES", "50")
    monkeypatch.setenv("EVO_STUDIO_TEAM_USER_QUOTA_BYTES", "200")
    monkeypatch.setenv("EVO_STUDIO_BILLING_ALERT_USD", "20")
    monkeypatch.setenv("EVO_STUDIO_BILLING_CONFIRM_USD", "50")
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage/status")

    assert response.status_code == 200
    policy = response.json()["policy"]
    assert policy["pendingRetentionDays"] == 7
    assert policy["privateRetentionDays"] == 30
    assert policy["privateRetentionByRole"] == {"free": 30, "contributor": 90, "team": 180}
    assert policy["quotas"] == {"free": 10, "contributor": 50, "team": 200}
    assert policy["billingAlertUsd"] == 20.0
    assert policy["billingConfirmUsd"] == 50.0
    assert ".tar" in policy["acceptedUploadExtensions"]


def test_upload_url_clamps_max_size_to_user_quota(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVO_STUDIO_STORAGE_PROVIDER", "s3")
    monkeypatch.setenv("EVO_STUDIO_S3_ENDPOINT", "https://example.storage.dev")
    monkeypatch.setenv("EVO_STUDIO_S3_BUCKET", "evo-studio-data")
    monkeypatch.setenv("EVO_STUDIO_S3_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("EVO_STUDIO_S3_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("EVO_STUDIO_FREE_USER_QUOTA_BYTES", "1234")
    captured: dict[str, Any] = {}

    def fake_presigned_upload(_settings, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "method": "POST",
            "url": "https://upload.example",
            "fields": {},
            "expiresIn": 3600,
            "maxSizeBytes": kwargs["max_size_bytes"],
            "objectUri": _settings.object_uri(kwargs["key"]),
            "objectKey": kwargs["key"],
        }

    monkeypatch.setattr(dataset_routes, "create_presigned_upload_post", fake_presigned_upload)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/upload-url",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "filename": "demo.raw-folder",
            "max_size_bytes": 999999,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["max_size_bytes"] == 1234
    assert captured["key"] == "private/users/pearl/datasets/demo/v1/shards/demo.raw-folder"
    assert payload["upload"]["maxSizeBytes"] == 1234
    assert payload["upload"]["objectUri"] == "s3://evo-studio-data/private/users/pearl/datasets/demo/v1/shards/demo.raw-folder"
    assert payload["policy"]["user"]["quotaBytes"] == 1234
    assert payload["warnings"]


def test_build_submission_layout_is_scoped_and_safe() -> None:
    layout = build_submission_layout(
        username="pearl@example.com",
        dataset_id="../SO 101 Demo",
        filename="episode 001.tar",
        prefix="",
        submission_id="sub 001",
    )

    assert layout["submissionPrefix"] == "pending/submissions/pearl-example.com/sub-001/"
    assert layout["shardKey"] == "pending/submissions/pearl-example.com/sub-001/shards/episode-001.tar"
    assert layout["manifestKey"] == "pending/submissions/pearl-example.com/sub-001/manifest.jsonl"
    assert layout["approvedPrefix"] == "approved/datasets/SO-101-Demo/v1/"
    assert layout["previewPrefix"] == "previews/SO-101-Demo/"
    assert layout["redemptionPrefix"] == "redemption/SO-101-Demo/"
    assert ".." not in "\n".join(layout.values())


def test_publish_request_marks_public_and_starts_quality(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path)
    calls: list[dict[str, Any]] = []

    class FakeCurationService:
        async def start_quality_run(
            self,
            dataset_path_arg: Path,
            dataset_name: str,
            selected_validators: list[str],
            episode_indices: list[int] | None,
            threshold_overrides: dict[str, float] | None,
            username: str = "",
        ) -> dict[str, str]:
            calls.append({
                "dataset_path": dataset_path_arg,
                "dataset_name": dataset_name,
                "selected_validators": selected_validators,
                "episode_indices": episode_indices,
                "threshold_overrides": threshold_overrides,
                "username": username,
            })
            return {"status": "started"}

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    upload = client.post(
        "/api/datasets/complete-upload",
        json={"dataset_id": "demo", "username": "pearl"},
    )
    assert upload.status_code == 200

    response = client.post(
        "/api/datasets/demo/publish-request",
        json={"username": "pearl", "selected_validators": ["timing"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "publish_requested"
    assert payload["visibility"] == "private"
    assert payload["publicationStatus"] == "pending_quality"
    assert payload["quality"] == {"status": "started", "autoTriggered": True}
    assert calls == [
        {
            "dataset_path": dataset_path,
            "dataset_name": "demo",
            "selected_validators": ["timing"],
            "episode_indices": None,
            "threshold_overrides": None,
            "username": "pearl",
        }
    ]

    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["visibility"] == "private"
    assert info["requestedVisibility"] == "public"
    assert info["publicationStatus"] == "pending_quality"


def test_ingest_mounted_path_materializes_private_dataset(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "mounted"
    source_dataset = _write_dataset(source_root, "source-demo")
    monkeypatch.setenv("ROBOCLAW_DATASET_INGEST_ROOTS", str(source_root))

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "source_kind": "mounted_path",
            "source_uri": str(source_dataset),
            "force": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ingested"
    assert payload["visibility"] == "private"
    assert payload["quality"] == {"autoTriggered": False, "status": "skipped"}

    target_info = json.loads((tmp_path / "catalog" / "demo" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert target_info["ownerUsername"] == "pearl"
    assert target_info["sourceKind"] == "mounted_path"
    assert target_info["sourceUri"] == str(source_dataset)


def test_ingest_local_archive_materializes_dataset(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "archives"
    source_dataset = _write_dataset(source_root, "archive-demo")
    archive_base = tmp_path / "archive-demo"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=source_root, base_dir="archive-demo"))
    monkeypatch.setenv("ROBOCLAW_DATASET_INGEST_ROOTS", str(tmp_path))

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "source_kind": "local_archive",
            "source_uri": str(archive_path),
            "force": True,
        },
    )

    assert response.status_code == 200
    assert (tmp_path / "catalog" / "demo" / "meta" / "info.json").is_file()


def test_ingest_remote_dataset_uses_catalog_pull(tmp_path, monkeypatch) -> None:
    pulled: dict[str, Any] = {}
    monkeypatch.setenv("ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST", "1")

    def fake_pull_dataset(self: DatasetCatalog, repo_id: str, **kwargs: Any):
        pulled["repo_id"] = repo_id
        pulled.update(kwargs)
        dataset_path = self.resolve_local_path(kwargs["dataset_id"])
        (dataset_path / "meta").mkdir(parents=True, exist_ok=True)
        (dataset_path / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 1, "total_frames": 30, "fps": 30}),
            encoding="utf-8",
        )
        return self.require_local_dataset(kwargs["dataset_id"])

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    monkeypatch.setattr(DatasetCatalog, "pull_dataset", fake_pull_dataset)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "gr00t-libero",
            "username": "pearl",
            "source_kind": "remote_dataset",
            "source_uri": "hf://nvidia/GR00T-N1.7-LIBERO",
        },
    )

    assert response.status_code == 200
    assert pulled["repo_id"] == "nvidia/GR00T-N1.7-LIBERO"
    assert pulled["dataset_id"] == "gr00t-libero"
    assert response.json()["dataset"]["id"] == "gr00t-libero"


def test_ingest_remote_dataset_normalizes_huggingface_dataset_url(tmp_path, monkeypatch) -> None:
    pulled: dict[str, Any] = {}
    monkeypatch.setenv("ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST", "1")

    def fake_pull_dataset(self: DatasetCatalog, repo_id: str, **kwargs: Any):
        pulled["repo_id"] = repo_id
        pulled.update(kwargs)
        dataset_path = self.resolve_local_path(kwargs["dataset_id"])
        (dataset_path / "meta").mkdir(parents=True, exist_ok=True)
        (dataset_path / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 1, "total_frames": 30, "fps": 30}),
            encoding="utf-8",
        )
        return self.require_local_dataset(kwargs["dataset_id"])

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    monkeypatch.setattr(DatasetCatalog, "pull_dataset", fake_pull_dataset)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "libero",
            "username": "pearl",
            "source_kind": "remote_dataset",
            "source_uri": "https://huggingface.co/datasets/HuggingFaceVLA/libero",
        },
    )

    assert response.status_code == 200
    assert pulled["repo_id"] == "HuggingFaceVLA/libero"
    assert pulled["dataset_id"] == "libero"


def test_ingest_remote_dataset_requires_cloud_pool_by_default(tmp_path, monkeypatch) -> None:
    called = False

    def fake_pull_dataset(self: DatasetCatalog, repo_id: str, **kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("local pull should not be called")

    class FakeCurationService:
        pass

    monkeypatch.delenv("ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST", raising=False)
    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    monkeypatch.setattr(DatasetCatalog, "pull_dataset", fake_pull_dataset)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "libero",
            "username": "pearl",
            "source_kind": "remote_dataset",
            "source_uri": "https://huggingface.co/datasets/HuggingFaceVLA/libero",
        },
    )

    assert response.status_code == 501
    assert "cloud data pool" in response.json()["detail"]
    assert called is False


def test_ingest_public_reference_registers_without_local_download(tmp_path, monkeypatch) -> None:
    called = False

    def fake_pull_dataset(self: DatasetCatalog, repo_id: str, **kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("external references must not download")

    class FakeCurationService:
        pass

    monkeypatch.delenv("ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST", raising=False)
    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    monkeypatch.setattr(DatasetCatalog, "pull_dataset", fake_pull_dataset)
    client = _make_client(tmp_path / "catalog")

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "gr00t-libero-ref",
            "username": "pearl",
            "source_kind": "remote_dataset",
            "source_uri": "hf://nvidia/GR00T-N1.7-LIBERO",
            "storage_mode": "external_reference",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "referenced"
    assert payload["storageMode"] == "external_reference"
    assert payload["dataset"]["id"] == "gr00t-libero-ref"
    assert payload["dataset"]["capabilities"]["can_train"] is False
    assert called is False

    info = json.loads((tmp_path / "catalog" / "gr00t-libero-ref" / "meta" / "info.json").read_text())
    assert info["ownerUsername"] == "pearl"
    assert info["sourceKind"] == "remote_dataset"
    assert info["sourceUri"] == "hf://nvidia/GR00T-N1.7-LIBERO"
    assert info["storageMode"] == "external_reference"


def test_ingest_cloud_object_requires_provider(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/ingest",
        json={
            "dataset_id": "demo",
            "username": "pearl",
            "source_kind": "cloud_object",
            "source_uri": "s3://bucket/demo",
        },
    )

    assert response.status_code == 501
    assert "storage provider" in response.json()["detail"]


def test_storage_usage_reports_owner_private_and_public_bytes(tmp_path, monkeypatch) -> None:
    pearl_private = _write_dataset(tmp_path, "pearl-private")
    pearl_public = _write_dataset(tmp_path, "pearl-public")
    other_dataset = _write_dataset(tmp_path, "other")
    (pearl_private / "data.bin").write_bytes(b"a" * 11)
    (pearl_public / "data.bin").write_bytes(b"b" * 17)
    (other_dataset / "data.bin").write_bytes(b"c" * 23)

    for dataset_path, owner, visibility in [
        (pearl_private, "pearl", "private"),
        (pearl_public, "pearl", "public"),
        (other_dataset, "other", "private"),
    ]:
        info_path = dataset_path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["ownerUsername"] = owner
        info["visibility"] = visibility
        info["sourceKind"] = "mounted_path"
        info["sourceUri"] = f"/mnt/{dataset_path.name}"
        info_path.write_text(json.dumps(info), encoding="utf-8")

    monkeypatch.setenv("EVO_STUDIO_FREE_USER_QUOTA_BYTES", "1000")

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage-usage", params={"username": "pearl"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["username"] == "pearl"
    assert payload["role"] == "free"
    assert payload["quotaBytes"] == 1000
    assert payload["datasetCount"] == 2
    assert payload["usedBytes"] == payload["privateBytes"] + payload["publicBytes"]
    assert payload["availableBytes"] == 1000 - payload["usedBytes"]
    assert payload["privateBytes"] > 11
    assert payload["publicBytes"] > 17
    assert {item["id"] for item in payload["datasets"]} == {"pearl-private", "pearl-public"}
    assert {item["visibility"] for item in payload["datasets"]} == {"private", "public"}
    assert payload["policy"]["pendingRetentionDays"] == 14
    assert payload["policy"]["user"]["quotaBytes"] == 1000


def test_storage_usage_requires_username(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage-usage")

    assert response.status_code == 400
    assert response.json()["detail"] == "username is required"


def test_dataset_list_filters_private_data_by_owner(tmp_path, monkeypatch) -> None:
    pearl_private = _write_dataset(tmp_path, "pearl-private")
    other_private = _write_dataset(tmp_path, "other-private")
    public_dataset = _write_dataset(tmp_path, "public-demo")

    for dataset_path, owner, visibility in [
        (pearl_private, "pearl", "private"),
        (other_private, "other", "private"),
        (public_dataset, "owner", "public"),
    ]:
        info_path = dataset_path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["ownerUsername"] = owner
        info["visibility"] = visibility
        info_path.write_text(json.dumps(info), encoding="utf-8")

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets", params={"username": "pearl"})

    assert response.status_code == 200
    assert {item["id"] for item in response.json()} == {"pearl-private", "public-demo"}


def test_dataset_list_requires_username(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets")

    assert response.status_code == 400
    assert response.json()["detail"] == "username is required"


def test_trusted_header_auth_mode_ignores_query_username(tmp_path, monkeypatch) -> None:
    pearl_private = _write_dataset(tmp_path, "pearl-private")
    other_private = _write_dataset(tmp_path, "other-private")
    for dataset_path, owner in [(pearl_private, "pearl"), (other_private, "other")]:
        info_path = dataset_path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["ownerUsername"] = owner
        info["visibility"] = "private"
        info_path.write_text(json.dumps(info), encoding="utf-8")

    class FakeCurationService:
        pass

    monkeypatch.setenv("EVO_STUDIO_AUTH_MODE", "trusted_header")
    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    missing_header = client.get("/api/datasets", params={"username": "pearl"})
    spoofed_query = client.get(
        "/api/datasets",
        params={"username": "other"},
        headers={"X-Evo-Studio-User": "pearl"},
    )

    assert missing_header.status_code == 400
    assert missing_header.json()["detail"] == "username is required"
    assert spoofed_query.status_code == 200
    assert {item["id"] for item in spoofed_query.json()} == {"pearl-private"}


def test_dataset_detail_and_delete_reject_non_owner_private_data(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path, "private-demo")
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ownerUsername"] = "owner"
    info["visibility"] = "private"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    detail = client.get("/api/datasets/private-demo", params={"username": "other"})
    delete = client.delete("/api/datasets/private-demo", params={"username": "other"})

    assert detail.status_code == 403
    assert detail.json()["detail"] == "dataset is private"
    assert delete.status_code == 403
    assert delete.json()["detail"] == "dataset belongs to another user"
    assert dataset_path.exists()


def test_dataset_delete_allows_owner(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path, "private-demo")
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ownerUsername"] = "owner"
    info["visibility"] = "private"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.delete("/api/datasets/private-demo", params={"username": "owner"})

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "id": "private-demo"}
    assert not dataset_path.exists()


def test_publish_request_rejects_non_owner(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path, "private-demo")
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ownerUsername"] = "owner"
    info["visibility"] = "private"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    class FakeCurationService:
        async def start_quality_run(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            raise AssertionError("quality should not run for non-owner")

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post("/api/datasets/private-demo/publish-request", json={"username": "other"})

    assert response.status_code == 403
    assert response.json()["detail"] == "dataset belongs to another user"
    refreshed = json.loads(info_path.read_text(encoding="utf-8"))
    assert refreshed["visibility"] == "private"
    assert "publicationStatus" not in refreshed


def test_redeem_public_dataset_access_spends_points_and_rewards_owner(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path, "shared-demo")
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ownerUsername"] = "owner"
    info["visibility"] = "public"
    info["accessPricePoints"] = 12
    info_path.write_text(json.dumps(info), encoding="utf-8")

    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.grant_dataset_reward("buyer", "seed-buyer", 50)
    set_ledger_for_tests(ledger)

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    monkeypatch.setenv("EVO_STUDIO_DATASET_CONTRIBUTOR_SHARE_BPS", "5000")
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/shared-demo/redeem-access",
        json={"username": "buyer"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["granted"] is True
    assert payload["pricePoints"] == 12
    assert payload["wallet"]["creditPoints"] == 38
    assert payload["accessGrant"]["contributorUsername"] == "owner"
    assert payload["accessGrant"]["contributorPoints"] == 6
    assert payload["buyerRecord"]["kind"] == "dataset_access_charge"
    assert payload["contributorRecord"]["kind"] == "dataset_access_reward"
    assert ledger.wallet("owner").reward_points == 6

    duplicate = client.post(
        "/api/datasets/shared-demo/redeem-access",
        json={"username": "buyer"},
    )

    assert duplicate.status_code == 200
    assert duplicate.json()["granted"] is False
    assert duplicate.json()["wallet"]["creditPoints"] == 38
    assert duplicate.json()["buyerRecord"] is None
    set_ledger_for_tests(None)


def test_redeem_private_dataset_access_rejected(tmp_path, monkeypatch) -> None:
    dataset_path = _write_dataset(tmp_path, "private-demo")
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ownerUsername"] = "owner"
    info["visibility"] = "private"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/private-demo/redeem-access",
        json={"username": "buyer"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "dataset is not public"
    set_ledger_for_tests(None)


def test_complete_upload_can_skip_auto_quality(tmp_path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    calls: list[dict[str, Any]] = []

    class FakeCurationService:
        async def start_quality_run(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            calls.append({})
            return {"status": "started"}

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/complete-upload",
        json={
            "dataset_id": "demo",
            "owner_username": "owner",
            "auto_quality": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["quality"] == {"autoTriggered": False, "status": "skipped"}
    assert calls == []


def test_complete_upload_private_dataset_does_not_auto_quality(tmp_path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    calls: list[dict[str, Any]] = []

    class FakeCurationService:
        async def start_quality_run(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            calls.append({})
            return {"status": "started"}

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/complete-upload",
        json={
            "dataset_id": "demo",
            "owner_username": "owner",
            "visibility": "private",
            "auto_quality": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["quality"] == {"autoTriggered": False, "status": "skipped"}
    assert calls == []


def test_complete_upload_registers_cloud_upload_when_local_dataset_does_not_exist(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/complete-upload",
        json={
            "dataset_id": "so101-cloud-upload",
            "username": "pearl",
            "source_kind": "local_upload",
            "source_uri": "s3://evo-studio-data/pending/submissions/pearl/sub-001/shards/so101.tar",
            "source_auth_ref": "evo-studio-data-pool",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["dataset"]["id"] == "so101-cloud-upload"
    assert payload["visibility"] == "private"

    info = json.loads((tmp_path / "so101-cloud-upload" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["ownerUsername"] == "pearl"
    assert info["sourceKind"] == "local_upload"
    assert info["sourceUri"].startswith("s3://evo-studio-data/")
    assert info["storageMode"] == "managed_upload"
    assert info["privateRetentionDays"] == 30
    assert info["privateExpiresAt"]


def test_complete_upload_rejects_missing_dataset(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/complete-upload",
        json={"dataset_id": "missing", "username": "pearl"},
    )

    assert response.status_code == 404
