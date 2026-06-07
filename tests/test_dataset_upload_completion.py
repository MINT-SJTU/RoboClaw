from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.datasets import DatasetCatalog
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
    assert payload["visibility"] == "public"
    assert payload["publicationStatus"] == "pending_quality"
    assert payload["quality"] == {"status": "started", "autoTriggered": True}
    assert calls == [
        {
            "dataset_path": dataset_path,
            "dataset_name": "demo",
            "selected_validators": ["timing"],
            "episode_indices": None,
            "threshold_overrides": None,
            "username": "",
        }
    ]

    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["visibility"] == "public"
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

    monkeypatch.setenv("ROBOCLAW_DATASET_STORAGE_QUOTA_BYTES", "1000")

    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage-usage", params={"username": "pearl"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["username"] == "pearl"
    assert payload["quotaBytes"] == 1000
    assert payload["datasetCount"] == 2
    assert payload["usedBytes"] == payload["privateBytes"] + payload["publicBytes"]
    assert payload["availableBytes"] == 1000 - payload["usedBytes"]
    assert payload["privateBytes"] > 11
    assert payload["publicBytes"] > 17
    assert {item["id"] for item in payload["datasets"]} == {"pearl-private", "pearl-public"}
    assert {item["visibility"] for item in payload["datasets"]} == {"private", "public"}


def test_storage_usage_requires_username(tmp_path, monkeypatch) -> None:
    class FakeCurationService:
        pass

    monkeypatch.setattr(dataset_routes, "CurationService", FakeCurationService)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/storage-usage")

    assert response.status_code == 400
    assert response.json()["detail"] == "username is required"


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
