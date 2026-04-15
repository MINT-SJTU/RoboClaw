from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.http.routes import curation as curation_routes
from roboclaw.data.curation import exports as curation_exports
from roboclaw.data.curation import hf_import as curation_hf_import
from roboclaw.data.curation import serializers as curation_serializers
from roboclaw.data.curation import service as curation_service
from roboclaw.data.curation.state import (
    load_workflow_state,
    save_prototype_results,
    save_quality_results,
    save_workflow_state,
    set_stage_pause_requested,
)


def _write_demo_dataset(root: Path, total_episodes: int = 1) -> Path:
    dataset_path = root / "demo"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "videos" / "chunk-000" / "episode_000000").mkdir(parents=True)

    info = {
        "total_episodes": total_episodes,
        "total_frames": total_episodes * 2,
        "fps": 30,
        "robot_type": "so101",
        "features": {
            "action": {"names": ["joint_1", "joint_2"]},
            "observation.state": {"names": ["joint_1", "joint_2"]},
        },
    }
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": 1.0, "task": "pick"}) + "\n"
            for index in range(total_episodes)
        ),
        encoding="utf-8",
    )
    (dataset_path / "videos" / "chunk-000" / "episode_000000" / "front.mp4").write_bytes(
        b"",
    )

    return dataset_path


def _build_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_demo_dataset(dataset_root)
    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    video_path = dataset_path / "videos" / "chunk-000" / "episode_000000" / "front.mp4"

    monkeypatch.setattr(
        curation_routes,
        "datasets_root",
        lambda: dataset_root,
    )
    monkeypatch.setattr(
        curation_routes,
        "resolve_dataset_path",
        lambda name: (dataset_root / name).resolve(),
    )
    monkeypatch.setattr(
        curation_serializers,
        "load_episode_data",
        lambda _dataset_path, _episode_index: {
            "info": info,
            "episode_meta": {"episode_index": 0, "length": 1.0, "task": "pick"},
            "rows": [
                {
                    "timestamp": 0.0,
                    "frame_index": 0,
                    "action": [0.1, 0.2],
                    "observation.state": [0.0, 0.1],
                    "task": "pick",
                },
                {
                    "timestamp": 1.0,
                    "frame_index": 1,
                    "action": [0.3, 0.4],
                    "observation.state": [0.2, 0.3],
                    "task": "pick",
                },
            ],
            "video_files": [video_path],
        },
    )

    app = FastAPI()
    curation_routes.register_curation_routes(app)
    return TestClient(app), dataset_path


def test_annotation_save_versions_and_updates_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)
    body = {
        "dataset": "demo",
        "episode_index": 0,
        "task_context": {"label": "Pick", "text": "pick the object"},
        "annotations": [
            {
                "id": "ann-1",
                "label": "Pick",
                "category": "movement",
                "color": "#ff8a5b",
                "startTime": 0.0,
                "endTime": 0.7,
                "text": "pick the object",
                "tags": ["manual"],
                "source": "user",
            }
        ],
    }

    first = client.post("/api/curation/annotations", json=body)
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["version_number"] == 1
    assert first_payload["episode_index"] == 0
    assert first_payload["task_context"]["label"] == "Pick"

    second = client.post("/api/curation/annotations", json=body)
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["version_number"] == 2

    state_response = client.get("/api/curation/state", params={"dataset": "demo"})
    assert state_response.status_code == 200
    stage = state_response.json()["stages"]["annotation"]
    assert stage["annotated_episodes"] == [0]
    assert stage["summary"]["annotated_count"] == 1
    assert stage["summary"]["last_saved_episode_index"] == 0


def test_annotation_workspace_returns_video_and_joint_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)

    response = client.get(
        "/api/curation/annotation-workspace",
        params={"dataset": "demo", "episode_index": 0},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["summary"]["record_key"] == "0"
    assert payload["summary"]["duration_s"] == 1.0
    assert payload["videos"][0]["path"].endswith("front.mp4")
    assert payload["videos"][0]["from_timestamp"] == 0
    assert payload["videos"][0]["to_timestamp"] == 1.0
    assert payload["joint_trajectory"]["frame_values"] == [0, 1]
    assert len(payload["joint_trajectory"]["joint_trajectories"]) == 2
    assert payload["annotations"]["version_number"] == 0


def test_workflow_result_endpoints_serialize_ui_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    save_quality_results(
        dataset_path,
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "overall_score": 92.5,
            "episodes": [{"episode_index": 0, "passed": True, "score": 92.5}],
            "selected_validators": ["metadata"],
        },
    )
    save_prototype_results(
        dataset_path,
        {
            "candidate_count": 1,
            "entry_count": 1,
            "cluster_count": 1,
            "refinement": {
                "anchor_record_keys": ["0"],
                "clusters": [
                    {
                        "cluster_index": 0,
                        "prototype_record_key": "0",
                        "anchor_record_key": "0",
                        "member_count": 1,
                        "members": [
                            {
                                "record_key": "0",
                                "distance_to_prototype": 0.0,
                                "distance_to_barycenter": 0.0,
                                "quality": {"score": 92.5, "passed": True},
                            }
                        ],
                    }
                ],
            },
        },
    )

    quality_response = client.get(
        "/api/curation/quality-results",
        params={"dataset": "demo"},
    )
    assert quality_response.status_code == 200
    assert quality_response.json()["overall_score"] == 92.5

    prototype_response = client.get(
        "/api/curation/prototype-results",
        params={"dataset": "demo"},
    )
    assert prototype_response.status_code == 200
    prototype_payload = prototype_response.json()
    assert prototype_payload["anchor_record_keys"] == ["0"]
    assert prototype_payload["clusters"][0]["anchor_record_key"] == "0"
    assert prototype_payload["clusters"][0]["members"][0]["episode_index"] == 0


def test_quality_pause_request_marks_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    state = load_workflow_state(dataset_path)
    state["stages"]["quality_validation"]["status"] = "running"
    save_workflow_state(dataset_path, state)

    response = client.post("/api/curation/quality-pause", json={"dataset": "demo"})
    assert response.status_code == 200
    assert response.json()["status"] == "pause_requested"

    updated = load_workflow_state(dataset_path)
    assert updated["stages"]["quality_validation"]["pause_requested"] is True


def test_quality_batch_can_pause_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_demo_dataset(dataset_root, total_episodes=3)
    service = curation_service._LegacyCurationService(dataset_path, "demo")

    def _fake_run_quality_validators(
        target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        if episode_index == 0:
            set_stage_pause_requested(target_dataset_path, "quality_validation", True)
        return {
            "passed": episode_index != 1,
            "score": 100.0 if episode_index != 1 else 50.0,
            "validators": {
                "metadata": {
                    "passed": episode_index != 1,
                    "score": 100.0 if episode_index != 1 else 50.0,
                },
            },
            "issues": [] if episode_index != 1 else [{"check_name": "fps", "passed": False}],
        }

    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)

    paused = service.run_quality_batch(["metadata"], threshold_overrides={"metadata_min_duration_s": 1.0})
    assert paused["episodes"][0]["episode_index"] == 0
    assert len(paused["episodes"]) == 1

    paused_state = load_workflow_state(dataset_path)
    assert paused_state["stages"]["quality_validation"]["status"] == "paused"
    assert paused_state["stages"]["quality_validation"]["pause_requested"] is False
    assert paused_state["stages"]["quality_validation"]["summary"]["completed"] == 1

    resumed = service.run_quality_batch(
        ["metadata"],
        episode_indices=[1, 2],
        threshold_overrides={"metadata_min_duration_s": 1.0},
        resume_existing=True,
    )
    assert resumed["total"] == 3
    assert [episode["episode_index"] for episode in resumed["episodes"]] == [0, 1, 2]

    resumed_state = load_workflow_state(dataset_path)
    assert resumed_state["stages"]["quality_validation"]["status"] == "completed"
    assert resumed_state["stages"]["quality_validation"]["summary"]["completed"] == 3


def test_delete_quality_results_clears_artifacts_and_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    save_quality_results(
        dataset_path,
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "overall_score": 92.5,
            "episodes": [{"episode_index": 0, "passed": True, "score": 92.5}],
            "selected_validators": ["metadata"],
        },
    )

    working_parquet = curation_exports.workflow_quality_parquet_path(dataset_path)
    working_parquet.parent.mkdir(parents=True, exist_ok=True)
    working_parquet.write_bytes(b"working")

    published_parquet = curation_exports.dataset_quality_parquet_path(dataset_path)
    published_parquet.parent.mkdir(parents=True, exist_ok=True)
    published_parquet.write_bytes(b"published")

    state = load_workflow_state(dataset_path)
    state["stages"]["quality_validation"] = {
        "status": "completed",
        "selected_validators": ["metadata"],
        "latest_run": {"id": "quality-run-1"},
        "summary": {"total": 1, "passed": 1},
    }
    save_workflow_state(dataset_path, state)

    response = client.delete(
        "/api/curation/quality-results",
        params={"dataset": "demo"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "deleted"
    assert len(payload["removed_paths"]) == 3

    assert not (dataset_path / ".workflow" / "quality" / "latest.json").exists()
    assert not working_parquet.exists()
    assert not published_parquet.exists()

    refreshed_state = load_workflow_state(dataset_path)
    quality_stage = refreshed_state["stages"]["quality_validation"]
    assert quality_stage["status"] == "idle"
    assert quality_stage["selected_validators"] == []
    assert quality_stage["latest_run"] is None
    assert quality_stage["summary"] is None

    quality_response = client.get(
        "/api/curation/quality-results",
        params={"dataset": "demo"},
    )
    assert quality_response.status_code == 200
    assert quality_response.json()["episodes"] == []



def test_workflow_publish_endpoints_build_quality_and_text_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)
    written: list[tuple[str, list[dict[str, object]]]] = []

    def _fake_write_parquet(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
        written.append((str(path), rows))
        return {"path": str(path), "row_count": len(rows)}

    monkeypatch.setattr(curation_exports, "write_parquet_rows", _fake_write_parquet)

    save_quality_results(
        dataset_path,
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "overall_score": 92.5,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 92.5,
                    "validators": {
                        "metadata": {"passed": True, "score": 100.0},
                        "timing": {"passed": True, "score": 90.0},
                    },
                    "issues": [],
                }
            ],
            "selected_validators": ["metadata", "timing"],
        },
    )

    save_prototype_results(
        dataset_path,
        {
            "candidate_count": 1,
            "entry_count": 1,
            "cluster_count": 1,
            "refinement": {
                "clusters": [
                    {
                        "cluster_index": 0,
                        "prototype_record_key": "0",
                        "anchor_record_key": "0",
                        "member_count": 1,
                        "members": [{"record_key": "0"}],
                    }
                ],
            },
        },
    )

    client.post(
        "/api/curation/annotations",
        json={
            "dataset": "demo",
            "episode_index": 0,
            "task_context": {"label": "Pick", "text": "pick"},
            "annotations": [
                {
                    "id": "ann-1",
                    "label": "approach",
                    "category": "movement",
                    "color": "#ff8a5b",
                    "startTime": 0.0,
                    "endTime": 0.5,
                    "text": "approach object",
                    "tags": ["manual"],
                    "source": "user",
                }
            ],
        },
    )

    quality_publish = client.post("/api/curation/quality-publish", json={"dataset": "demo"})
    assert quality_publish.status_code == 200
    assert quality_publish.json()["row_count"] == 1

    text_publish = client.post(
        "/api/curation/text-annotations-publish",
        json={"dataset": "demo"},
    )
    assert text_publish.status_code == 200
    assert text_publish.json()["row_count"] == 1

    assert written[0][0].endswith("meta/quality_results.parquet")
    assert written[0][1][0]["episode_index"] == 0
    assert written[1][0].endswith("meta/text_annotations.parquet")
    assert written[1][1][0]["annotation_id"] == "ann-1"


def test_workflow_datasets_preserve_nested_hf_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    nested = dataset_root / "cadene" / "droid_1.0.1" / "meta"
    nested.mkdir(parents=True)
    (nested / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 20, "fps": 10, "robot_type": "aloha"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        curation_routes,
        "datasets_root",
        lambda: dataset_root,
    )
    monkeypatch.setattr(
        curation_routes,
        "resolve_dataset_path",
        lambda name: (dataset_root / name).resolve(),
    )
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)

    response = client.get("/api/curation/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["name"] == "cadene/droid_1.0.1"

    # Detail route must handle the nested name with slash
    detail = client.get("/api/curation/datasets/cadene/droid_1.0.1")
    assert detail.status_code == 200
    assert detail.json()["name"] == "cadene/droid_1.0.1"


def test_resolve_dataset_path_rejects_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    monkeypatch.setattr(
        curation_routes,
        "datasets_root",
        lambda: dataset_root,
    )
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)

    response = client.get(
        "/api/curation/state",
        params={"dataset": "../../etc/passwd"},
    )
    assert response.status_code == 404


def test_workflow_import_hf_dataset_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    def _fake_snapshot_download(*, repo_id: str, local_dir: str, **_: object) -> str:
        target_dir = Path(local_dir)
        (target_dir / "meta").mkdir(parents=True, exist_ok=True)
        (target_dir / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 1, "total_frames": 2, "fps": 30}),
            encoding="utf-8",
        )
        return str(target_dir)

    monkeypatch.setattr(curation_hf_import, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(
        curation_hf_import,
        "datasets_root",
        lambda: dataset_root,
    )
    monkeypatch.setattr(
        curation_routes,
        "datasets_root",
        lambda: dataset_root,
    )
    monkeypatch.setattr(
        curation_routes,
        "resolve_dataset_path",
        lambda name: (dataset_root / name).resolve(),
    )
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)

    queued = client.post(
        "/api/curation/datasets/import-hf",
        json={"dataset_id": "cadene/droid_1.0.1", "include_videos": False},
    )
    assert queued.status_code == 200
    job_id = queued.json()["job_id"]

    final_payload = None
    for _ in range(100):
        status = client.get(f"/api/curation/datasets/import-status/{job_id}")
        assert status.status_code == 200
        final_payload = status.json()
        if final_payload["status"] in {"completed", "error"}:
            break
        time.sleep(0.02)

    assert final_payload is not None
    assert final_payload["status"] == "completed"
    assert final_payload["imported_dataset"] == "cadene/droid_1.0.1"


def test_workflow_dataset_detail_uses_remote_dataset_info(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        curation_routes,
        "get_dataset_info",
        lambda _root, _name: None,
    )
    monkeypatch.setattr(
        curation_routes,
        "build_remote_dataset_info",
        lambda dataset: {
            "name": dataset,
            "total_episodes": 2,
            "total_frames": 20,
            "fps": 30,
            "episode_lengths": [8, 12],
            "features": ["action"],
            "robot_type": "aloha",
            "source_dataset": dataset,
        },
    )

    response = client.get("/api/curation/datasets/cadene/droid_1.0.1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "cadene/droid_1.0.1"
    assert payload["total_episodes"] == 2
