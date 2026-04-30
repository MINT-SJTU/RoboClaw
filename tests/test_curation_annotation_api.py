from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.curation import bridge as curation_bridge
from roboclaw.data.curation import exports as curation_exports
from roboclaw.data.curation import propagation_history
from roboclaw.data.curation import serializers as curation_serializers
from roboclaw.data.curation import service as curation_service
from roboclaw.data import dataset_sessions
from roboclaw.data.curation.state import (
    load_quality_results,
    load_workflow_state,
    save_prototype_results,
    save_quality_results,
    save_workflow_state,
    set_stage_pause_requested,
)
from roboclaw.data.curation.validators import validate_metadata
from roboclaw.http.routes import curation as curation_routes


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
    def _fake_load_episode_data(
        _dataset_path: Path,
        _episode_index: int,
        *,
        include_videos: bool = True,
    ) -> dict[str, object]:
        assert include_videos is False
        return {
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
        }

    monkeypatch.setattr(curation_serializers, "load_episode_data", _fake_load_episode_data)

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


def test_workflow_state_save_writes_json_atomically(tmp_path: Path) -> None:
    dataset_path = _write_demo_dataset(tmp_path)
    state = load_workflow_state(dataset_path)
    state["stages"]["prototype_discovery"]["summary"] = {"candidate_count": 271}

    save_workflow_state(dataset_path, state)

    workflow_dir = dataset_path / ".workflow"
    assert not list(workflow_dir.glob("*.tmp"))
    assert load_workflow_state(dataset_path)["stages"]["prototype_discovery"]["summary"] == {
        "candidate_count": 271,
    }


def test_legacy_propagation_result_serializes_source_history() -> None:
    payload = curation_serializers.serialize_propagation_results(
        {
            "source_episode_index": 2,
            "target_count": 0,
            "propagated": [],
        },
    )

    assert payload["source_episode_indices"] == [2]


def test_quality_defaults_adapt_to_dataset_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_demo_dataset(dataset_root)
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.front"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
    }
    info["features"]["observation.images.wrist"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(curation_routes, "datasets_root", lambda: dataset_root)

    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)
    response = client.get("/api/curation/quality-defaults", params={"dataset": "demo"})

    assert response.status_code == 200
    payload = response.json()
    assert "trajectory_dtw" in payload["selected_validators"]
    assert "visual" in payload["selected_validators"]
    assert payload["threshold_overrides"]["metadata_require_videos"] == 1.0
    assert payload["threshold_overrides"]["visual_min_video_count"] == 2.0
    assert payload["threshold_overrides"]["visual_min_resolution_width"] == 640.0
    assert payload["threshold_overrides"]["visual_min_resolution_height"] == 480.0
    assert payload["checks"]["task_descriptions_present"] is True


def test_metadata_validator_checks_task_description(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo"
    dataset_path.mkdir()
    parquet_path = dataset_path / "data.parquet"
    parquet_path.write_bytes(b"placeholder")

    result = validate_metadata(
        {
            "dataset_path": dataset_path,
            "info": {
                "fps": 30,
                "robot_type": "so101",
                "features": {"action": {"names": ["joint"]}},
            },
            "episode_meta": {"episode_index": 0, "length": 2.0},
            "rows": [],
            "parquet_path": parquet_path,
            "video_files": [],
        },
        threshold_overrides={"metadata_require_videos": 0.0},
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["task_description"]["passed"] is False


def test_metadata_validator_accepts_episode_tasks_list(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo"
    dataset_path.mkdir()
    parquet_path = dataset_path / "data.parquet"
    parquet_path.write_bytes(b"placeholder")

    result = validate_metadata(
        {
            "dataset_path": dataset_path,
            "info": {
                "fps": 30,
                "robot_type": "so101",
                "features": {"action": {"names": ["joint"]}},
            },
            "episode_meta": {
                "episode_index": 0,
                "length": 2.0,
                "tasks": ["pick the yellow cube"],
            },
            "rows": [],
            "parquet_path": parquet_path,
            "video_files": [],
        },
        threshold_overrides={"metadata_require_videos": 0.0},
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["task_description"]["passed"] is True


def test_quality_defaults_accept_task_descriptions_from_episode_tasks_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_demo_dataset(dataset_root)
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 1.0,
                "tasks": ["pick the yellow cube"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(curation_routes, "datasets_root", lambda: dataset_root)

    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)
    response = client.get("/api/curation/quality-defaults", params={"dataset": "demo"})

    assert response.status_code == 200
    assert response.json()["checks"]["task_descriptions_present"] is True


def test_quality_defaults_reads_nested_episode_parquet_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_demo_dataset(dataset_root)
    (dataset_path / "meta" / "episodes.jsonl").unlink()
    episode_parquet = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    curation_bridge.write_parquet_rows(
        episode_parquet,
        [
            {
                "episode_index": 0,
                "length": 30,
                "tasks": ["pick the yellow cube"],
            }
        ],
    )
    monkeypatch.setattr(curation_routes, "datasets_root", lambda: dataset_root)

    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)
    response = client.get("/api/curation/quality-defaults", params={"dataset": "demo"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["episode_metadata_present"] is True
    assert payload["checks"]["task_descriptions_present"] is True


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


def test_annotation_workspace_uses_shared_video_clip_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_path = dataset_root / "demo"
    video_path = (
        dataset_path
        / "videos"
        / "observation.images.front"
        / "chunk-000"
        / "file-000.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    info = {
        "fps": 30,
        "robot_type": "so101",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"names": ["gripper.pos"]},
            "observation.state": {"names": ["gripper.pos"]},
            "observation.images.front": {"dtype": "video"},
        },
    }
    episode_meta = {
        "episode_index": 1,
        "task": "pick",
        "videos/observation.images.front/chunk_index": 0,
        "videos/observation.images.front/file_index": 0,
        "videos/observation.images.front/from_timestamp": 25.633333333333333,
        "videos/observation.images.front/to_timestamp": 51.4,
    }

    monkeypatch.setattr(curation_routes, "datasets_root", lambda: dataset_root)
    def _fake_load_episode_data(
        _dataset_path: Path,
        _episode_index: int,
        *,
        include_videos: bool = True,
    ) -> dict[str, object]:
        assert include_videos is False
        return {
            "info": info,
            "episode_meta": episode_meta,
            "rows": [
                {
                    "timestamp": 0.0,
                    "frame_index": 0,
                    "action": [1.0],
                    "observation.state": [1.0],
                    "task": "pick",
                },
                {
                    "timestamp": 1.0,
                    "frame_index": 30,
                    "action": [2.0],
                    "observation.state": [2.0],
                    "task": "pick",
                },
            ],
            "video_files": [],
        }

    monkeypatch.setattr(curation_serializers, "load_episode_data", _fake_load_episode_data)

    app = FastAPI()
    curation_routes.register_curation_routes(app)
    response = TestClient(app).get(
        "/api/curation/annotation-workspace",
        params={"dataset": "demo", "episode_index": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["videos"][0]["path"] == (
        "videos/observation.images.front/chunk-000/file-000.mp4"
    )
    assert payload["videos"][0]["stream"] == "front"
    assert payload["videos"][0]["from_timestamp"] == 25.633333333333333
    assert payload["videos"][0]["to_timestamp"] == 51.4


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


def test_curation_dataset_list_includes_session_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)

    monkeypatch.setattr(
        curation_routes,
        "list_curation_dataset_summaries",
        lambda: [
            {
                "name": "demo",
                "display_name": "demo",
                "source_kind": "workspace",
                "total_episodes": 1,
                "total_frames": 2,
                "fps": 30,
                "robot_type": "so101",
            },
            {
                "name": "session:remote:abc123",
                "display_name": "cadene/droid_1.0.1",
                "source_kind": "remote_session",
                "total_episodes": 3,
                "total_frames": 42,
                "fps": 30,
                "robot_type": "so101",
            },
        ],
    )

    response = client.get("/api/curation/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert payload[1]["name"] == "session:remote:abc123"
    assert payload[1]["display_name"] == "cadene/droid_1.0.1"


def test_quality_detail_can_resolve_session_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    monkeypatch.setattr(
        curation_routes,
        "resolve_dataset_path",
        lambda name: dataset_path if name == "session:remote:abc123" else (tmp_path / name),
    )

    save_quality_results(
        dataset_path,
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "overall_score": 100,
            "episodes": [{"episode_index": 0, "passed": True, "score": 100}],
            "selected_validators": ["metadata"],
        },
    )

    response = client.get(
        "/api/curation/quality-results",
        params={"dataset": "session:remote:abc123"},
    )
    assert response.status_code == 200
    assert response.json()["overall_score"] == 100


def test_prototype_run_passes_selected_episode_indices_and_filter_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    async def _fake_start_prototype_run(
        dataset_path: Path,
        dataset_name: str,
        cluster_count: int | None,
        candidate_limit: int | None,
        episode_indices: list[int] | None = None,
        quality_filter_mode: str = "passed",
    ) -> dict[str, str]:
        captured["dataset_path"] = dataset_path
        captured["dataset_name"] = dataset_name
        captured["cluster_count"] = cluster_count
        captured["candidate_limit"] = candidate_limit
        captured["episode_indices"] = episode_indices
        captured["quality_filter_mode"] = quality_filter_mode
        return {"status": "started"}

    monkeypatch.setattr(curation_routes, "_service", curation_service.CurationService())
    monkeypatch.setattr(curation_routes._service, "start_prototype_run", _fake_start_prototype_run)

    response = client.post(
        "/api/curation/prototype-run",
        json={
            "dataset": "demo",
            "cluster_count": 3,
            "candidate_limit": 40,
            "episode_indices": [0, 2, 5],
            "quality_filter_mode": "all",
        },
    )

    assert response.status_code == 200
    assert captured["dataset_name"] == "demo"
    assert captured["cluster_count"] == 3
    assert captured["candidate_limit"] == 40
    assert captured["episode_indices"] == [0, 2, 5]
    assert captured["quality_filter_mode"] == "all"


def test_prototype_run_uses_all_candidates_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    async def _fake_start_prototype_run(
        dataset_path: Path,
        dataset_name: str,
        cluster_count: int | None,
        candidate_limit: int | None,
        episode_indices: list[int] | None = None,
        quality_filter_mode: str = "passed",
    ) -> dict[str, str]:
        captured["candidate_limit"] = candidate_limit
        return {"status": "started"}

    monkeypatch.setattr(curation_routes, "_service", curation_service.CurationService())
    monkeypatch.setattr(curation_routes._service, "start_prototype_run", _fake_start_prototype_run)

    response = client.post(
        "/api/curation/prototype-run",
        json={
            "dataset": "demo",
            "cluster_count": None,
            "quality_filter_mode": "raw",
        },
    )

    assert response.status_code == 200
    assert captured["candidate_limit"] is None


def test_prototype_run_keeps_explicit_large_candidate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _dataset_path = _build_client(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    async def _fake_start_prototype_run(
        dataset_path: Path,
        dataset_name: str,
        cluster_count: int | None,
        candidate_limit: int | None,
        episode_indices: list[int] | None = None,
        quality_filter_mode: str = "passed",
    ) -> dict[str, str]:
        captured["candidate_limit"] = candidate_limit
        return {"status": "started"}

    monkeypatch.setattr(curation_routes, "_service", curation_service.CurationService())
    monkeypatch.setattr(curation_routes._service, "start_prototype_run", _fake_start_prototype_run)

    response = client.post(
        "/api/curation/prototype-run",
        json={
            "dataset": "demo",
            "candidate_limit": 271,
            "quality_filter_mode": "raw",
        },
    )

    assert response.status_code == 200
    assert captured["candidate_limit"] == 271


def test_prototype_discovery_raw_mode_uses_all_dataset_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=271)
    captured: dict[str, object] = {}

    def _fake_build_canonical_entries(
        _dataset_path: Path,
        episode_indices: list[int],
        _progress_callback: object = None,
    ) -> list[dict[str, object]]:
        captured["episode_indices"] = list(episode_indices)
        return [
            {
                "record_key": str(index),
                "episode_index": index,
                "sequence": [[float(index)]],
                "quality": {"passed": True, "score": 100.0},
            }
            for index in episode_indices
        ]

    def _fake_discover_grouped_prototypes(
        entries: list[dict[str, object]],
        *,
        cluster_count: int | None = None,
        progress_callback: object = None,
    ) -> dict[str, object]:
        return {
            "clustering": {"cluster_count": 1},
            "refinement": {
                "cluster_count": 1,
                "anchor_record_keys": ["0"],
                "clusters": [
                    {
                        "cluster_index": 0,
                        "prototype_record_key": "0",
                        "anchor_record_key": "0",
                        "member_count": len(entries),
                        "members": [
                            {"record_key": entry["record_key"], "episode_index": entry["episode_index"]}
                            for entry in entries
                        ],
                    }
                ],
            },
            "group_count": 1,
        }

    monkeypatch.setattr(curation_service, "_build_canonical_entries", _fake_build_canonical_entries)
    monkeypatch.setattr(curation_service, "discover_grouped_prototypes", _fake_discover_grouped_prototypes)

    result = curation_service._LegacyCurationService(dataset_path, "demo").run_prototype_discovery(
        quality_filter_mode="raw",
    )

    assert captured["episode_indices"] == list(range(271))
    assert result["quality_filter_mode"] == "raw"
    assert result["selected_episode_indices"] == list(range(271))
    assert result["candidate_count"] == 271
    state = load_workflow_state(dataset_path)
    prototype_stage = state["stages"]["prototype_discovery"]
    assert prototype_stage["selected_episode_indices"] == list(range(271))
    assert prototype_stage["summary"]["candidate_count"] == 271
    assert prototype_stage["summary"]["entry_count"] == 271


def test_prototype_discovery_loads_trajectory_without_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=1)
    captured: dict[str, object] = {}

    def _fake_load_episode_data(
        _dataset_path: Path,
        episode_index: int,
        *,
        include_videos: bool = True,
    ) -> dict[str, object]:
        captured["include_videos"] = include_videos
        return {
            "info": {
                "robot_type": "so101",
                "features": {
                    "action": {"names": ["joint_1"]},
                    "observation.state": {"names": ["joint_1"]},
                },
            },
            "episode_meta": {"episode_index": episode_index, "task": "pick"},
            "rows": [
                {
                    "timestamp": 0.0,
                    "action": [0.0],
                    "observation.state": [0.0],
                    "task": "pick",
                },
                {
                    "timestamp": 0.1,
                    "action": [1.0],
                    "observation.state": [1.0],
                    "task": "pick",
                },
            ],
            "video_files": [],
        }

    def _fake_discover_grouped_prototypes(
        entries: list[dict[str, object]],
        *,
        cluster_count: int | None = None,
        progress_callback: object = None,
    ) -> dict[str, object]:
        return {
            "clustering": {"cluster_count": 1},
            "refinement": {
                "cluster_count": 1,
                "anchor_record_keys": ["0"],
                "clusters": [],
            },
            "group_count": 1,
        }

    monkeypatch.setattr(curation_service, "load_episode_data", _fake_load_episode_data)
    monkeypatch.setattr(curation_service, "discover_grouped_prototypes", _fake_discover_grouped_prototypes)

    result = curation_service._LegacyCurationService(dataset_path, "demo").run_prototype_discovery(
        episode_indices=[0],
    )

    assert captured["include_videos"] is False
    assert result["entry_count"] == 1


def test_duplicate_propagation_run_keeps_existing_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        dataset_path = _write_demo_dataset(tmp_path)
        service = curation_service.CurationService()
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[int] = []

        async def _fake_to_thread(
            _function: object,
            source_episode_index: int,
        ) -> dict[str, object]:
            calls.append(source_episode_index)
            started.set()
            await release.wait()
            curation_service._update_stage_summary(
                dataset_path,
                "annotation",
                {
                    "source_episode_index": source_episode_index,
                    "target_count": 0,
                    "annotated_count": 1,
                },
            )
            return {"source_episode_index": source_episode_index}

        monkeypatch.setattr(curation_service.asyncio, "to_thread", _fake_to_thread)

        first = await service.start_propagation_run(dataset_path, "demo", 0)
        assert first == {"status": "started"}
        await asyncio.wait_for(started.wait(), timeout=1)

        running_state = load_workflow_state(dataset_path)
        assert running_state["stages"]["annotation"]["status"] == "running"

        second = await service.start_propagation_run(dataset_path, "demo", 0)
        assert second == {"status": "already_running"}
        assert calls == [0]

        duplicate_state = load_workflow_state(dataset_path)
        assert duplicate_state["stages"]["annotation"]["status"] == "running"

        task = service._active_stage_task(dataset_path, "annotation")
        assert task is not None
        release.set()
        await asyncio.wait_for(task, timeout=1)

        completed_state = load_workflow_state(dataset_path)
        assert completed_state["stages"]["annotation"]["status"] == "completed"

    asyncio.run(_run())


def test_propagation_source_history_accumulates_across_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        dataset_path = _write_demo_dataset(tmp_path)
        service = curation_service.CurationService()
        calls: list[int] = []

        async def _fake_to_thread(
            _function: object,
            source_episode_index: int,
        ) -> dict[str, object]:
            calls.append(source_episode_index)
            previous_results = curation_service.load_propagation_results(dataset_path)
            state = load_workflow_state(dataset_path)
            annotation_stage = state["stages"]["annotation"]
            source_history = propagation_history.collect_propagated_source_episodes(
                annotation_stage,
                previous_results,
                source_episode_index,
            )
            curation_service.save_propagation_results(
                dataset_path,
                {
                    "source_episode_index": source_episode_index,
                    "source_episode_indices": source_history,
                    "target_count": 0,
                    "propagated": [],
                },
            )
            annotation_stage["propagated_source_episodes"] = source_history
            save_workflow_state(dataset_path, state)
            curation_service._update_stage_summary(
                dataset_path,
                "annotation",
                {
                    "source_episode_index": source_episode_index,
                    "propagated_source_episodes": source_history,
                    "target_count": 0,
                },
            )
            return {"source_episode_index": source_episode_index}

        monkeypatch.setattr(curation_service.asyncio, "to_thread", _fake_to_thread)

        first = await service.start_propagation_run(dataset_path, "demo", 1)
        assert first == {"status": "started"}
        first_task = service._active_stage_task(dataset_path, "annotation")
        assert first_task is not None
        await asyncio.wait_for(first_task, timeout=1)

        second = await service.start_propagation_run(dataset_path, "demo", 2)
        assert second == {"status": "started"}
        second_task = service._active_stage_task(dataset_path, "annotation")
        assert second_task is not None
        await asyncio.wait_for(second_task, timeout=1)

        state = load_workflow_state(dataset_path)
        results = curation_service.load_propagation_results(dataset_path)
        assert calls == [1, 2]
        assert state["stages"]["annotation"]["propagated_source_episodes"] == [1, 2]
        assert results is not None
        assert results["source_episode_indices"] == [1, 2]

    asyncio.run(_run())


def test_workflow_state_recovers_propagated_sources_from_saved_annotations(
    tmp_path: Path,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=2)
    curation_service.save_annotations(
        dataset_path,
        1,
        {
            "episode_index": 1,
            "task_context": {
                "source": "propagation",
                "source_episode_index": 0,
            },
            "annotations": [
                {
                    "id": "ann-1",
                    "label": "Pick",
                    "startTime": 0.0,
                    "endTime": 0.5,
                    "source": "dtw_propagated",
                },
            ],
        },
    )

    state = curation_service.CurationService().get_workflow_state(dataset_path)

    assert state["stages"]["annotation"]["propagated_source_episodes"] == [0]
    assert load_workflow_state(dataset_path)["stages"]["annotation"][
        "propagated_source_episodes"
    ] == [0]


def test_quality_pause_request_marks_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)
    save_quality_results(
        dataset_path,
        {
            "total": 3,
            "passed": 1,
            "failed": 0,
            "overall_score": 100.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
            ],
            "selected_validators": ["metadata"],
        },
    )

    state = load_workflow_state(dataset_path)
    state["stages"]["quality_validation"]["status"] = "running"
    state["stages"]["quality_validation"]["selected_validators"] = ["metadata"]
    state["stages"]["quality_validation"]["active_run_id"] = "run-1"
    save_workflow_state(dataset_path, state)

    response = client.post("/api/curation/quality-pause", json={"dataset": "demo"})
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

    updated = load_workflow_state(dataset_path)
    quality_stage = updated["stages"]["quality_validation"]
    assert quality_stage["status"] == "paused"
    assert quality_stage["pause_requested"] is False
    assert quality_stage["active_run_id"] is None
    assert quality_stage["summary"]["completed"] == 1
    assert quality_stage["summary"]["remaining"] == 2


def test_quality_pause_cancels_active_task_without_error(tmp_path: Path) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=2)
    service = curation_service.CurationService()

    async def _run() -> None:
        started = asyncio.Event()

        async def _task() -> None:
            state = load_workflow_state(dataset_path)
            stage = state["stages"]["quality_validation"]
            stage["status"] = "running"
            stage["active_run_id"] = "run-1"
            save_workflow_state(dataset_path, state)
            started.set()
            await asyncio.sleep(30)

        service._register_workflow_task(dataset_path, "quality_validation", _task())
        await started.wait()
        task = service._active_stage_task(dataset_path, "quality_validation")
        assert task is not None

        response = service.pause_quality_run(dataset_path, "demo")
        assert response == {"status": "paused", "pause_requested": False}
        with pytest.raises(asyncio.CancelledError):
            await task

        updated = load_workflow_state(dataset_path)
        quality_stage = updated["stages"]["quality_validation"]
        assert quality_stage["status"] == "paused"
        assert quality_stage["pause_requested"] is False
        assert quality_stage["active_run_id"] is None

    asyncio.run(_run())


def test_stale_quality_run_does_not_overwrite_paused_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=2)
    service = curation_service.CurationService()
    legacy = curation_service._LegacyCurationService(dataset_path, "demo")
    save_quality_results(
        dataset_path,
        {
            "total": 2,
            "passed": 1,
            "failed": 0,
            "overall_score": 100.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
            ],
            "selected_validators": ["metadata"],
        },
    )

    def _fake_run_quality_validators(
        target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        service.pause_quality_run(target_dataset_path, "demo")
        return {
            "passed": False,
            "score": 10.0,
            "validators": {"metadata": {"passed": False, "score": 10.0}},
            "issues": [{"check_name": "metadata", "passed": False}],
        }

    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)

    result = legacy.run_quality_batch(
        ["metadata"],
        episode_indices=[1],
        resume_existing=True,
        run_id="run-1",
    )

    assert [episode["episode_index"] for episode in result["episodes"]] == [0]
    updated = load_workflow_state(dataset_path)
    quality_stage = updated["stages"]["quality_validation"]
    assert quality_stage["status"] == "paused"
    assert quality_stage["active_run_id"] is None
    assert quality_stage["summary"]["completed"] == 1
    assert load_quality_results(dataset_path)["episodes"][0]["episode_index"] == 0


def test_alignment_overview_combines_quality_and_alignment_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    save_quality_results(
        dataset_path,
        {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "overall_score": 91.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 98.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
                {
                    "episode_index": 1,
                    "passed": False,
                    "score": 84.0,
                    "validators": {"timing": {"passed": False, "score": 70.0}},
                    "issues": [{"check_name": "timing", "passed": False, "message": "bad timing"}],
                },
            ],
            "selected_validators": ["metadata", "timing"],
        },
    )
    save_prototype_results(
        dataset_path,
        {
            "candidate_count": 2,
            "entry_count": 2,
            "cluster_count": 1,
            "quality_filter_mode": "all",
            "selected_episode_indices": [0, 1],
            "refinement": {
                "anchor_record_keys": ["0"],
                "clusters": [],
            },
        },
    )
    curation_service.save_annotations(
        dataset_path,
        0,
        {
            "episode_index": 0,
            "task_context": {"label": "Pick", "text": "pick object"},
            "annotations": [
                {
                    "id": "ann-1",
                    "label": "Pick",
                    "category": "movement",
                    "color": "#ff8a5b",
                    "startTime": 0.0,
                    "endTime": 0.5,
                    "text": "pick object",
                    "tags": ["manual"],
                    "source": "user",
                }
            ],
        },
    )
    curation_service.save_propagation_results(
        dataset_path,
        {
            "source_episode_index": 0,
            "target_count": 1,
            "propagated": [
                {
                    "episode_index": 1,
                    "prototype_score": 0.88,
                    "alignment_method": "dtw",
                    "spans": [
                        {
                            "id": "ann-1",
                            "label": "Pick",
                            "startTime": 0.1,
                            "endTime": 0.4,
                            "source": "dtw_propagated",
                        }
                    ],
                }
            ],
        },
    )

    response = client.get(
        "/api/curation/alignment-overview",
        params={"dataset": "demo"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_checked"] == 2
    assert payload["summary"]["aligned_count"] == 2
    assert payload["summary"]["quality_filter_mode"] == "all"
    assert payload["distribution"]["issue_types"][0]["label"] == "timing"
    rows = {row["episode_index"]: row for row in payload["rows"]}
    assert rows[0]["alignment_status"] == "annotated"
    assert rows[0]["task"] == "pick object"
    assert rows[0]["semantic_task_text"] == "pick object"
    assert rows[0]["task_source"] == "semantic_supplement"
    assert rows[0]["task_is_supplemental"] is True
    assert rows[0]["annotation_spans"][0]["label"] == "Pick"
    assert rows[1]["alignment_status"] == "propagated"
    assert rows[1]["quality_status"] == "failed"
    assert rows[1]["task"] == "Pick"
    assert rows[1]["task_is_supplemental"] is True
    assert rows[1]["propagation_source_episode_index"] == 0
    assert rows[1]["propagation_alignment_method"] == "dtw"
    assert rows[1]["propagation_spans"][0]["label"] == "Pick"
    assert rows[1]["propagation_spans"][0]["dtw_start_delay_s"] == 0.1
    assert rows[1]["propagation_spans"][0]["duration_delta_s"] == -0.2


def test_alignment_overview_recovers_saved_propagated_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)

    save_quality_results(
        dataset_path,
        {
            "total": 2,
            "passed": 2,
            "failed": 0,
            "overall_score": 100.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
                {
                    "episode_index": 1,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
            ],
        },
    )
    curation_service.save_annotations(
        dataset_path,
        0,
        {
            "episode_index": 0,
            "task_context": {"label": "Pick", "text": "pick object"},
            "annotations": [
                {
                    "id": "ann-1",
                    "label": "Pick",
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "text": "pick object",
                    "source": "user",
                }
            ],
        },
    )
    curation_service.save_annotations(
        dataset_path,
        1,
        {
            "episode_index": 1,
            "task_context": {
                "source": "propagation",
                "source_episode_index": 0,
            },
            "annotations": [
                {
                    "id": "ann-1",
                    "label": "Pick",
                    "startTime": 1.25,
                    "endTime": 2.5,
                    "text": "pick object",
                    "source": "dtw_propagated",
                    "propagated": True,
                    "prototype_score": 0.5,
                }
            ],
        },
    )
    curation_service.save_propagation_results(
        dataset_path,
        {
            "source_episode_index": 0,
            "source_episode_indices": [0],
            "target_count": 0,
            "propagated": [],
        },
    )

    response = client.get(
        "/api/curation/alignment-overview",
        params={"dataset": "demo"},
    )

    assert response.status_code == 200
    rows = {row["episode_index"]: row for row in response.json()["rows"]}
    assert rows[1]["alignment_status"] == "propagated"
    assert rows[1]["propagated_count"] == 1
    assert rows[1]["propagation_source_episode_index"] == 0
    assert rows[1]["propagation_alignment_method"] == "dtw"
    assert rows[1]["propagation_spans"][0]["dtw_start_delay_s"] == 0.25
    assert rows[1]["propagation_spans"][0]["dtw_end_delay_s"] == 0.5


def test_alignment_overview_raw_mode_uses_dataset_episodes_without_quality_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, dataset_path = _build_client(tmp_path, monkeypatch)
    info = json.loads((dataset_path / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = 2
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    save_prototype_results(
        dataset_path,
        {
            "candidate_count": 2,
            "entry_count": 2,
            "cluster_count": 1,
            "quality_filter_mode": "raw",
            "selected_episode_indices": [0, 1],
            "refinement": {
                "anchor_record_keys": ["0"],
                "clusters": [],
            },
        },
    )

    response = client.get(
        "/api/curation/alignment-overview",
        params={"dataset": "demo"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_checked"] == 2
    assert payload["summary"]["quality_filter_mode"] == "raw"
    assert [row["episode_index"] for row in payload["rows"]] == [0, 1]
    assert all(row["quality_status"] == "passed" for row in payload["rows"])


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


def test_quality_resume_empty_remaining_does_not_rerun_base_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=2)
    save_quality_results(
        dataset_path,
        {
            "total": 2,
            "passed": 2,
            "failed": 0,
            "overall_score": 100.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
                {
                    "episode_index": 1,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                },
            ],
            "selected_validators": ["metadata"],
        },
    )

    def _unexpected_run_quality_validators(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("resume with an empty remaining list must not rerun base validators")

    monkeypatch.setattr(curation_service, "run_quality_validators", _unexpected_run_quality_validators)

    result = curation_service._LegacyCurationService(dataset_path, "demo").run_quality_batch(
        ["metadata"],
        episode_indices=[],
        resume_existing=True,
    )

    assert result["total"] == 2
    assert [episode["episode_index"] for episode in result["episodes"]] == [0, 1]
    state = load_workflow_state(dataset_path)
    assert state["stages"]["quality_validation"]["status"] == "completed"
    assert state["stages"]["quality_validation"]["summary"]["completed"] == 2


def test_quality_batch_cleans_remote_video_cache_after_last_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "session:remote:cache-demo"
    (dataset_path / "meta").mkdir(parents=True)
    video_dir = dataset_path / "videos" / "observation.images.front" / "chunk-000"
    video_dir.mkdir(parents=True)
    first_video = video_dir / "file-000.mp4"
    second_video = video_dir / "file-001.mp4"
    first_video.write_bytes(b"first")
    second_video.write_bytes(b"second")
    info = {
        "total_episodes": 3,
        "total_frames": 3,
        "fps": 30,
        "robot_type": "so101",
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video.fps": 30, "video.width": 640, "video.height": 480},
            }
        },
    }
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_index": 0,
                    "length": 1.0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                }),
                json.dumps({
                    "episode_index": 1,
                    "length": 1.0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                }),
                json.dumps({
                    "episode_index": 2,
                    "length": 1.0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 1,
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cleanup_calls: list[tuple[int, list[int]]] = []

    def _fake_run_quality_validators(
        _target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        return {
            "passed": True,
            "score": 100.0,
            "validators": {"metadata": {"passed": True, "score": 100.0}},
            "issues": [],
        }

    original_cleanup = curation_service._cleanup_completed_remote_episode_assets

    def _spy_cleanup(
        target_dataset_path: Path,
        target_info: dict[str, object],
        completed_episode_index: int,
        remaining_episode_indices: set[int],
    ) -> dict[str, object]:
        result = original_cleanup(
            target_dataset_path,
            target_info,
            completed_episode_index,
            remaining_episode_indices,
        )
        cleanup_calls.append((completed_episode_index, sorted(remaining_episode_indices)))
        return result

    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)
    monkeypatch.setattr(curation_service, "_cleanup_completed_remote_episode_assets", _spy_cleanup)

    result = curation_service._LegacyCurationService(dataset_path, dataset_path.name).run_quality_batch(["metadata"])

    assert result["total"] == 3
    assert cleanup_calls == [(0, [1, 2]), (1, [2]), (2, [])]
    assert not first_video.exists()
    assert not second_video.exists()
    assert not (dataset_path / "videos").exists()


def test_quality_batch_cleans_real_remote_session_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_root = tmp_path / "dataset-sessions" / "remote"
    dataset_path = remote_root / "cache-demo" / "dataset"
    (dataset_path / "meta").mkdir(parents=True)
    video_dir = dataset_path / "videos" / "observation.images.front" / "chunk-000"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "file-000.mp4"
    video_path.write_bytes(b"remote")
    info = {
        "total_episodes": 1,
        "total_frames": 1,
        "fps": 30,
        "robot_type": "so101",
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video.fps": 30, "video.width": 640, "video.height": 480},
            }
        },
    }
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        json.dumps({
            "episode_index": 0,
            "length": 1.0,
            "videos/observation.images.front/chunk_index": 0,
            "videos/observation.images.front/file_index": 0,
        })
        + "\n",
        encoding="utf-8",
    )

    def _fake_run_quality_validators(
        _target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        return {
            "passed": True,
            "score": 100.0,
            "validators": {"metadata": {"passed": True, "score": 100.0}},
            "issues": [],
        }

    monkeypatch.setattr(dataset_sessions, "_session_root", lambda: tmp_path / "dataset-sessions")
    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)

    result = curation_service._LegacyCurationService(dataset_path, "session:remote:cache-demo").run_quality_batch(["metadata"])

    assert result["total"] == 1
    assert not video_path.exists()
    assert not (dataset_path / "videos").exists()


def test_quality_resume_cleans_completed_remote_video_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "session:remote:resume-demo"
    (dataset_path / "meta").mkdir(parents=True)
    video_dir = dataset_path / "videos" / "observation.images.front" / "chunk-000"
    video_dir.mkdir(parents=True)
    completed_video = video_dir / "file-000.mp4"
    remaining_video = video_dir / "file-001.mp4"
    completed_video.write_bytes(b"completed")
    remaining_video.write_bytes(b"remaining")
    info = {
        "total_episodes": 2,
        "total_frames": 2,
        "fps": 30,
        "robot_type": "so101",
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video.fps": 30, "video.width": 640, "video.height": 480},
            }
        },
    }
    (dataset_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_index": 0,
                    "length": 1.0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                }),
                json.dumps({
                    "episode_index": 1,
                    "length": 1.0,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 1,
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_quality_results(
        dataset_path,
        {
            "total": 2,
            "passed": 1,
            "failed": 0,
            "overall_score": 100.0,
            "episodes": [
                {
                    "episode_index": 0,
                    "passed": True,
                    "score": 100.0,
                    "validators": {"metadata": {"passed": True, "score": 100.0}},
                    "issues": [],
                }
            ],
            "selected_validators": ["metadata"],
        },
    )

    def _fake_run_quality_validators(
        _target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        assert not completed_video.exists()
        assert remaining_video.exists()
        return {
            "passed": True,
            "score": 100.0,
            "validators": {"metadata": {"passed": True, "score": 100.0}},
            "issues": [],
        }

    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)

    result = curation_service._LegacyCurationService(dataset_path, dataset_path.name).run_quality_batch(
        ["metadata"],
        episode_indices=[1],
        resume_existing=True,
    )

    assert result["total"] == 2
    assert not completed_video.exists()
    assert not remaining_video.exists()


def test_quality_batch_keeps_local_dataset_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_demo_dataset(tmp_path, total_episodes=1)
    video_path = dataset_path / "videos" / "chunk-000" / "episode_000000" / "front.mp4"
    video_path.write_bytes(b"local")

    def _fake_run_quality_validators(
        _target_dataset_path: Path,
        episode_index: int,
        *,
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, object]:
        return {
            "passed": True,
            "score": 100.0,
            "validators": {"metadata": {"passed": True, "score": 100.0}},
            "issues": [],
        }

    monkeypatch.setattr(curation_service, "run_quality_validators", _fake_run_quality_validators)

    curation_service._LegacyCurationService(dataset_path, "demo").run_quality_batch(["metadata"])

    assert video_path.read_bytes() == b"local"


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
                        "trajectory_dtw": {"passed": True, "score": 100.0},
                    },
                    "issues": [],
                }
            ],
            "selected_validators": ["metadata", "timing", "trajectory_dtw"],
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
    assert written[0][1][0]["trajectory_dtw_score"] == 100.0
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
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    client = TestClient(app)

    response = client.get("/api/curation/datasets")
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["id"] == "cadene/droid_1.0.1"
    assert payload[0]["label"] == "cadene/droid_1.0.1"

    # Detail route must handle the nested name with slash
    detail = client.get("/api/curation/datasets/cadene/droid_1.0.1")
    assert detail.status_code == 200
    assert detail.json()["id"] == "cadene/droid_1.0.1"


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

    monkeypatch.setattr(
        curation_routes,
        "datasets_root",
        lambda: dataset_root,
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)
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
    assert final_payload["imported_dataset_id"] == "cadene/droid_1.0.1"


def test_workflow_dataset_detail_uses_remote_dataset_info(
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

    monkeypatch.setattr(
        "roboclaw.data.explorer.remote.build_remote_dataset_info",
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
    assert payload["id"] == "cadene/droid_1.0.1"
    assert payload["kind"] == "remote"
    assert payload["stats"]["total_episodes"] == 2
