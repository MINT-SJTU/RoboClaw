from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.curation import bridge as curation_bridge
from roboclaw.data.curation.state import save_quality_results
from roboclaw.http.routes import curation as curation_routes


def _write_dataset(root: Path) -> Path:
    dataset_path = root / "demo"
    (dataset_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset_path / "data" / "chunk-000").mkdir(parents=True)
    (dataset_path / "videos" / "chunk-000" / "episode_000000").mkdir(parents=True)
    (dataset_path / "videos" / "chunk-000" / "episode_000000" / "front.mp4").write_bytes(
        b"",
    )
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "total_frames": 4,
                "total_tasks": 1,
                "fps": 30,
                "robot_type": "so101",
                "features": {
                    "action": {"names": ["joint_1"]},
                    "observation.state": {"names": ["joint_1"]},
                },
            },
        ),
        encoding="utf-8",
    )
    curation_bridge.write_parquet_rows(
        dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 0,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "tasks": ["pick"],
                "length": 2,
            },
            {
                "episode_index": 1,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": 2,
                "dataset_to_index": 4,
                "tasks": ["pick"],
                "length": 2,
            },
        ],
    )
    curation_bridge.write_parquet_rows(
        dataset_path / "data" / "chunk-000" / "file-000.parquet",
        [
            {"index": 0, "episode_index": 0, "task_index": 0},
            {"index": 1, "episode_index": 0, "task_index": 0},
            {"index": 2, "episode_index": 1, "task_index": 0},
            {"index": 3, "episode_index": 1, "task_index": 0},
        ],
    )
    curation_bridge.write_parquet_rows(
        dataset_path / "meta" / "tasks.parquet",
        [{"task": "pick", "task_index": 0}],
    )
    return dataset_path


def _build_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path]:
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = _write_dataset(dataset_root)
    monkeypatch.setattr(curation_routes, "datasets_root", lambda: dataset_root)
    app = FastAPI()
    curation_routes.register_curation_routes(app)
    return TestClient(app), dataset_path


def test_text_annotations_apply_rewrites_training_task_files(
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
                {"episode_index": 0, "passed": True, "score": 100.0, "task": "pick"},
                {"episode_index": 1, "passed": True, "score": 100.0, "task": "pick"},
            ],
            "selected_validators": ["metadata"],
        },
    )

    response = client.post(
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
    assert response.status_code == 200

    apply_response = client.post(
        "/api/curation/text-annotations-apply",
        json={"dataset": "demo"},
    )
    assert apply_response.status_code == 200
    payload = apply_response.json()
    assert payload["status"] == "applied"
    assert payload["updated_episode_count"] == 1
    assert payload["updated_episode_file_count"] == 1
    assert payload["updated_data_file_count"] == 1
    assert payload["updated_task_file_count"] == 1
    assert payload["task_count"] == 2

    episode_rows = curation_bridge.read_parquet_rows(
        dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    assert episode_rows[0]["tasks"] == ["approach object"]
    assert episode_rows[1]["tasks"] == ["pick"]

    data_rows = curation_bridge.read_parquet_rows(
        dataset_path / "data" / "chunk-000" / "file-000.parquet",
    )
    assert [row["task_index"] for row in data_rows] == [0, 0, 1, 1]

    tasks = curation_bridge.read_parquet_rows(dataset_path / "meta" / "tasks.parquet")
    assert [(row["task_index"], row["task"]) for row in tasks] == [
        (0, "approach object"),
        (1, "pick"),
    ]
    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["old_task"] == "pick"
    assert manifest["episodes"][0]["new_task"] == "approach object"
    assert manifest["files"]["episodes"] == ["meta/episodes/chunk-000/file-000.parquet"]
    assert (Path(payload["backup_dir"]) / "meta" / "tasks.parquet").exists()
    assert (Path(payload["backup_dir"]) / "data" / "chunk-000" / "file-000.parquet").exists()

    quality = client.get("/api/curation/quality-results", params={"dataset": "demo"}).json()
    assert quality["episodes"][0]["task"] == "approach object"
    assert quality["episodes"][1]["task"] == "pick"
