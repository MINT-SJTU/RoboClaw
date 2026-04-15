from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.http.routes import explorer as explorer_routes


def test_explorer_dashboard_uses_remote_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "build_remote_explorer_payload",
        lambda dataset: {
            "dataset": dataset,
            "summary": {
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 30,
                "robot_type": "aloha",
                "codebase_version": "",
                "chunks_size": 1000,
            },
            "files": {
                "total_files": 5,
                "parquet_files": 2,
                "video_files": 1,
                "meta_files": 2,
                "other_files": 0,
            },
            "feature_names": ["action", "observation.state"],
            "feature_stats": [],
            "feature_type_distribution": [{"name": "sequence", "value": 2}],
            "dataset_stats": {"row_count": 10, "features_with_stats": 0, "vector_features": 2},
            "modality_summary": [],
            "episodes": [{"episode_index": 0, "length": 10}],
        },
    )

    response = client.get("/api/explorer/dashboard", params={"dataset": "cadene/droid_1.0.1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "cadene/droid_1.0.1"
    assert payload["summary"]["total_episodes"] == 2


def test_explorer_episode_uses_remote_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "load_remote_episode_detail",
        lambda dataset, episode_index: {
            "episode_index": episode_index,
            "summary": {
                "row_count": 12,
                "fps": 30,
                "duration_s": 1.2,
                "video_count": 1,
            },
            "sample_rows": [{"frame_index": 0, "timestamp": 0.0}],
            "joint_trajectory": {
                "x_axis_key": "timestamp",
                "x_values": [0.0, 0.5],
                "time_values": [0.0, 0.5],
                "frame_values": [0, 1],
                "joint_trajectories": [],
                "sampled_points": 2,
                "total_points": 2,
            },
            "videos": [
                {
                    "path": "videos/chunk-000/episode_000000/front.mp4",
                    "url": "https://huggingface.co/datasets/cadene/droid_1.0.1/resolve/main/videos/chunk-000/episode_000000/front.mp4",
                    "stream": "front",
                }
            ],
        },
    )

    response = client.get(
        "/api/explorer/episode",
        params={"dataset": "cadene/droid_1.0.1", "episode_index": 0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_index"] == 0
    assert payload["summary"]["video_count"] == 1
    assert payload["videos"][0]["url"].startswith("https://huggingface.co/datasets/")


def test_explorer_dataset_info_uses_remote_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "build_remote_dataset_info",
        lambda dataset: {
            "name": dataset,
            "total_episodes": 3,
            "total_frames": 42,
            "fps": 30,
            "episode_lengths": [10, 12, 20],
            "features": ["action", "observation.state"],
            "robot_type": "so101",
            "source_dataset": dataset,
        },
    )

    response = client.get("/api/explorer/dataset-info", params={"dataset": "cadene/droid_1.0.1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "cadene/droid_1.0.1"
    assert payload["total_episodes"] == 3
