from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.explorer import remote as remote_explorer
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
        lambda dataset, episode_index, preview_only=False: {
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


def test_explorer_episode_passes_preview_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    calls: dict[str, bool] = {}

    def _fake_detail(dataset: str, episode_index: int, *, preview_only: bool = False) -> dict:
        calls["preview_only"] = preview_only
        return {
            "episode_index": episode_index,
            "summary": {
                "row_count": 12 if preview_only else 500,
                "fps": 30,
                "duration_s": 0.4,
                "video_count": 0,
            },
            "sample_rows": [],
            "joint_trajectory": {
                "x_axis_key": "timestamp",
                "x_values": [],
                "time_values": [],
                "frame_values": [],
                "joint_trajectories": [],
                "sampled_points": 0,
                "total_points": 0,
            },
            "videos": [],
        }

    monkeypatch.setattr(explorer_routes, "load_remote_episode_detail", _fake_detail)

    response = client.get(
        "/api/explorer/episode",
        params={
            "dataset": "cadene/droid_1.0.1",
            "episode_index": 0,
            "preview_only": "true",
        },
    )

    assert response.status_code == 200
    assert response.json()["summary"]["row_count"] == 12
    assert calls["preview_only"] is True


def test_explorer_episode_accepts_preview_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    calls: dict[str, bool] = {}

    def _fake_detail(dataset: str, episode_index: int, *, preview_only: bool = False) -> dict:
        calls["preview_only"] = preview_only
        return {
            "episode_index": episode_index,
            "summary": {
                "row_count": 12,
                "fps": 30,
                "duration_s": 0.4,
                "video_count": 0,
            },
            "sample_rows": [],
            "joint_trajectory": {
                "x_axis_key": "timestamp",
                "x_values": [],
                "time_values": [],
                "frame_values": [],
                "joint_trajectories": [],
                "sampled_points": 0,
                "total_points": 0,
            },
            "videos": [],
        }

    monkeypatch.setattr(explorer_routes, "load_remote_episode_detail", _fake_detail)

    response = client.get(
        "/api/explorer/episode",
        params={
            "dataset": "cadene/droid_1.0.1",
            "episode_index": 0,
            "preview": "true",
        },
    )

    assert response.status_code == 200
    assert calls["preview_only"] is True


def test_explorer_dataset_suggestions_use_remote_search(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "search_remote_datasets",
        lambda query, limit: [{"id": f"{query}-{limit}"}],
    )

    response = client.get("/api/explorer/datasets", params={"query": "droid", "limit": 3})

    assert response.status_code == 200
    assert response.json() == [{"id": "droid-3"}]


def test_explorer_datasets_can_list_local_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "list_local_dataset_options",
        lambda: [
            {
                "id": "workspace/keep",
                "label": "Keep",
                "path": "/tmp/keep",
                "source": "local",
            },
            {
                "id": "workspace/drop",
                "label": "Drop",
                "path": "/tmp/drop",
                "source": "local",
            },
        ],
    )

    response = client.get(
        "/api/explorer/datasets",
        params={"source": "local", "query": "keep"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "workspace/keep",
            "label": "Keep",
            "path": "/tmp/keep",
            "source": "local",
        }
    ]


def test_explorer_local_directory_session_rejects_path_escape() -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    response = client.post(
        "/api/explorer/local-directory-session",
        data={"relative_paths": "../escape.txt"},
        files={"files": ("escape.txt", b"x")},
    )

    assert response.status_code == 400


def test_explorer_local_video_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    root = tmp_path / "dataset"
    sibling = tmp_path / "dataset_evil"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    sibling.mkdir()
    (sibling / "clip.mp4").write_bytes(b"not-video")

    response = client.get(
        "/api/explorer/local-video/%2E%2E/dataset_evil/clip.mp4",
        params={"source": "path", "dataset_path": str(root)},
    )

    assert response.status_code == 403


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


def test_remote_shared_paths_use_lerobot_default_templates() -> None:
    info = {
        "features": {
            "observation.images.front": {"dtype": "video"},
        },
        "data_path": "",
        "video_path": "",
    }
    episode_meta = {
        "data/chunk_index": 1,
        "data/file_index": 2,
        "videos/observation.images.front/chunk_index": 3,
        "videos/observation.images.front/file_index": 4,
        "videos/observation.images.front/from_timestamp": 1.5,
        "videos/observation.images.front/to_timestamp": 2.5,
    }

    assert remote_explorer._resolve_shared_data_path(info, episode_meta) == (
        "data/chunk-001/file-002.parquet"
    )

    videos = remote_explorer._resolve_shared_video_clips("org/dataset", info, episode_meta)

    assert videos == [
        {
            "path": "videos/observation.images.front/chunk-003/file-004.mp4",
            "url": "https://huggingface.co/datasets/org/dataset/resolve/main/videos/observation.images.front/chunk-003/file-004.mp4",
            "stream": "front",
            "from_timestamp": 1.5,
            "to_timestamp": 2.5,
        }
    ]


def test_viewer_fetch_episode_rows_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_explorer, "_VIEWER_PAGE_SIZE", 2)
    requested_urls: list[str] = []

    def _fake_viewer_fetch_json(url: str) -> dict:
        requested_urls.append(url)
        if "offset=0&length=2" in url:
            return {"rows": [{"row": {"index": 0}}, {"row": {"index": 1}}]}
        if "offset=2&length=1" in url:
            return {"rows": [{"row": {"index": 2}}]}
        return {"rows": []}

    monkeypatch.setattr(remote_explorer, "_viewer_fetch_json", _fake_viewer_fetch_json)

    rows = remote_explorer._viewer_fetch_episode_rows(
        "org/dataset",
        "default",
        "train",
        7,
        length=3,
    )

    assert rows == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert len(requested_urls) == 2


def test_viewer_fetch_episode_rows_stays_under_viewer_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def _fake_viewer_fetch_json(url: str) -> dict:
        requested_urls.append(url)
        if "offset=0&length=100" in url:
            return {"rows": [{"row": {"index": index}} for index in range(100)]}
        if "offset=100&length=1" in url:
            return {"rows": [{"row": {"index": 100}}]}
        return {"rows": []}

    monkeypatch.setattr(remote_explorer, "_viewer_fetch_json", _fake_viewer_fetch_json)

    rows = remote_explorer._viewer_fetch_episode_rows(
        "org/dataset",
        "default",
        "train",
        7,
        length=101,
    )

    assert len(rows) == 101
    assert len(requested_urls) == 2


def test_shared_episode_returns_preview_when_viewer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = {
        "dataset": "org/dataset",
        "siblings": [],
        "info": {
            "fps": 30,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {
                "observation.images.front": {"dtype": "video"},
                "action": {"dtype": "float32", "shape": [2]},
                "observation.state": {"dtype": "float32", "shape": [2]},
            },
        },
        "stats": {},
        "episodes_meta": [
            {
                "episode_index": 0,
                "length": 12,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
            }
        ],
    }

    monkeypatch.setattr(remote_explorer, "get_remote_dataset_artifacts", lambda dataset: artifacts)
    monkeypatch.setattr(
        remote_explorer,
        "_viewer_get_split",
        lambda dataset: (_ for _ in ()).throw(remote_explorer.httpx.ConnectError("down")),
    )
    monkeypatch.setattr(
        remote_explorer,
        "_fetch_optional_bytes",
        lambda url: pytest.fail("shared parquet fallback should not fetch whole files"),
    )

    payload = remote_explorer.load_remote_episode_detail("org/dataset", 0)

    assert payload["summary"]["row_count"] == 12
    assert payload["sample_rows"] == []
    assert payload["joint_trajectory"]["total_points"] == 0
