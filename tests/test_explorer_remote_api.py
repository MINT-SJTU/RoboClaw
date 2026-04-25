# ruff: noqa: E402, I001

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.data.explorer import remote as remote_explorer
from roboclaw.http.routes import explorer as explorer_routes


def _parquet_bytes(rows: list[dict]) -> bytes:
    table = pa.Table.from_pylist(rows)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


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

    called: dict[str, object] = {}

    monkeypatch.setattr(
        explorer_routes,
        "load_remote_episode_detail",
        lambda dataset, episode_index, preview_only=False: called.update({
            "dataset": dataset,
            "episode_index": episode_index,
            "preview_only": preview_only,
        }) or {
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
    assert called == {
        "dataset": "cadene/droid_1.0.1",
        "episode_index": 0,
        "preview_only": False,
    }


def test_explorer_episode_preview_forwards_preview_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    called: dict[str, object] = {}

    monkeypatch.setattr(
        explorer_routes,
        "load_remote_episode_detail",
        lambda dataset, episode_index, preview_only=False: called.update({
            "dataset": dataset,
            "episode_index": episode_index,
            "preview_only": preview_only,
        }) or {
            "episode_index": episode_index,
            "summary": {"row_count": 2, "fps": 30, "duration_s": 0.5, "video_count": 1},
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
            "videos": [
                {
                    "path": "videos/shared.mp4",
                    "url": "https://huggingface.co/datasets/demo/shared.mp4",
                    "stream": "front",
                    "from_timestamp": 12.5,
                    "to_timestamp": 15.0,
                }
            ],
        },
    )

    response = client.get(
        "/api/explorer/episode",
        params={"dataset": "demo/shared-layout", "episode_index": 0, "preview": True},
    )
    assert response.status_code == 200
    assert called == {
        "dataset": "demo/shared-layout",
        "episode_index": 0,
        "preview_only": True,
    }


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


def test_explorer_suggest_uses_remote_search(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    called: dict[str, object] = {}

    monkeypatch.setattr(
        explorer_routes,
        "search_remote_datasets",
        lambda query, limit=8: called.update({
            "query": query,
            "limit": limit,
        }) or [
            {"id": "Elvinky/bi-so101-fold_towel_d1"},
            {"id": "Elvinky/bi-so101-fold_towel_d0_1"},
        ],
    )

    response = client.get("/api/explorer/suggest", params={"q": "Elvinky", "limit": 6})
    assert response.status_code == 200
    assert response.json() == [
        {"id": "Elvinky/bi-so101-fold_towel_d1"},
        {"id": "Elvinky/bi-so101-fold_towel_d0_1"},
    ]
    assert called == {
        "query": "Elvinky",
        "limit": 6,
    }


def test_explorer_datasets_lists_local_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "list_local_dataset_options",
        lambda: [
            {
                "id": "local/demo",
                "label": "local/demo",
                "path": "/tmp/local/demo",
                "source": "local",
            }
        ],
    )

    response = client.get("/api/explorer/datasets", params={"source": "local"})
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "local/demo",
            "label": "local/demo",
            "path": "/tmp/local/demo",
            "source": "local",
        }
    ]


def test_explorer_suggest_filters_local_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    monkeypatch.setattr(
        explorer_routes,
        "list_local_dataset_options",
        lambda: [
            {"id": "local/demo", "label": "local/demo", "path": "/tmp/local/demo", "source": "local"},
            {"id": "other/task", "label": "other/task", "path": "/tmp/other/task", "source": "local"},
        ],
    )

    response = client.get(
        "/api/explorer/suggest",
        params={"source": "local", "q": "demo", "limit": 8},
    )
    assert response.status_code == 200
    assert response.json() == [
        {"id": "local/demo", "label": "local/demo", "path": "/tmp/local/demo", "source": "local"}
    ]


def test_explorer_episode_supports_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "datasets" / "local" / "demo"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "action": {"names": ["joint_1"]},
                    "observation.state": {"names": ["joint_1"]},
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_path / "videos").mkdir(parents=True)
    video_path = dataset_path / "videos" / "front.mp4"
    video_path.write_bytes(b"")

    monkeypatch.setattr(
        explorer_routes,
        "resolve_local_dataset_path",
        lambda _dataset: dataset_path,
    )
    monkeypatch.setattr(
        explorer_routes,
        "_local_dataset_name",
        lambda _path: "local/demo",
    )
    monkeypatch.setattr(
        explorer_routes,
        "load_episode_data",
        lambda _dataset_path, _episode_index: {
            "info": {
                "fps": 30,
                "features": {
                    "action": {"names": ["joint_1"]},
                    "observation.state": {"names": ["joint_1"]},
                },
            },
            "rows": [
                {"timestamp": 0.0, "frame_index": 0, "action": [0.1], "observation.state": [0.0]},
                {"timestamp": 1.0, "frame_index": 1, "action": [0.2], "observation.state": [0.1]},
            ],
            "video_files": [video_path],
        },
    )

    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    response = client.get(
        "/api/explorer/episode",
        params={"source": "local", "dataset": "local/demo", "episode_index": 0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["row_count"] == 2
    assert payload["videos"][0]["url"].startswith("/api/explorer/local-video/")
    assert payload["sample_rows"][0]["frame_index"] == 0


def test_explorer_prepare_remote_uses_workspace_prep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    explorer_routes.register_explorer_routes(app)
    client = TestClient(app)

    called: dict[str, object] = {}
    monkeypatch.setattr(
        explorer_routes,
        "register_remote_dataset_session",
        lambda dataset_id, include_videos=False, force=False: called.update({
            "dataset_id": dataset_id,
            "include_videos": include_videos,
            "force": force,
        }) or {
            "dataset_id": dataset_id,
            "dataset_name": dataset_id,
            "local_path": f"/tmp/{dataset_id}",
        },
    )

    response = client.post(
        "/api/explorer/prepare-remote",
        json={"dataset_id": "cadene/droid_1.0.1", "include_videos": False, "force": False},
    )
    assert response.status_code == 200
    assert response.json()["dataset_name"] == "cadene/droid_1.0.1"
    assert called == {
        "dataset_id": "cadene/droid_1.0.1",
        "include_videos": False,
        "force": False,
    }


def test_remote_episode_meta_falls_back_to_parquet_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = {
        "features": {
            "observation.images.right_front": {"dtype": "video"},
        },
    }
    rows = [
        {
            "episode_index": 0,
            "length": 12,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 12,
            "videos/observation.images.right_front/chunk_index": 0,
            "videos/observation.images.right_front/file_index": 0,
            "videos/observation.images.right_front/from_timestamp": 0.0,
            "videos/observation.images.right_front/to_timestamp": 0.4,
        },
        {
            "episode_index": 1,
            "length": 18,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 12,
            "dataset_to_index": 30,
            "videos/observation.images.right_front/chunk_index": 0,
            "videos/observation.images.right_front/file_index": 0,
            "videos/observation.images.right_front/from_timestamp": 0.4,
            "videos/observation.images.right_front/to_timestamp": 1.0,
        },
    ]
    parquet_bytes = _parquet_bytes(rows)

    def fake_fetch(url: str) -> bytes | None:
        if url.endswith("meta/episodes.jsonl"):
            return None
        if url.endswith("meta/episodes/chunk-000/file-000.parquet"):
            return parquet_bytes
        return None

    monkeypatch.setattr(remote_explorer, "_fetch_optional_bytes", fake_fetch)

    payload = remote_explorer._get_remote_episodes_meta(
        "demo/shared-layout",
        [{"rfilename": "meta/episodes/chunk-000/file-000.parquet"}],
        info,
    )

    assert [entry["episode_index"] for entry in payload] == [0, 1]
    assert [entry["length"] for entry in payload] == [12, 18]


def test_remote_episode_meta_normalizes_legacy_numeric_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = {
        "features": {
            "observation.images.front": {"dtype": "video"},
        },
    }
    parquet_bytes = _parquet_bytes([
        {
            "0": 5,
            "1": 2,
            "2": 7,
            "3": 100,
            "4": 140,
            "5": 9,
            "6": 3,
            "7": 12.5,
            "8": 15.0,
            "9": 40,
        }
    ])

    def fake_fetch(url: str) -> bytes | None:
        if url.endswith("meta/episodes.jsonl"):
            return None
        if url.endswith("meta/episodes/chunk-000/file-000.parquet"):
            return parquet_bytes
        return None

    monkeypatch.setattr(remote_explorer, "_fetch_optional_bytes", fake_fetch)

    payload = remote_explorer._get_remote_episodes_meta(
        "demo/shared-layout",
        [{"rfilename": "meta/episodes/chunk-000/file-000.parquet"}],
        info,
    )

    assert payload == [
        {
            "0": 5,
            "1": 2,
            "2": 7,
            "3": 100,
            "4": 140,
            "5": 9,
            "6": 3,
            "7": 12.5,
            "8": 15.0,
            "9": 40,
            "episode_index": 5,
            "length": 40,
            "data/chunk_index": 2,
            "data/file_index": 7,
            "dataset_from_index": 100,
            "dataset_to_index": 140,
            "video_chunk_index": 9,
            "video_file_index": 3,
            "video_from_timestamp": 12.5,
            "video_to_timestamp": 15.0,
            "videos/observation.images.front/chunk_index": 9,
            "videos/observation.images.front/file_index": 3,
            "videos/observation.images.front/from_timestamp": 12.5,
            "videos/observation.images.front/to_timestamp": 15.0,
        }
    ]


def test_select_rows_for_episode_uses_dataset_row_range_offsets() -> None:
    rows = [
        {"index": 100, "timestamp": 0.0, "frame_index": 0, "action": [0.0], "observation.state": [0.1]},
        {"timestamp": 0.5, "frame_index": 1, "action": [0.1], "observation.state": [0.2]},
        {"timestamp": 1.0, "frame_index": 2, "action": [1.0], "observation.state": [1.1]},
        {"timestamp": 1.5, "frame_index": 3, "action": [1.1], "observation.state": [1.2]},
        {"timestamp": 2.0, "frame_index": 4, "action": [2.0], "observation.state": [2.1]},
    ]

    selected = remote_explorer._select_rows_for_episode(
        rows,
        episode_index=7,
        episode_meta={"dataset_from_index": 102, "dataset_to_index": 104},
    )

    assert [row["frame_index"] for row in selected] == [2, 3]


def test_remote_episode_detail_supports_shared_file_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = "Elvinky/bi-so101-insert-screw-271ep"
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"names": ["joint_1"]},
            "observation.state": {"names": ["joint_1"]},
            "observation.images.right_front": {"dtype": "video"},
            "observation.images.right_wrist": {"dtype": "video"},
        },
    }
    episodes_meta = [
        {
            "episode_index": 1,
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 2,
            "dataset_to_index": 4,
            "videos/observation.images.right_front/chunk_index": 0,
            "videos/observation.images.right_front/file_index": 0,
            "videos/observation.images.right_front/from_timestamp": 12.5,
            "videos/observation.images.right_front/to_timestamp": 15.0,
            "videos/observation.images.right_wrist/chunk_index": 0,
            "videos/observation.images.right_wrist/file_index": 2,
            "videos/observation.images.right_wrist/from_timestamp": 9.0,
            "videos/observation.images.right_wrist/to_timestamp": 11.0,
        },
    ]
    artifacts = {
        "dataset": dataset,
        "siblings": [],
        "info": info,
        "stats": {},
        "episodes_meta": episodes_meta,
    }
    parquet_bytes = _parquet_bytes([
        {"episode_index": 0, "timestamp": 0.0, "frame_index": 0, "action": [0.1], "observation.state": [0.2]},
        {"episode_index": 0, "timestamp": 0.5, "frame_index": 1, "action": [0.2], "observation.state": [0.3]},
        {"episode_index": 1, "timestamp": 0.0, "frame_index": 0, "action": [1.1], "observation.state": [1.2]},
        {"episode_index": 1, "timestamp": 0.5, "frame_index": 1, "action": [1.2], "observation.state": [1.3]},
    ])

    monkeypatch.setattr(remote_explorer, "get_remote_dataset_artifacts", lambda _dataset: artifacts)
    monkeypatch.setattr(remote_explorer, "_viewer_get_split", lambda _dataset: ("default", "train"))
    monkeypatch.setattr(remote_explorer, "_download_repo_file_to_cache", lambda *_args, **_kwargs: None)

    def broken_viewer(*_args, **_kwargs):
        raise RuntimeError("viewer broken")

    def fake_fetch(url: str) -> bytes | None:
        if url.endswith("data/chunk-000/file-000.parquet"):
            return parquet_bytes
        return None

    monkeypatch.setattr(remote_explorer, "_viewer_fetch_episode_rows", broken_viewer)
    monkeypatch.setattr(remote_explorer, "_fetch_optional_bytes", fake_fetch)

    payload = remote_explorer.load_remote_episode_detail(dataset, 1)

    assert payload["episode_index"] == 1
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["video_count"] == 2
    assert payload["joint_trajectory"]["total_points"] == 2
    assert payload["videos"][0]["path"] == "videos/observation.images.right_front/chunk-000/file-000.mp4"
    assert payload["videos"][0]["from_timestamp"] == 12.5
    assert payload["videos"][0]["to_timestamp"] == 15.0
    assert payload["videos"][1]["path"] == "videos/observation.images.right_wrist/chunk-000/file-002.mp4"


def test_remote_episode_detail_requests_only_chart_relevant_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = "demo/shared-layout"
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [1], "names": ["joint_1"]},
            "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint_1"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]},
        },
    }
    artifacts = {
        "dataset": dataset,
        "siblings": [],
        "info": info,
        "stats": {},
        "episodes_meta": [
            {
                "episode_index": 0,
                "length": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
            }
        ],
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(remote_explorer, "get_remote_dataset_artifacts", lambda _dataset: artifacts)
    monkeypatch.setattr(remote_explorer, "_viewer_get_split", lambda _dataset: ("default", "train"))
    monkeypatch.setattr(remote_explorer, "_download_repo_file_to_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        remote_explorer,
        "_viewer_fetch_episode_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("viewer broken")),
    )
    monkeypatch.setattr(remote_explorer, "_fetch_optional_bytes", lambda _url: b"parquet")

    def fake_read_parquet(_raw: bytes, columns: list[str] | None = None) -> list[dict]:
        captured["columns"] = columns
        return [
            {
                "index": 0,
                "episode_index": 0,
                "timestamp": 0.0,
                "frame_index": 0,
                "action": [0.1],
                "observation.state": [0.2],
            },
            {
                "index": 1,
                "episode_index": 0,
                "timestamp": 0.5,
                "frame_index": 1,
                "action": [0.2],
                "observation.state": [0.3],
            },
        ]

    monkeypatch.setattr(remote_explorer, "_read_parquet_rows_from_bytes", fake_read_parquet)

    payload = remote_explorer.load_remote_episode_detail(dataset, 0)

    assert payload["summary"]["row_count"] == 2
    assert captured["columns"] is not None
    assert "index" in captured["columns"]
    assert "action" in captured["columns"]
    assert "observation.state" in captured["columns"]
    assert "observation.images.front" not in captured["columns"]


def test_read_parquet_rows_from_file_uses_episode_row_slice(tmp_path) -> None:
    path = tmp_path / "shared.parquet"
    pq.write_table(
        pa.Table.from_pylist([
            {"index": 100, "episode_index": 0, "timestamp": 0.0, "frame_index": 0, "action": [0.0], "observation.state": [0.1]},
            {"index": 101, "episode_index": 0, "timestamp": 0.5, "frame_index": 1, "action": [0.1], "observation.state": [0.2]},
            {"index": 102, "episode_index": 7, "timestamp": 1.0, "frame_index": 2, "action": [1.0], "observation.state": [1.1]},
            {"index": 103, "episode_index": 7, "timestamp": 1.5, "frame_index": 3, "action": [1.1], "observation.state": [1.2]},
            {"index": 104, "episode_index": 8, "timestamp": 2.0, "frame_index": 4, "action": [2.0], "observation.state": [2.1]},
        ]),
        path,
        row_group_size=5,
    )

    rows = remote_explorer._read_parquet_rows_from_file(
        path,
        columns=["index", "episode_index", "timestamp", "frame_index", "action", "observation.state"],
        episode_meta={"dataset_from_index": 102, "dataset_to_index": 104},
    )

    assert [row["frame_index"] for row in rows] == [2, 3]


def test_remote_episode_preview_payload_uses_episode_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = "demo/shared-layout"
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front": {"dtype": "video"},
        },
    }
    artifacts = {
        "dataset": dataset,
        "siblings": [],
        "info": info,
        "stats": {},
        "episodes_meta": [
            {
                "episode_index": 0,
                "length": 90,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 1,
                "videos/observation.images.front/from_timestamp": 12.0,
                "videos/observation.images.front/to_timestamp": 15.0,
            }
        ],
    }

    monkeypatch.setattr(remote_explorer, "get_remote_dataset_artifacts", lambda _dataset: artifacts)

    payload = remote_explorer.load_remote_episode_detail(dataset, 0, preview_only=True)

    assert payload["summary"] == {
        "row_count": 90,
        "fps": 30,
        "duration_s": 3.0,
        "video_count": 1,
    }
    assert payload["videos"][0]["path"] == "videos/observation.images.front/chunk-000/file-001.mp4"
    assert payload["videos"][0]["from_timestamp"] == 12.0
    assert payload["joint_trajectory"]["total_points"] == 0
