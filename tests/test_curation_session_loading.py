from __future__ import annotations

import json
from pathlib import Path

from roboclaw.data.curation import validators as curation_validators


def test_remote_session_downloads_into_remote_cache(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "session:remote:abc123"
    (dataset_path / "meta").mkdir(parents=True)
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "source_dataset": "cadene/droid_1.0.1",
                "chunks_size": 1000,
                "fps": 30,
                "features": {
                    "observation.images.front": {"dtype": "video"},
                },
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 1.0,
                "data/chunk_index": 0,
                "data/file_index": 7,
                "videos/observation.images.front/chunk_index": 1,
                "videos/observation.images.front/file_index": 3,
            }
        ) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        curation_validators,
        "read_session_metadata",
        lambda _handle: {"source_dataset": "cadene/droid_1.0.1"},
    )

    captured: dict[str, object] = {}

    def fake_download(dataset_id: str, relative_path: Path, *, local_root: Path | None = None) -> Path:
        captured["dataset_id"] = dataset_id
        captured["relative_path"] = relative_path.as_posix()
        captured["local_root"] = str(local_root) if local_root else None
        target = (local_root or dataset_path) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return target

    downloaded_paths: list[str] = []

    def fake_download_with_record(dataset_id: str, relative_path: Path, *, local_root: Path | None = None) -> Path:
        downloaded_paths.append(relative_path.as_posix())
        return fake_download(dataset_id, relative_path, local_root=local_root)

    monkeypatch.setattr(curation_validators, "_download_remote_file", fake_download_with_record)
    monkeypatch.setattr(curation_validators, "_read_parquet_rows", lambda _path, **_kwargs: [])

    curation_validators.load_episode_data(dataset_path, 0)

    assert captured["dataset_id"] == "cadene/droid_1.0.1"
    assert downloaded_paths == [
        "data/chunk-000/file-007.parquet",
        "videos/observation.images.front/chunk-001/file-003.mp4",
    ]
    assert captured["relative_path"] == "videos/observation.images.front/chunk-001/file-003.mp4"
    assert captured["local_root"] == str((dataset_path / ".remote-cache").resolve())


def test_episode_meta_can_load_from_parquet_and_slice_shared_data_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "session:remote:abc123"
    (dataset_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "source_dataset": "Elvinky/so101_pick_place_bottle",
                "chunks_size": 1000,
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "features": {
                    "observation.images.front": {"dtype": "video"},
                },
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    meta_parquet = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_parquet.write_bytes(b"parquet")
    data_file = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_bytes(b"parquet")
    video_file = dataset_path / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"video")

    captured_filters: list[object] = []

    def fake_read_parquet(path: Path, **kwargs) -> list[dict]:
        captured_filters.append(kwargs.get("filters"))
        if path == meta_parquet:
            return [
                {
                    "episode_index": 0,
                    "length": 769,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": 3,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                    "videos/observation.images.front/from_timestamp": 0.0,
                    "videos/observation.images.front/to_timestamp": 0.1,
                }
            ]
        if path == data_file:
            return [
                {"episode_index": 0, "index": 0, "timestamp": 0.0, "frame_index": 0},
                {"episode_index": 0, "index": 1, "timestamp": 0.0333, "frame_index": 1},
                {"episode_index": 0, "index": 2, "timestamp": 0.0666, "frame_index": 2},
                {"episode_index": 1, "index": 3, "timestamp": 0.0, "frame_index": 0},
            ]
        return []

    monkeypatch.setattr(curation_validators, "read_parquet_rows", fake_read_parquet)
    monkeypatch.setattr(
        curation_validators,
        "read_session_metadata",
        lambda _handle: {"source_dataset": "Elvinky/so101_pick_place_bottle"},
    )

    payload = curation_validators.load_episode_data(dataset_path, 0)

    assert payload["episode_meta"]["episode_index"] == 0
    assert payload["episode_meta"]["dataset_to_index"] == 3
    assert len(payload["rows"]) == 3
    assert all(row["episode_index"] == 0 for row in payload["rows"])
    assert [("index", ">=", 0), ("index", "<", 3)] in captured_filters
