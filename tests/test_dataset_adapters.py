from __future__ import annotations

import json
from pathlib import Path

from roboclaw.data.curation.validators import load_episode_data, run_quality_validators
from roboclaw.data.dataset_adapters import LeRobotAdapter, MappingAdapter, resolve_dataset_adapter
from roboclaw.data.dataset_adapters import lerobot as lerobot_adapter


def _write_lerobot_dataset(root: Path) -> Path:
    dataset_path = root / "lerobot-demo"
    (dataset_path / "meta").mkdir(parents=True)
    parquet_path = dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"placeholder")
    video_dir = dataset_path / "videos" / "chunk-000" / "episode_000000"
    video_dir.mkdir(parents=True)
    (video_dir / "front.mp4").write_bytes(b"")
    (dataset_path / "meta" / "info.json").write_text(
        json.dumps({
            "total_episodes": 1,
            "fps": 30,
            "robot_type": "so101",
            "features": {
                "action": {"names": ["j1", "j2"]},
                "observation.state": {"names": ["j1", "j2"]},
            },
        }),
        encoding="utf-8",
    )
    (dataset_path / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 1.0, "task": "pick"}) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def test_lerobot_adapter_preserves_episode_shape(tmp_path, monkeypatch) -> None:
    dataset_path = _write_lerobot_dataset(tmp_path)
    rows = [
        {"timestamp": 0.0, "action": [0.0, 0.0], "observation.state": [0.0, 0.0]},
        {"timestamp": 0.05, "action": [0.1, 0.1], "observation.state": [0.1, 0.1]},
    ]
    monkeypatch.setattr(lerobot_adapter, "read_parquet_rows", lambda _path: rows)

    adapter = resolve_dataset_adapter(dataset_path)
    data = load_episode_data(dataset_path, 0)

    assert isinstance(adapter, LeRobotAdapter)
    assert data["info"]["robot_type"] == "so101"
    assert data["episode_meta"]["task"] == "pick"
    assert data["rows"] == rows
    assert data["video_files"] == [dataset_path / "videos" / "chunk-000" / "episode_000000" / "front.mp4"]


def test_mapping_adapter_maps_jsonl_rows_to_canonical_fields(tmp_path) -> None:
    dataset_path = tmp_path / "custom-demo"
    (dataset_path / "data").mkdir(parents=True)
    rows = [
        {
            "episode_id": 0,
            "time_sec": 0.0,
            "joint_cmd": [0.0, 0.0],
            "robot": {"qpos": [0.0, 0.0]},
            "task_text": "pick up the cup",
        },
        {
            "episode_id": 0,
            "time_sec": 0.05,
            "joint_cmd": [0.1, 0.2],
            "robot": {"qpos": [0.1, 0.2]},
            "task_text": "pick up the cup",
        },
    ]
    (dataset_path / "data" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    mapping = {
        "episode_index_field": "episode_id",
        "fields": {
            "timestamp": "time_sec",
            "action": "joint_cmd",
            "observation.state": "robot.qpos",
            "language_instruction": "task_text",
        },
    }

    adapter = MappingAdapter(dataset_path, mapping)
    data = adapter.load_episode(0)

    assert adapter.list_episodes() == [0]
    assert data["rows"] == [
        {
            "timestamp": 0.0,
            "action": [0.0, 0.0],
            "observation.state": [0.0, 0.0],
            "language_instruction": "pick up the cup",
        },
        {
            "timestamp": 0.05,
            "action": [0.1, 0.2],
            "observation.state": [0.1, 0.2],
            "language_instruction": "pick up the cup",
        },
    ]
    assert data["episode_meta"]["task"] == "pick up the cup"
    assert data["parquet_path"] == dataset_path / "data" / "episodes.jsonl"


def test_mapping_adapter_can_drive_existing_quality_validators(tmp_path) -> None:
    dataset_path = tmp_path / "custom-demo"
    (dataset_path / "data").mkdir(parents=True)
    rows = [
        {"episode_id": 0, "time_sec": 0.00, "joint_cmd": [0.0, 0.0]},
        {"episode_id": 0, "time_sec": 0.05, "joint_cmd": [0.1, 0.2]},
        {"episode_id": 0, "time_sec": 0.10, "joint_cmd": [0.2, 0.4]},
    ]
    (dataset_path / "data" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset_mapping.json").write_text(
        json.dumps({
            "episode_index_field": "episode_id",
            "fields": {
                "timestamp": "time_sec",
                "action": "joint_cmd",
            },
        }),
        encoding="utf-8",
    )

    result = run_quality_validators(
        dataset_path,
        0,
        selected_validators=["timing", "action"],
        threshold_overrides={"action_min_duration_s": 0.0, "action_max_velocity_rad_s": 10.0},
    )

    assert result["validators"]["timing"]["passed"] is True
    assert result["validators"]["action"]["passed"] is True


def test_mapping_adapter_reports_missing_source_fields(tmp_path) -> None:
    dataset_path = tmp_path / "custom-demo"
    dataset_path.mkdir(parents=True)
    (dataset_path / "episodes.jsonl").write_text(
        json.dumps({"time_sec": 0.0}) + "\n",
        encoding="utf-8",
    )
    adapter = MappingAdapter(
        dataset_path,
        {"fields": {"timestamp": "time_sec", "action": "joint_cmd"}},
    )

    try:
        adapter.load_episode(0)
    except ValueError as exc:
        assert "joint_cmd" in str(exc)
    else:
        raise AssertionError("expected missing source field error")
