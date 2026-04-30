from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from roboclaw.data.repair.diagnosis import diagnose_dataset
from roboclaw.data.repair.types import DamageType


def _write_info(dataset_dir: Path, **overrides: object) -> None:
    info = {
        "total_episodes": 1,
        "total_frames": 3,
        "fps": 30,
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [64, 64, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    info.update(overrides)
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _write_recovery(dataset_dir: Path, count: int) -> None:
    rows = [json.dumps({"observation.state": [float(index), float(index + 1)]}) for index in range(count)]
    (dataset_dir / "recovery_frames.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_images(dataset_dir: Path, count: int, camera: str = "observation.images.front") -> None:
    image_dir = dataset_dir / "images" / camera / "episode-000000"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (8, 8), (index, index, index)).save(image_dir / f"frame-{index:06d}.png")


def _write_parquet(dataset_dir: Path, rows: int, episodes: list[int] | None = None) -> None:
    if episodes is None:
        episodes = [0] * rows
    data_dir = dataset_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "episode_index": episodes,
            "observation.state": [[0.0, 1.0] for _ in episodes],
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")


def _write_video(dataset_dir: Path, episode_index: int = 0, camera: str = "observation.images.front") -> None:
    video_dir = dataset_dir / "videos" / camera / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"file-{episode_index:03d}.mp4").write_bytes(b"mp4")


class TestDatasetDiagnosis:
    def test_tmp_videos_stuck_wins_before_crash_no_save(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "tmp_stuck"
        _write_info(dataset_dir, total_episodes=0, total_frames=0)
        _write_recovery(dataset_dir, 2)
        tmp_dir = dataset_dir / "tmpabc"
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "observation.images.front_000.mp4").write_bytes(b"mp4")

        diagnosis = diagnose_dataset(dataset_dir)

        assert diagnosis.damage_type is DamageType.TMP_VIDEOS_STUCK
        assert diagnosis.repairable is True

    def test_missing_cp_detected_from_log_when_data_present(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "missing_cp"
        _write_info(dataset_dir)
        _write_parquet(dataset_dir, 3)
        _write_images(dataset_dir, 3)
        _write_video(dataset_dir)
        (dataset_dir.parent / "missing_cp.log").write_text(
            "[CP] END at episode 0, frame 2 (segment: 1-2, 1 frames, outcome=success)\n",
            encoding="utf-8",
        )

        diagnosis = diagnose_dataset(dataset_dir)

        assert diagnosis.damage_type is DamageType.MISSING_CP
        assert diagnosis.details["n_log_cp"] == 1

    def test_frame_mismatch_uses_smallest_available_count(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "frame_mismatch"
        _write_info(dataset_dir, total_frames=4)
        _write_recovery(dataset_dir, 4)
        _write_images(dataset_dir, 3)
        _write_parquet(dataset_dir, 2)
        _write_video(dataset_dir)

        diagnosis = diagnose_dataset(dataset_dir)

        assert diagnosis.damage_type is DamageType.FRAME_MISMATCH
        assert diagnosis.details["truncate_target_frames"] == 2
