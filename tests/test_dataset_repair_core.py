from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from roboclaw.data.repair.repairers import DatasetRepairService
from roboclaw.data.repair.types import DamageType, DiagnosisResult


def _write_info(dataset_dir: Path, *, total_episodes: int = 0, total_frames: int = 0) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": 30,
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [4, 4, 3], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _write_recovery(dataset_dir: Path, count: int) -> None:
    rows = [json.dumps({"observation.state": [index, index + 1]}) for index in range(count)]
    (dataset_dir / "recovery_frames.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_images(dataset_dir: Path, count: int) -> None:
    image_dir = dataset_dir / "images" / "observation.images.front" / "episode-000000"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (4, 4), (index, index, index)).save(image_dir / f"frame-{index:06d}.png")


def _write_parquet(dataset_dir: Path, episodes: list[int]) -> None:
    data_dir = dataset_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "episode_index": episodes,
            "observation.state": [[0.0, 1.0] for _ in episodes],
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")


def _write_video(dataset_dir: Path, episode_index: int = 0) -> None:
    video_dir = dataset_dir / "videos" / "observation.images.front" / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"file-{episode_index:03d}.mp4").write_bytes(b"mp4")


class TestDatasetRepairCore:
    def test_meta_stale_refreshes_info_totals(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "meta_stale"
        _write_info(dataset_dir, total_episodes=0, total_frames=0)
        _write_parquet(dataset_dir, [0, 0, 1])
        _write_video(dataset_dir, 0)
        _write_video(dataset_dir, 1)
        diagnosis = DiagnosisResult(
            dataset_dir=dataset_dir,
            damage_type=DamageType.META_STALE,
            repairable=True,
            details={"n_parquet_rows": 3},
        )

        result = DatasetRepairService().repair(
            diagnosis,
            task="task",
            vcodec="h264",
            dry_run=False,
            force=False,
        )

        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        assert result.outcome == "repaired"
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 3
        assert info["splits"] == {"train": "0:2"}

    def test_missing_cp_writes_intervals(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "missing_cp"
        _write_info(dataset_dir, total_episodes=1, total_frames=1)
        _write_parquet(dataset_dir, [0])
        _write_video(dataset_dir)
        diagnosis = DiagnosisResult(
            dataset_dir=dataset_dir,
            damage_type=DamageType.MISSING_CP,
            repairable=True,
            details={
                "n_parquet_rows": 1,
                "log_cp_intervals": [{"episode_index": 0, "start_frame": 0, "end_frame": 1, "outcome": "success"}],
                "log_path": dataset_dir.parent / "missing_cp.log",
            },
        )

        result = DatasetRepairService().repair(
            diagnosis,
            task="task",
            vcodec="h264",
            dry_run=False,
            force=False,
        )

        intervals = json.loads((dataset_dir / "critical_phase_intervals.json").read_text(encoding="utf-8"))
        assert result.outcome == "repaired"
        assert intervals[0]["outcome"] == "success"

    def test_frame_mismatch_truncates_recovery_images_and_parquet(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "frame_mismatch"
        _write_info(dataset_dir, total_episodes=1, total_frames=3)
        _write_recovery(dataset_dir, 3)
        _write_images(dataset_dir, 3)
        _write_parquet(dataset_dir, [0, 0, 0])
        _write_video(dataset_dir)
        diagnosis = DiagnosisResult(
            dataset_dir=dataset_dir,
            damage_type=DamageType.FRAME_MISMATCH,
            repairable=True,
            details={"truncate_target_frames": 2, "n_parquet_rows": 3},
        )

        result = DatasetRepairService().repair(
            diagnosis,
            task="task",
            vcodec="h264",
            dry_run=False,
            force=False,
        )

        recovery_lines = (dataset_dir / "recovery_frames.jsonl").read_text(encoding="utf-8").splitlines()
        image_files = sorted((dataset_dir / "images").rglob("frame-*.png"))
        table = pq.read_table(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        assert result.outcome == "repaired"
        assert len(recovery_lines) == 2
        assert len(image_files) == 2
        assert table.num_rows == 2
        assert info["total_frames"] == 2
