from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .diagnosis import verify_repaired_dataset
from .io import (
    DEFAULT_VIDEO_PATH,
    build_video_path,
    coerce_recovery_value,
    get_video_keys,
    get_visual_keys,
    list_episode_dirs,
    list_frame_pngs,
    load_info,
    load_png_copy,
    normalize_feature_shapes,
    read_recovery_rows,
    safe_read_parquet_metadata,
    safe_read_parquet_table,
    sanitize_jsonl_line,
    scan_parquet_files,
    write_info,
)
from .lerobot_adapter import LeRobotDatasetAdapter
from .types import SKIP_FRAME_KEYS, DamageType, DiagnosisResult, RepairResult

log = logging.getLogger(__name__)


def prepare_output_dir(dataset_dir: Path, force: bool) -> tuple[Path, bool]:
    out_dir = dataset_dir.parent / f"{dataset_dir.name}_repaired"
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
        return out_dir, True
    if out_dir.exists():
        return out_dir, False
    return out_dir, True


def get_single_episode_name(images_dir: Path, image_key: str) -> str:
    episode_dirs = list_episode_dirs(images_dir / image_key)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode image directories found for {image_key} under {images_dir}")
    return episode_dirs[0].name


def build_frame_dict(
    *,
    recovery_row: dict[str, Any],
    features: dict[str, Any],
    images_dir: Path,
    image_keys: list[str],
    episode_name_by_key: dict[str, str],
    frame_index: int,
    task: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {"task": task}
    for key, feature in features.items():
        if key in SKIP_FRAME_KEYS:
            continue
        if key in image_keys:
            png_path = images_dir / key / episode_name_by_key[key] / f"frame-{frame_index:06d}.png"
            frame[key] = load_png_copy(png_path)
            continue
        if key in recovery_row:
            frame[key] = coerce_recovery_value(recovery_row[key], feature)
    return frame


def copy_critical_phase_intervals(src_dir: Path, dst_dir: Path, max_frames: int | None = None) -> None:
    src_path = src_dir / "critical_phase_intervals.json"
    if not src_path.exists():
        return
    intervals = json.loads(src_path.read_text(encoding="utf-8"))
    if max_frames is not None:
        truncated: list[dict[str, Any]] = []
        for interval in intervals:
            item = dict(interval)
            if item["start_frame"] >= max_frames:
                continue
            if item["end_frame"] > max_frames:
                item["end_frame"] = max_frames
            truncated.append(item)
        intervals = truncated
    (dst_dir / "critical_phase_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_episode_index(episode_dir: Path) -> int:
    return int(episode_dir.name.split("-")[-1])


def patch_episodes_video_columns(dataset_dir: Path, video_keys: list[str], n_frames: int, fps: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_parquets = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episode_parquets:
        return

    for episode_path in episode_parquets:
        table = pq.read_table(episode_path)
        to_timestamp = (n_frames - 1) / fps if fps > 0 else 0.0
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            if f"{prefix}/chunk_index" in table.column_names:
                continue
            n_rows = len(table)
            table = table.append_column(f"{prefix}/chunk_index", pa.array([0] * n_rows, type=pa.int64()))
            table = table.append_column(f"{prefix}/file_index", pa.array([0] * n_rows, type=pa.int64()))
            table = table.append_column(
                f"{prefix}/from_timestamp", pa.array([0.0] * n_rows, type=pa.float64())
            )
            table = table.append_column(
                f"{prefix}/to_timestamp", pa.array([to_timestamp] * n_rows, type=pa.float64())
            )
        pq.write_table(table, episode_path)


def add_frames_from_recovery(
    *,
    dataset: Any,
    recovery_rows: list[dict[str, Any]],
    features: dict[str, Any],
    images_dir: Path,
    image_keys: list[str],
    episode_name_by_key: dict[str, str],
    task: str,
) -> int:
    actual_frames = 0
    for frame_index, recovery_row in enumerate(recovery_rows):
        frame = build_frame_dict(
            recovery_row=recovery_row,
            features=features,
            images_dir=images_dir,
            image_keys=image_keys,
            episode_name_by_key=episode_name_by_key,
            frame_index=frame_index,
            task=task,
        )
        if any(frame.get(key) is None for key in image_keys):
            break
        dataset.add_frame(frame)
        actual_frames += 1
    return actual_frames


class DatasetRepairService:
    def __init__(self, adapter: LeRobotDatasetAdapter | None = None) -> None:
        self._adapter = adapter or LeRobotDatasetAdapter()

    def repair(
        self,
        diagnosis: DiagnosisResult,
        *,
        task: str,
        vcodec: str,
        dry_run: bool,
        force: bool,
    ) -> RepairResult:
        dataset_dir = diagnosis.dataset_dir
        damage = diagnosis.damage_type

        if damage == DamageType.HEALTHY:
            return RepairResult(dataset_dir, damage, "healthy")
        if damage == DamageType.EMPTY_SHELL:
            return RepairResult(dataset_dir, damage, "skipped", error="empty shell -- nothing to recover")
        if not diagnosis.repairable:
            return RepairResult(dataset_dir, damage, "skipped", error="unrepairable")
        if dry_run:
            return RepairResult(dataset_dir, damage, "skipped", error="dry run")

        result = self._dispatch_repair(diagnosis, task=task, vcodec=vcodec, force=force)
        if result.outcome != "repaired":
            return result

        verify_dir = dataset_dir
        if damage in {DamageType.CRASH_NO_SAVE, DamageType.TMP_VIDEOS_STUCK}:
            verify_dir = dataset_dir.parent / f"{dataset_dir.name}_repaired"
        verify_errors = verify_repaired_dataset(verify_dir)
        if not verify_errors:
            return result
        return RepairResult(dataset_dir, damage, "failed", error="; ".join(verify_errors))

    def _dispatch_repair(
        self,
        diagnosis: DiagnosisResult,
        *,
        task: str,
        vcodec: str,
        force: bool,
    ) -> RepairResult:
        dataset_dir = diagnosis.dataset_dir
        damage = diagnosis.damage_type
        if damage == DamageType.CRASH_NO_SAVE:
            _, rebuilt = self._repair_crash_no_save(dataset_dir, diagnosis, task=task, vcodec=vcodec, force=force)
            if not rebuilt:
                return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
        elif damage == DamageType.TMP_VIDEOS_STUCK:
            _, rebuilt = self._repair_tmp_videos_stuck(dataset_dir, diagnosis, task=task, force=force)
            if not rebuilt:
                return RepairResult(dataset_dir, damage, "skipped", error="_repaired already exists")
        elif damage == DamageType.PARQUET_NO_VIDEO:
            self._repair_parquet_no_video(dataset_dir, vcodec=vcodec)
        elif damage == DamageType.META_STALE:
            self._repair_meta_stale(dataset_dir)
        elif damage == DamageType.FRAME_MISMATCH:
            self._repair_frame_mismatch(dataset_dir, diagnosis)
        elif damage == DamageType.MISSING_CP:
            self._repair_missing_cp(dataset_dir, diagnosis)
        return RepairResult(dataset_dir, damage, "repaired")

    def _repair_crash_no_save(
        self,
        dataset_dir: Path,
        diagnosis: DiagnosisResult,
        *,
        task: str,
        vcodec: str,
        force: bool,
    ) -> tuple[Path, bool]:
        out_dir, should_proceed = prepare_output_dir(dataset_dir, force)
        if not should_proceed:
            return out_dir, False

        info = load_info(dataset_dir)
        recovery_rows = read_recovery_rows(dataset_dir)
        features = normalize_feature_shapes(info["features"])
        image_keys = get_visual_keys(info)
        images_dir = dataset_dir / "images"
        n_usable = min(len(recovery_rows), diagnosis.details["min_images_per_camera"])
        if n_usable <= 0:
            raise ValueError(f"No usable frames available to rebuild {dataset_dir}")

        dataset = self._adapter.create_dataset(
            repo_id=f"local/{out_dir.name}",
            fps=int(info["fps"]),
            root=out_dir,
            robot_type=info.get("robot_type"),
            features=features,
            use_videos=bool(image_keys),
            vcodec=vcodec,
        )
        episode_name_by_key = {key: get_single_episode_name(images_dir, key) for key in image_keys}
        actual_frames = add_frames_from_recovery(
            dataset=dataset,
            recovery_rows=recovery_rows[:n_usable],
            features=features,
            images_dir=images_dir,
            image_keys=image_keys,
            episode_name_by_key=episode_name_by_key,
            task=task,
        )
        dataset.save_episode()
        dataset.finalize()
        copy_critical_phase_intervals(dataset_dir, out_dir, max_frames=actual_frames)
        return out_dir, True

    def _repair_tmp_videos_stuck(
        self,
        dataset_dir: Path,
        diagnosis: DiagnosisResult,
        *,
        task: str,
        force: bool,
    ) -> tuple[Path, bool]:
        out_dir, should_proceed = prepare_output_dir(dataset_dir, force)
        if not should_proceed:
            return out_dir, False

        info = load_info(dataset_dir)
        recovery_rows = read_recovery_rows(dataset_dir)
        features = normalize_feature_shapes(info["features"])
        video_keys = get_video_keys(info)
        tmp_videos: dict[str, Path] = diagnosis.details["tmp_videos"]
        non_video_features = {
            key: value for key, value in features.items() if value.get("dtype") not in {"video", "image"}
        }

        dataset = self._adapter.create_dataset(
            repo_id=f"local/{out_dir.name}",
            fps=int(info["fps"]),
            root=out_dir,
            robot_type=info.get("robot_type"),
            features=non_video_features,
            use_videos=False,
            vcodec="auto",
        )
        for frame_index, recovery_row in enumerate(recovery_rows):
            dataset.add_frame(
                build_frame_dict(
                    recovery_row=recovery_row,
                    features=non_video_features,
                    images_dir=dataset_dir / "images",
                    image_keys=[],
                    episode_name_by_key={},
                    frame_index=frame_index,
                    task=task,
                )
            )
        dataset.save_episode()
        dataset.finalize()

        for video_key in video_keys:
            if video_key not in tmp_videos:
                continue
            src_mp4 = tmp_videos[video_key]
            dst_mp4 = build_video_path(out_dir, info, video_key, episode_index=0)
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mp4, dst_mp4)

        out_info = load_info(out_dir)
        out_info["features"] = dict(info["features"])
        out_info["total_episodes"] = 1
        out_info["total_frames"] = len(recovery_rows)
        out_info["video_path"] = info.get("video_path") or DEFAULT_VIDEO_PATH
        write_info(out_dir, out_info)
        patch_episodes_video_columns(out_dir, video_keys, len(recovery_rows), int(info["fps"]))
        copy_critical_phase_intervals(dataset_dir, out_dir)
        return out_dir, True

    def _repair_parquet_no_video(self, dataset_dir: Path, *, vcodec: str) -> None:
        info = load_info(dataset_dir)
        images_dir = dataset_dir / "images"
        fps = int(info["fps"])
        for video_key in get_video_keys(info):
            episode_dirs = list_episode_dirs(images_dir / video_key)
            if not episode_dirs:
                raise FileNotFoundError(
                    f"No PNG episode directories found for video key {video_key} in {dataset_dir}"
                )
            for episode_dir in episode_dirs:
                self._adapter.encode_video_frames(
                    frames_dir=episode_dir,
                    video_path=build_video_path(dataset_dir, info, video_key, parse_episode_index(episode_dir)),
                    fps=fps,
                    vcodec=vcodec,
                )

    def _patch_info_totals_from_parquet(self, dataset_dir: Path) -> tuple[int, int]:
        info = load_info(dataset_dir)
        _n_files, total_episodes, total_frames = scan_parquet_files(dataset_dir)
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{total_episodes}"} if total_episodes > 0 else {}
        write_info(dataset_dir, info)
        return total_episodes, total_frames

    def _repair_meta_stale(self, dataset_dir: Path) -> None:
        self._patch_info_totals_from_parquet(dataset_dir)

    def _repair_missing_cp(self, dataset_dir: Path, diagnosis: DiagnosisResult) -> None:
        cp_path = dataset_dir / "critical_phase_intervals.json"
        cp_path.write_text(
            json.dumps(diagnosis.details["log_cp_intervals"], indent=2) + "\n",
            encoding="utf-8",
        )

    def _repair_frame_mismatch(self, dataset_dir: Path, diagnosis: DiagnosisResult) -> None:
        n_keep = diagnosis.details["truncate_target_frames"]
        if n_keep <= 0:
            raise ValueError(f"No positive truncate target for {dataset_dir}")
        self._truncate_recovery_jsonl(dataset_dir, n_keep)
        self._truncate_images(dataset_dir, n_keep)
        self._truncate_parquet(dataset_dir, n_keep)
        if diagnosis.details["n_parquet_rows"] > 0:
            self._patch_info_totals_from_parquet(dataset_dir)

    def _truncate_recovery_jsonl(self, dataset_dir: Path, n_keep: int) -> None:
        recovery_path = dataset_dir / "recovery_frames.jsonl"
        if not recovery_path.exists():
            return
        kept_lines: list[str] = []
        with recovery_path.open() as handle:
            for line in handle:
                sanitized = sanitize_jsonl_line(line)
                if not sanitized:
                    break
                kept_lines.append(f"{sanitized}\n")
                if len(kept_lines) >= n_keep:
                    break
        recovery_path.write_text("".join(kept_lines), encoding="utf-8")

    def _truncate_images(self, dataset_dir: Path, n_keep: int) -> None:
        for camera_dir in list_episode_dirs(dataset_dir / "images"):
            seen = 0
            for episode_dir in list_episode_dirs(camera_dir):
                for png_path in list_frame_pngs(episode_dir):
                    seen += 1
                    if seen > n_keep:
                        png_path.unlink()

    def _truncate_parquet(self, dataset_dir: Path, n_keep: int) -> None:
        import pyarrow.parquet as pq

        remaining = n_keep
        for parquet_path in sorted((dataset_dir / "data").rglob("*.parquet")):
            metadata = safe_read_parquet_metadata(parquet_path)
            if metadata is None:
                parquet_path.unlink()
                continue
            if remaining <= 0:
                parquet_path.unlink()
                continue
            if metadata.num_rows <= remaining:
                remaining -= metadata.num_rows
                continue
            table = safe_read_parquet_table(parquet_path)
            if table is None:
                parquet_path.unlink()
                continue
            pq.write_table(table.slice(0, remaining), parquet_path)
            remaining = 0


_REPAIR_SERVICE = DatasetRepairService()


def repair_dataset(
    diagnosis: DiagnosisResult,
    *,
    task: str,
    vcodec: str,
    dry_run: bool,
    force: bool,
) -> RepairResult:
    return _REPAIR_SERVICE.repair(
        diagnosis,
        task=task,
        vcodec=vcodec,
        dry_run=dry_run,
        force=force,
    )
