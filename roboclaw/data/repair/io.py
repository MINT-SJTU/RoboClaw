from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

_CP_END_RE = re.compile(
    r"\[CP\] END at episode (\d+), frame (\d+) "
    r"\(segment: (\d+)-(\d+), \d+ frames(?:, outcome=(\w+))?\)"
)
_PARQUET_ERRORS = (OSError, pa.lib.ArrowException)


def sanitize_jsonl_line(line: str) -> str:
    return line.replace("\x00", "").strip()


def parse_cp_from_log(log_path: Path) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    with log_path.open() as handle:
        for line in handle:
            match = _CP_END_RE.search(line)
            if match is None:
                continue
            intervals.append(
                {
                    "episode_index": int(match.group(1)),
                    "start_frame": int(match.group(3)),
                    "end_frame": int(match.group(4)),
                    "outcome": match.group(5),
                }
            )
    return intervals


def find_log_for_dataset(dataset_dir: Path) -> Path | None:
    log_path = dataset_dir.parent / f"{dataset_dir.name}.log"
    return log_path if log_path.exists() else None


def load_info(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "meta" / "info.json").open() as handle:
        return json.load(handle)


def write_info(dataset_dir: Path, info: dict[str, Any]) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def read_recovery_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    recovery_path = dataset_dir / "recovery_frames.jsonl"
    if not recovery_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with recovery_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            sanitized = sanitize_jsonl_line(line)
            if not sanitized:
                break
            try:
                rows.append(json.loads(sanitized))
            except json.JSONDecodeError as exc:
                log.warning("Corrupt JSON in %s at line %d: %s", recovery_path, line_number, exc)
                break
    return rows


def normalize_feature_shapes(features: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(features)
    for feature in normalized.values():
        if "shape" in feature:
            feature["shape"] = tuple(feature["shape"])
    return normalized


def coerce_recovery_value(value: Any, feature: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return np.array(value, dtype=np.dtype(feature["dtype"]))
    return value


def list_episode_dirs(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return [path for path in sorted(parent.iterdir()) if path.is_dir()]


def list_frame_pngs(episode_dir: Path) -> list[Path]:
    return sorted(episode_dir.glob("frame-*.png"))


def count_images_per_camera(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    images_dir = dataset_dir / "images"
    for camera_dir in list_episode_dirs(images_dir):
        counts[camera_dir.name] = sum(
            len(list_frame_pngs(episode_dir)) for episode_dir in list_episode_dirs(camera_dir)
        )
    return counts


def min_images_per_camera(images_per_camera: dict[str, int]) -> int:
    return min(images_per_camera.values()) if images_per_camera else 0


def count_video_files(dataset_dir: Path) -> int:
    videos_dir = dataset_dir / "videos"
    return len(list(videos_dir.rglob("*.mp4"))) if videos_dir.exists() else 0


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, value in info["features"].items() if value.get("dtype") == "video"]


def get_visual_keys(info: dict[str, Any]) -> list[str]:
    return [
        key for key, value in info["features"].items() if value.get("dtype") in {"image", "video"}
    ]


def safe_read_parquet_metadata(path: Path) -> pq.FileMetaData | None:
    try:
        return pq.read_metadata(path)
    except _PARQUET_ERRORS as exc:
        log.warning("Corrupt parquet file (cannot read metadata): %s: %s", path, exc)
        return None


def safe_read_parquet_table(path: Path, columns: list[str] | None = None) -> pa.Table | None:
    try:
        return pq.read_table(path, columns=columns)
    except _PARQUET_ERRORS as exc:
        log.warning("Corrupt parquet file (cannot read table): %s: %s", path, exc)
        return None


def scan_parquet_files(dataset_dir: Path) -> tuple[int, int, int]:
    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    episode_indices: set[int] = set()
    total_rows = 0
    valid_files = 0
    for parquet_path in parquet_files:
        metadata = safe_read_parquet_metadata(parquet_path)
        if metadata is None:
            continue
        total_rows += metadata.num_rows
        valid_files += 1
        table = safe_read_parquet_table(parquet_path, columns=["episode_index"])
        if table is not None:
            episode_indices.update(int(value) for value in table["episode_index"].to_pylist())
    return valid_files, len(episode_indices), total_rows


def load_png_copy(png_path: Path) -> PIL.Image.Image | None:
    try:
        with PIL.Image.open(png_path) as image:
            return image.copy()
    except (OSError, PIL.UnidentifiedImageError):
        log.warning("Corrupt PNG: %s", png_path)
        return None


def build_video_path(
    dataset_dir: Path,
    info: dict[str, Any],
    video_key: str,
    episode_index: int,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path") or DEFAULT_VIDEO_PATH
    chunk_index = episode_index // chunks_size
    file_index = episode_index % chunks_size
    return dataset_dir / template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def is_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def find_datasets(target: Path) -> list[Path]:
    if is_dataset_dir(target):
        return [target]
    if not target.is_dir():
        return []
    return [entry for entry in sorted(target.iterdir()) if entry.is_dir() and is_dataset_dir(entry)]
