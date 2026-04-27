from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .bridge import read_parquet_rows
from .state import load_dataset_info
from .task_descriptions import payload_has_task_description
from .validators import QUALITY_THRESHOLD_DEFAULTS, safe_float

VALIDATOR_ORDER = ["metadata", "timing", "action", "visual", "depth", "ee_trajectory"]


def build_quality_defaults(dataset_path: Path, dataset_name: str | None = None) -> dict[str, Any]:
    """Build dataset-aware quality validator defaults for the Pipeline UI and AI."""
    info_path = dataset_path / "meta" / "info.json"
    info = load_dataset_info(dataset_path)
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    episode_meta = _first_episode_meta(dataset_path)
    episode_lengths = _episode_lengths(info, episode_meta, dataset_path)
    fps = safe_float(info.get("fps")) or 0.0
    median_duration = _median_episode_duration_s(fps, episode_lengths)
    visual_features = _feature_keys(features, dtype="video", exclude="depth")
    depth_features = _feature_keys(features, contains="depth")
    has_action = "action" in features or any(key.startswith("action.") for key in features)
    has_state = "observation.state" in features or any("state" in key for key in features)
    has_gripper = _has_gripper_feature(features)
    video_resolution = _infer_video_resolution(features, visual_features)
    task_descriptions_present = _has_task_description(dataset_path, episode_meta)

    thresholds = dict(QUALITY_THRESHOLD_DEFAULTS)
    thresholds.update({
        "metadata_require_info_json": 1.0,
        "metadata_require_episode_metadata": 1.0,
        "metadata_require_data_files": 1.0,
        "metadata_require_videos": 1.0 if visual_features else 0.0,
        "metadata_require_task_description": 1.0,
        "metadata_min_duration_s": _duration_default(median_duration),
        "visual_min_video_count": 1.0 if visual_features else 0.0,
        "depth_min_stream_count": 1.0 if depth_features else 0.0,
    })

    if fps > 0:
        thresholds["timing_min_frequency_hz"] = round(max(1.0, min(20.0, fps * 0.66)), 2)
        thresholds["visual_min_frame_rate"] = round(max(1.0, fps * 0.8), 2)
        thresholds["visual_frame_rate_tolerance"] = round(max(1.0, fps * 0.1), 2)

    if median_duration > 0:
        thresholds["action_min_duration_s"] = thresholds["metadata_min_duration_s"]
        thresholds["action_max_all_static_s"] = round(max(0.5, min(3.0, median_duration * 0.35)), 2)
        thresholds["action_max_key_static_s"] = round(max(0.75, min(5.0, median_duration * 0.5)), 2)

    if video_resolution:
        thresholds["visual_min_resolution_width"] = float(video_resolution["width"])
        thresholds["visual_min_resolution_height"] = float(video_resolution["height"])

    selected_validators = ["metadata"]
    if fps > 0:
        selected_validators.append("timing")
    if has_action:
        selected_validators.append("action")
    if visual_features:
        selected_validators.append("visual")
    if depth_features:
        selected_validators.append("depth")
    if has_action and has_state and has_gripper:
        selected_validators.append("ee_trajectory")

    return {
        "dataset": dataset_name or dataset_path.name,
        "selected_validators": [
            validator for validator in VALIDATOR_ORDER if validator in selected_validators
        ],
        "threshold_overrides": thresholds,
        "profile": {
            "fps": fps,
            "median_episode_duration_s": round(median_duration, 3),
            "video_resolution": video_resolution,
            "visual_streams": visual_features,
            "depth_streams": depth_features,
            "has_action": has_action,
            "has_state": has_state,
            "has_gripper": has_gripper,
        },
        "checks": {
            "metadata_present": info_path.is_file(),
            "episode_metadata_present": _episode_metadata_present(dataset_path),
            "data_files_present": any((dataset_path / "data").rglob("*.parquet")),
            "video_files_present": any((dataset_path / "videos").rglob("*.mp4")),
            "task_descriptions_present": task_descriptions_present,
        },
    }


def _feature_keys(
    features: dict[str, Any],
    *,
    dtype: str | None = None,
    contains: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    keys: list[str] = []
    for key, config in features.items():
        lowered = key.lower()
        if contains and contains not in lowered:
            continue
        if exclude and exclude in lowered:
            continue
        if dtype is not None:
            if not isinstance(config, dict) or str(config.get("dtype", "")).lower() != dtype:
                continue
        keys.append(str(key))
    return keys


def _has_gripper_feature(features: dict[str, Any]) -> bool:
    for key, config in features.items():
        if "gripper" in key.lower():
            return True
        if not isinstance(config, dict):
            continue
        names = config.get("names")
        if _names_contain(names, "gripper"):
            return True
    return False


def _names_contain(names: Any, needle: str) -> bool:
    if isinstance(names, str):
        return needle in names.lower()
    if isinstance(names, dict):
        return any(_names_contain(value, needle) for value in names.values())
    if isinstance(names, (list, tuple)):
        return any(_names_contain(value, needle) for value in names)
    return False


def _infer_video_resolution(
    features: dict[str, Any],
    visual_features: list[str],
) -> dict[str, int] | None:
    for key in visual_features:
        config = features.get(key)
        if not isinstance(config, dict):
            continue
        shape = config.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            continue
        dims = [int(value) for value in shape if isinstance(value, (int, float))]
        if len(dims) < 2:
            continue
        if len(dims) >= 3 and dims[0] in {1, 3, 4}:
            height, width = dims[1], dims[2]
        else:
            height, width = dims[0], dims[1]
        if width > 0 and height > 0:
            return {"width": width, "height": height}
    return None


def _first_episode_meta(dataset_path: Path) -> dict[str, Any]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    episodes_dir = dataset_path / "meta" / "episodes"
    for path in sorted(episodes_dir.rglob("*.parquet"))[:1]:
        rows = read_parquet_rows(path)
        if rows:
            return rows[0]
    return {}


def _episode_metadata_present(dataset_path: Path) -> bool:
    return (
        (dataset_path / "meta" / "episodes.jsonl").is_file()
        or any((dataset_path / "meta" / "episodes").rglob("*.parquet"))
    )


def _episode_lengths(
    info: dict[str, Any],
    first_episode_meta: dict[str, Any],
    dataset_path: Path,
) -> list[float]:
    raw_lengths = info.get("episode_lengths")
    if isinstance(raw_lengths, list) and raw_lengths:
        return [value for value in (_coerce_positive_float(item) for item in raw_lengths) if value]

    lengths: list[float] = []
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for index, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines()):
            if index >= 200:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                value = _coerce_positive_float(payload.get("length"))
                if value:
                    lengths.append(value)
    if lengths:
        return lengths

    value = _coerce_positive_float(first_episode_meta.get("length"))
    return [value] if value else []


def _coerce_positive_float(value: Any) -> float | None:
    numeric = safe_float(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _median_episode_duration_s(fps: float, episode_lengths: list[float]) -> float:
    if not episode_lengths:
        return 0.0
    median = float(statistics.median(episode_lengths))
    if fps > 0 and median > fps:
        return median / fps
    return median


def _duration_default(median_duration_s: float) -> float:
    if median_duration_s <= 0:
        return QUALITY_THRESHOLD_DEFAULTS["metadata_min_duration_s"]
    return round(max(0.1, min(1.0, median_duration_s * 0.05)), 2)


def _has_task_description(dataset_path: Path, episode_meta: dict[str, Any]) -> bool:
    if payload_has_task_description(episode_meta):
        return True
    task_index = episode_meta.get("task_index")
    if task_index is not None and _task_lookup_has_description(dataset_path, task_index):
        return True
    return _tasks_file_has_any_description(dataset_path)


def _task_lookup_has_description(dataset_path: Path, task_index: Any) -> bool:
    expected_index = safe_float(task_index)
    if expected_index is None:
        return False
    for row in _task_rows(dataset_path):
        row_index = safe_float(row.get("task_index"))
        if row_index is None or int(row_index) != int(expected_index):
            continue
        if _row_has_task_text(row):
            return True
    return False


def _tasks_file_has_any_description(dataset_path: Path) -> bool:
    return any(_row_has_task_text(row) for row in _task_rows(dataset_path))


def _task_rows(dataset_path: Path) -> list[dict[str, Any]]:
    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"
    if tasks_jsonl.is_file():
        rows: list[dict[str, Any]] = []
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    if tasks_parquet.is_file():
        return read_parquet_rows(tasks_parquet)
    return []


def _row_has_task_text(row: dict[str, Any]) -> bool:
    return payload_has_task_description(row)
