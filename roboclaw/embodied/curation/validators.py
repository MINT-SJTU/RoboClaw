from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download
from loguru import logger
import numpy as np

from .bridge import read_parquet_rows
from .features import percentile
from .propagation import (
    _extract_gripper_series,
    _find_gripper_index,
    detect_grasp_place_events,
)
from .state import load_dataset_info

QUALITY_THRESHOLD_DEFAULTS: dict[str, float] = {
    "metadata_min_duration_s": 1.0,
    "timing_min_monotonicity": 0.99,
    "timing_max_interval_cv": 0.05,
    "timing_min_frequency_hz": 20.0,
    "timing_max_gap_ratio": 0.01,
    "timing_min_frequency_consistency": 0.98,
    "action_static_threshold": 0.001,
    "action_max_all_static_s": 3.0,
    "action_max_key_static_s": 5.0,
    "action_max_velocity_rad_s": 3.14,
    "action_min_duration_s": 1.0,
    "action_max_nan_ratio": 0.01,
    "visual_min_resolution_width": 640.0,
    "visual_min_resolution_height": 480.0,
    "visual_min_frame_rate": 20.0,
    "visual_frame_rate_tolerance": 2.0,
    "visual_color_shift_max": 0.10,
    "visual_overexposure_ratio_max": 0.05,
    "visual_underexposure_ratio_max": 0.10,
    "visual_abnormal_black_ratio_max": 0.95,
    "visual_abnormal_white_ratio_max": 0.95,
    "visual_min_video_count": 1.0,
    "visual_min_accessible_ratio": 1.0,
    "depth_min_stream_count": 0.0,
    "depth_min_accessible_ratio": 1.0,
    "depth_invalid_pixel_max": 0.10,
    "depth_continuity_min": 0.90,
    "ee_min_event_count": 1.0,
    "ee_min_gripper_span": 0.05,
}

# ---------------------------------------------------------------------------
# Issue / score model
# ---------------------------------------------------------------------------


def make_issue(
    *,
    operator_name: str,
    check_name: str,
    passed: bool,
    message: str,
    level: str = "major",
    value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "operator_name": operator_name,
        "check_name": check_name,
        "passed": passed,
        "message": message,
        "level": level,
        "value": value or {},
    }


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def finalize_validator(
    operator_name: str,
    issues: list[dict[str, Any]],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = max(len(issues), 1)
    passed_count = sum(1 for issue in issues if issue["passed"])
    score = round((passed_count / total) * 100, 1)
    blocking_levels = {"critical", "major"}
    passed = all(issue["passed"] for issue in issues if issue["level"] in blocking_levels)
    return {
        "name": operator_name,
        "passed": passed,
        "score": score,
        "issues": issues,
        "details": details or {},
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_threshold_overrides(threshold_overrides: dict[str, float] | None = None) -> dict[str, float]:
    merged = dict(QUALITY_THRESHOLD_DEFAULTS)
    if not threshold_overrides:
        return merged
    for key, value in threshold_overrides.items():
        if key not in merged:
            continue
        numeric = safe_float(value)
        if numeric is None:
            continue
        merged[key] = numeric
    return merged


# ---------------------------------------------------------------------------
# LeRobot directory helpers
# ---------------------------------------------------------------------------


def load_episode_data(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    """Load episode parquet data, metadata, and video paths from LeRobot directory."""
    info = _load_info_json(dataset_path)
    episodes_meta = _load_episode_meta(dataset_path, episode_index)
    chunk = _resolve_chunk(info, episode_index)
    parquet_relative_path = Path("data") / f"chunk-{chunk}" / f"episode_{episode_index:06d}.parquet"
    parquet_path = dataset_path / parquet_relative_path
    if parquet_path.exists():
        rows = _read_parquet_rows(parquet_path)
    else:
        remote_dataset_id = _resolve_remote_dataset_id(dataset_path, info)
        parquet_path = _download_remote_file(
            remote_dataset_id,
            parquet_relative_path,
            local_root=dataset_path,
        )
        rows = _read_parquet_rows(parquet_path)

    video_dir = dataset_path / "videos" / f"chunk-{chunk}" / f"episode_{episode_index:06d}"
    if video_dir.exists():
        video_files = _list_video_files(video_dir)
    else:
        remote_dataset_id = _resolve_remote_dataset_id(dataset_path, info)
        video_files = _download_remote_videos(
            remote_dataset_id,
            info,
            episode_index,
            local_root=dataset_path,
        )

    return {
        "info": info,
        "episode_meta": episodes_meta,
        "rows": rows,
        "parquet_path": parquet_path,
        "video_dir": video_dir,
        "video_files": video_files,
        "chunk": chunk,
    }


def _resolve_remote_dataset_id(dataset_path: Path, info: dict[str, Any]) -> str:
    source_dataset = info.get("source_dataset") or info.get("repo_id") or info.get("dataset_id")
    if isinstance(source_dataset, str) and source_dataset.strip():
        return source_dataset.strip()
    try:
        from roboclaw.embodied.curation.paths import datasets_root
        root = datasets_root().resolve()
        resolved = dataset_path.resolve()
        if str(resolved).startswith(str(root) + "/"):
            return resolved.relative_to(root).as_posix()
    except Exception:
        logger.debug("Failed to resolve remote dataset id", exc_info=True)
    return dataset_path.name


_load_info_json = load_dataset_info


def _load_episode_meta(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("episode_index") == episode_index:
            return entry
    return {}


def _resolve_chunk(info: dict[str, Any], episode_index: int) -> str:
    chunks_size = info.get("chunks_size", 1000)
    if chunks_size <= 0:
        chunks_size = 1000
    return f"{episode_index // chunks_size:03d}"


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_parquet_rows(path)


def _download_remote_file(
    dataset_id: str,
    relative_path: Path,
    *,
    local_root: Path | None = None,
) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": dataset_id,
        "filename": relative_path.as_posix(),
        "repo_type": "dataset",
    }
    if local_root is not None:
        kwargs["local_dir"] = str(local_root)
    cached_path = hf_hub_download(**kwargs)
    return Path(cached_path)


def _extract_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    keys: list[str] = []
    for name, config in features.items():
        if not isinstance(config, dict):
            continue
        if config.get("dtype") == "video":
            keys.append(str(name))
    return keys


def _download_remote_videos(
    dataset_id: str,
    info: dict[str, Any],
    episode_index: int,
    *,
    local_root: Path | None = None,
) -> list[Path]:
    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        return []

    chunk = _resolve_chunk(info, episode_index)
    chunk_index = int(chunk)
    video_keys = _extract_video_keys(info)
    results: list[Path] = []
    for video_key in video_keys:
        try:
            relative_path = template.format(
                episode_chunk=chunk_index,
                video_key=video_key,
                episode_index=episode_index,
            )
            results.append(
                _download_remote_file(
                    dataset_id,
                    Path(relative_path),
                    local_root=local_root,
                ),
            )
        except Exception:
            logger.warning("Failed to download video %s", video_key, exc_info=True)
            continue
    return results


def _list_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(video_dir.glob("*.mp4"))


from .visual_validators import validate_depth_assets, validate_visual_assets


# ---------------------------------------------------------------------------
# Metadata validator
# ---------------------------------------------------------------------------


def validate_metadata(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "metadata"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    info = data["info"]
    episode_meta = data["episode_meta"]
    issues: list[dict[str, Any]] = []

    if not info:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="info.json",
            passed=False,
            message="Missing meta/info.json",
            level="critical",
        ))
        return finalize_validator(operator_name, issues)

    episode_index = data.get("episode_meta", {}).get("episode_index")
    _check_episode_identity(issues, operator_name, episode_meta, episode_index)
    _check_info_fields(issues, operator_name, info)
    _check_data_files(issues, operator_name, data)
    _check_duration(issues, operator_name, episode_meta, thresholds)

    return finalize_validator(operator_name, issues, details={"info": info, "episode_meta": episode_meta})


def _check_episode_identity(
    issues: list[dict[str, Any]],
    operator_name: str,
    episode_meta: dict[str, Any],
    episode_index: int,
) -> None:
    has_identity = episode_meta.get("episode_index") is not None
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="episode identity",
        passed=has_identity,
        message=f"episode_index={'present' if has_identity else 'missing'} in episodes.jsonl",
        level="major" if not has_identity else "minor",
    ))


def _check_info_fields(
    issues: list[dict[str, Any]],
    operator_name: str,
    info: dict[str, Any],
) -> None:
    required = [("robot_type", "major"), ("fps", "major")]
    recommended = [("features", "minor")]
    for field, level in required + recommended:
        value = info.get(field)
        present = is_present(value)
        issues.append(make_issue(
            operator_name=operator_name,
            check_name=field,
            passed=present,
            message=f"{field}={'present' if present else 'missing'}",
            level=level if not present else "minor",
            value={"field": field, "value": value},
        ))


def _check_data_files(
    issues: list[dict[str, Any]],
    operator_name: str,
    data: dict[str, Any],
) -> None:
    parquet_exists = data["parquet_path"].exists()
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="parquet_data",
        passed=parquet_exists,
        message=f"parquet data={'exists' if parquet_exists else 'missing'}",
        level="major",
    ))
    has_videos = bool(data["video_files"])
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="videos",
        passed=has_videos,
        message=f"video files={'found' if has_videos else 'missing'}",
        level="minor",
    ))


def _check_duration(
    issues: list[dict[str, Any]],
    operator_name: str,
    episode_meta: dict[str, Any],
    thresholds: dict[str, float],
) -> None:
    duration = safe_float(episode_meta.get("length")) or 0.0
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="length",
        passed=duration >= thresholds["metadata_min_duration_s"],
        message=f"Episode length {duration}",
        level="major",
        value={"length": duration},
    ))


# ---------------------------------------------------------------------------
# Timing validator
# ---------------------------------------------------------------------------


def validate_timing(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "timing"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    rows = data["rows"]
    issues: list[dict[str, Any]] = []
    timestamps = _extract_timestamps(rows)

    if len(timestamps) < 2:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="timestamps",
            passed=False,
            message="Insufficient timestamps for timing validation",
            level="critical",
        ))
        return finalize_validator(operator_name, issues)

    diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    positive_diffs = [d for d in diffs if d > 0]
    _check_monotonicity(issues, operator_name, diffs, thresholds)

    if positive_diffs:
        _check_timing_details(issues, operator_name, positive_diffs, thresholds)

    return finalize_validator(operator_name, issues, details={"frame_count": len(timestamps)})


def _extract_timestamps(rows: list[dict[str, Any]]) -> list[float]:
    raw = [
        safe_float(
            row["timestamp"] if "timestamp" in row else row.get("timestamp_utc"),
        )
        for row in rows
    ]
    return [v for v in raw if v is not None]


def _check_monotonicity(
    issues: list[dict[str, Any]],
    operator_name: str,
    diffs: list[float],
    thresholds: dict[str, float],
) -> None:
    non_monotonic = sum(1 for d in diffs if d <= 0)
    ratio = 1.0 - (non_monotonic / len(diffs))
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="monotonicity",
        passed=ratio >= thresholds["timing_min_monotonicity"],
        message=f"Timestamp monotonicity {ratio * 100:.2f}%",
        level="major",
        value={"monotonic_ratio": ratio},
    ))


def _check_timing_details(
    issues: list[dict[str, Any]],
    operator_name: str,
    positive_diffs: list[float],
    thresholds: dict[str, float],
) -> None:
    median_interval = statistics.median(positive_diffs)
    mean_interval = statistics.fmean(positive_diffs)
    std_interval = statistics.pstdev(positive_diffs) if len(positive_diffs) > 1 else 0.0
    interval_cv = (std_interval / mean_interval) if mean_interval > 0 else 0.0
    estimated_freq = (1.0 / median_interval) if median_interval > 0 else 0.0
    gap_ratio = sum(1 for d in positive_diffs if d > 1.0) / len(positive_diffs)
    consistency = _trimmed_consistency(positive_diffs)

    issues.extend([
        make_issue(
            operator_name=operator_name,
            check_name="interval_cv",
            passed=interval_cv < thresholds["timing_max_interval_cv"],
            message=f"Sampling interval CV {interval_cv * 100:.2f}%",
            level="major",
            value={"interval_cv": interval_cv},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="estimated_frequency",
            passed=estimated_freq >= thresholds["timing_min_frequency_hz"],
            message=f"Estimated frequency {estimated_freq:.2f} Hz",
            level="major",
            value={"estimated_frequency_hz": estimated_freq},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="gap_ratio",
            passed=gap_ratio < thresholds["timing_max_gap_ratio"],
            message=f"Gaps >1s ratio {gap_ratio * 100:.2f}%",
            level="major",
            value={"gap_ratio": gap_ratio},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="frequency_consistency",
            passed=consistency >= thresholds["timing_min_frequency_consistency"],
            message=f"Frequency consistency {consistency * 100:.2f}%",
            level="major",
            value={"consistency": consistency},
        ),
    ])


def _trimmed_consistency(positive_diffs: list[float]) -> float:
    trimmed = positive_diffs[:]
    if len(trimmed) > 10:
        trim = max(int(len(trimmed) * 0.1), 1)
        trimmed = sorted(trimmed)[trim:-trim] or trimmed
    trimmed_mean = statistics.fmean(trimmed)
    trimmed_std = statistics.pstdev(trimmed) if len(trimmed) > 1 else 0.0
    return 1.0 - ((trimmed_std / trimmed_mean) if trimmed_mean > 0 else 0.0)


# ---------------------------------------------------------------------------
# Action validator
# ---------------------------------------------------------------------------


def validate_action(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "action"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    rows = data["rows"]
    issues: list[dict[str, Any]] = []
    timestamps = _extract_timestamps(rows)

    if len(timestamps) < 2:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="timestamps",
            passed=False,
            message="Insufficient timestamps for action validation",
            level="critical",
        ))
        return finalize_validator(operator_name, issues)

    primary_series = _collect_primary_series(rows, data.get("info"))
    if not primary_series:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="joint_series",
            passed=False,
            message="No joint series found for validation",
            level="critical",
        ))
        return finalize_validator(operator_name, issues)

    _check_static_duration(issues, operator_name, primary_series, timestamps, thresholds)
    _check_velocity_and_quality(issues, operator_name, primary_series, timestamps, thresholds)

    return finalize_validator(
        operator_name, issues,
        details={"joint_count": len(primary_series), "frame_count": len(timestamps)},
    )


def _action_candidate_columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({
        key for row in rows for key in row.keys()
        if key.startswith("state_")
        or key.startswith("action_")
        or key == "action"
        or key.startswith("action.")
        or key == "observation.state"
        or key.startswith("observation.state.")
    })


def _extract_numeric_components(value: Any) -> list[float | None]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [safe_float(value.item())]
        if value.ndim == 1:
            return [safe_float(item) for item in value.tolist()]
        return []
    if isinstance(value, (list, tuple)):
        return [safe_float(item) for item in value]
    return [safe_float(value)]


def _feature_axis_names(
    info: dict[str, Any] | None,
    key: str,
    component_count: int,
) -> list[str]:
    if not isinstance(info, dict):
        return [str(index) for index in range(component_count)]

    feature = info.get("features", {}).get(key, {})
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, dict):
        axes = names.get("axes")
        if isinstance(axes, list) and len(axes) == component_count:
            return [str(axis) for axis in axes]
    if isinstance(names, list) and len(names) == component_count:
        return [str(name) for name in names]
    return [str(index) for index in range(component_count)]


def _collect_primary_series(
    rows: list[dict[str, Any]],
    info: dict[str, Any] | None = None,
) -> dict[str, list[float | None]]:
    series: dict[str, list[float | None]] = {}
    for name in _action_candidate_columns(rows):
        row_components = [_extract_numeric_components(row.get(name)) for row in rows]
        component_count = max((len(components) for components in row_components), default=0)
        if component_count == 0:
            continue

        if component_count == 1:
            series[name] = [
                components[0] if components else None
                for components in row_components
            ]
            continue

        component_names = _feature_axis_names(info, name, component_count)
        for index in range(component_count):
            component_label = component_names[index] if index < len(component_names) else str(index)
            series[f"{name}.{component_label}"] = [
                components[index] if index < len(components) else None
                for components in row_components
            ]

    populated = {
        key: values for key, values in series.items()
        if any(value is not None for value in values)
    }
    non_gripper = {
        key: values for key, values in populated.items()
        if "gripper" not in key.lower()
    }
    return non_gripper or populated or series


def _longest_static_duration(
    series: dict[str, list[float | None]],
    timestamps: list[float],
    threshold: float,
) -> float:
    if not series or len(timestamps) < 2:
        return 0.0
    keys = list(series.keys())
    longest = 0.0
    current = 0.0
    for index in range(1, len(timestamps)):
        max_diff = 0.0
        valid = False
        for key in keys:
            cv = series[key][index]
            pv = series[key][index - 1]
            if cv is None or pv is None:
                continue
            valid = True
            max_diff = max(max_diff, abs(cv - pv))
        if valid and max_diff < threshold:
            current += max(timestamps[index] - timestamps[index - 1], 0.0)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _check_static_duration(
    issues: list[dict[str, Any]],
    operator_name: str,
    primary_series: dict[str, list[float | None]],
    timestamps: list[float],
    thresholds: dict[str, float],
) -> None:
    static_threshold = thresholds["action_static_threshold"]
    all_static = _longest_static_duration(primary_series, timestamps, static_threshold)
    key_subset = dict(list(primary_series.items())[:min(6, len(primary_series))])
    key_static = _longest_static_duration(key_subset, timestamps, static_threshold)
    issues.extend([
        make_issue(
            operator_name=operator_name,
            check_name="all_static_duration",
            passed=all_static <= thresholds["action_max_all_static_s"],
            message=f"All-joint longest static {all_static:.2f}s",
            level="major",
            value={"all_static_duration_s": all_static},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="key_static_duration",
            passed=key_static <= thresholds["action_max_key_static_s"],
            message=f"Key-joint longest static {key_static:.2f}s",
            level="major",
            value={"key_static_duration_s": key_static},
        ),
    ])


def _check_velocity_and_quality(
    issues: list[dict[str, Any]],
    operator_name: str,
    primary_series: dict[str, list[float | None]],
    timestamps: list[float],
    thresholds: dict[str, float],
) -> None:
    velocities: list[float] = []
    nan_like_count = 0
    total_value_count = 0
    absolute_values = [
        abs(v)
        for vals in primary_series.values()
        for v in vals
        if v is not None
    ]
    for values in primary_series.values():
        limit = min(len(values), len(timestamps))
        for index in range(1, limit):
            cv = values[index]
            pv = values[index - 1]
            if cv is None or pv is None:
                nan_like_count += 1
                total_value_count += 1
                continue
            dt = max(timestamps[index] - timestamps[index - 1], 1e-6)
            velocities.append(abs(cv - pv) / dt)
            total_value_count += 1

    unit_scale = (math.pi / 180.0) if percentile(absolute_values, 0.95) > 10.0 else 1.0
    scaled = [v * unit_scale for v in velocities]
    max_velocity = percentile(scaled, 0.99) if scaled else 0.0
    nan_ratio = (nan_like_count / total_value_count) if total_value_count else 0.0
    duration = timestamps[-1] - timestamps[0]

    issues.extend([
        make_issue(
            operator_name=operator_name,
            check_name="max_velocity",
            passed=max_velocity < thresholds["action_max_velocity_rad_s"],
            message=f"P99 velocity {max_velocity:.3f} rad/s",
            level="major",
            value={"max_velocity": max_velocity, "unit_scale": unit_scale},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="duration",
            passed=duration >= thresholds["action_min_duration_s"],
            message=f"Action duration {duration:.2f}s",
            level="major",
            value={"duration_s": duration},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="nan_ratio",
            passed=nan_ratio < thresholds["action_max_nan_ratio"],
            message=f"Missing value ratio {nan_ratio * 100:.2f}%",
            level="major",
            value={"nan_ratio": nan_ratio},
        ),
    ])




def validate_ee_trajectory(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "ee_trajectory"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    rows = data["rows"]
    info = data["info"]
    timestamps = _extract_timestamps(rows)
    duration_s = max(timestamps[-1] - timestamps[0], 0.0) if len(timestamps) > 1 else 0.0
    action_names = info.get("features", {}).get("action", {}).get("names", []) if isinstance(info.get("features", {}).get("action"), dict) else []
    state_names = info.get("features", {}).get("observation.state", {}).get("names", []) if isinstance(info.get("features", {}).get("observation.state"), dict) else []
    issues: list[dict[str, Any]] = []

    spans = detect_grasp_place_events(
        rows=rows,
        action_names=[str(name) for name in action_names] if isinstance(action_names, list) else [],
        state_names=[str(name) for name in state_names] if isinstance(state_names, list) else [],
        duration_s=duration_s,
    )
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="grasp_event_count",
        passed=len(spans) >= int(thresholds["ee_min_event_count"]),
        message=f"Detected grasp/place events {len(spans)}",
        level="minor",
        value={"event_count": len(spans)},
    ))

    gripper_series, _series_timestamps = _extract_gripper_series(rows, _find_gripper_index(
        [str(name) for name in action_names] if isinstance(action_names, list) else [],
        [str(name) for name in state_names] if isinstance(state_names, list) else [],
    ))
    if gripper_series:
        span = max(gripper_series) - min(gripper_series)
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="gripper_motion_span",
            passed=span >= thresholds["ee_min_gripper_span"],
            message=f"Gripper span {span:.3f}",
            level="minor",
            value={"gripper_span": span},
        ))

    return finalize_validator(operator_name, issues, details={"event_count": len(spans)})


# ---------------------------------------------------------------------------
# Registry and runner
# ---------------------------------------------------------------------------

VALIDATOR_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "metadata": validate_metadata,
    "timing": validate_timing,
    "action": validate_action,
    "visual": validate_visual_assets,
    "depth": validate_depth_assets,
    "ee_trajectory": validate_ee_trajectory,
}


def run_quality_validators(
    dataset_path: Path,
    episode_index: int,
    *,
    selected_validators: list[str] | None = None,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run selected validators. If None, runs all."""
    names = selected_validators or list(VALIDATOR_REGISTRY.keys())
    unknown = [n for n in names if n not in VALIDATOR_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown validators: {unknown}")
    data = load_episode_data(dataset_path, episode_index)
    results: list[dict[str, Any]] = []
    for name in names:
        results.append(VALIDATOR_REGISTRY[name](data, threshold_overrides))

    all_issues = [issue for result in results for issue in result["issues"]]
    total_score = statistics.fmean([r["score"] for r in results]) if results else 0.0
    passed = all(r["passed"] for r in results)
    validators_dict = {
        r["name"]: {"passed": r["passed"], "score": r["score"]}
        for r in results
    }
    return {
        "passed": passed,
        "score": round(total_score, 1),
        "validators": validators_dict,
        "issues": all_issues,
    }
