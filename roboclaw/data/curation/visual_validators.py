from __future__ import annotations

import io
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def _merge_threshold_overrides(threshold_overrides: dict[str, float] | None = None) -> dict[str, float]:
    from .validators import _merge_threshold_overrides as merge

    return merge(threshold_overrides)


def finalize_validator(operator_name: str, issues: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    from .validators import finalize_validator as finalize

    return finalize(operator_name, issues, **kwargs)


def make_issue(**kwargs: Any) -> dict[str, Any]:
    from .validators import make_issue as build_issue

    return build_issue(**kwargs)


def safe_float(value: Any) -> float | None:
    from .validators import safe_float as coerce_float

    return coerce_float(value)


def _sample_video_frames(
    video_path: Path,
    max_samples: int = 10,
    *,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> tuple[list[np.ndarray], float, int, int, int]:
    cv2_result = _sample_video_frames_with_cv2(video_path, max_samples, clip_start_s, clip_end_s)
    if cv2_result is not None:
        return cv2_result

    try:
        import av
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        frame_count = int(stream.frames or 0)
        sample_indexes = set(_sample_frame_indexes(max_samples, fps, frame_count, clip_start_s, clip_end_s))
        start_frame, end_frame = _clip_frame_bounds(fps, frame_count, clip_start_s, clip_end_s)
        if start_frame > 0 and stream.time_base:
            try:
                container.seek(int((start_frame / max(fps, 1.0)) / stream.time_base), stream=stream)
            except Exception:
                container.seek(int((start_frame / max(fps, 1.0)) * 1_000_000))
        frames: list[np.ndarray] = []
        for index, frame in enumerate(container.decode(stream)):
            frame_time = frame.time
            if frame_time is not None and fps > 0:
                frame_index = int(round(frame_time * fps))
            else:
                frame_index = index
            if frame_index < start_frame:
                continue
            if frame_count > 0 and frame_index >= end_frame:
                break
            if frame_index not in sample_indexes:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= len(sample_indexes):
                break
        container.close()
        return frames, fps, width, height, frame_count
    except Exception:
        pass

    return [], 0.0, 0, 0, 0


def _sample_video_frames_with_cv2(
    video_path: Path,
    max_samples: int,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> tuple[list[np.ndarray], float, int, int, int] | None:
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames: list[np.ndarray] = []
        for index in _sample_frame_indexes(max_samples, fps, frame_count, clip_start_s, clip_end_s):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames, fps, width, height, frame_count


def _video_sample_count(
    max_samples: int,
    fps: float,
    frame_count: int,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
) -> int:
    limit = max(int(max_samples), 1)
    if frame_count <= 0:
        return limit
    start_frame, end_frame = _clip_frame_bounds(fps, frame_count, clip_start_s, clip_end_s)
    return max(min(limit, end_frame - start_frame), 0)


def _clip_frame_bounds(
    fps: float,
    frame_count: int,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0
    start_frame = 0
    end_frame = frame_count
    if fps > 0 and clip_start_s is not None:
        start_frame = int(np.floor(max(clip_start_s, 0.0) * fps))
    if fps > 0 and clip_end_s is not None:
        end_frame = int(np.ceil(max(clip_end_s, 0.0) * fps))
    start_frame = min(max(start_frame, 0), frame_count - 1)
    end_frame = min(max(end_frame, start_frame + 1), frame_count)
    return start_frame, end_frame


def _sample_frame_indexes(
    max_samples: int,
    fps: float,
    frame_count: int,
    clip_start_s: float | None,
    clip_end_s: float | None,
) -> list[int]:
    sample_count = _video_sample_count(max_samples, fps, frame_count, clip_start_s, clip_end_s)
    if sample_count <= 0:
        return []
    start_frame, end_frame = _clip_frame_bounds(fps, frame_count, clip_start_s, clip_end_s)
    available = end_frame - start_frame
    if available <= sample_count:
        return list(range(start_frame, end_frame))
    return [int(value) for value in np.linspace(start_frame, end_frame - 1, sample_count)]


def _video_metadata_from_feature(config: Any) -> tuple[float, int, int]:
    if not isinstance(config, dict):
        return 0.0, 0, 0

    info = config.get("info")
    info = info if isinstance(info, dict) else {}
    fps = safe_float(info.get("video.fps")) or 0.0
    width = int(safe_float(info.get("video.width")) or 0)
    height = int(safe_float(info.get("video.height")) or 0)

    shape = config.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        height = height or int(safe_float(shape[0]) or 0)
        width = width or int(safe_float(shape[1]) or 0)
    return fps, width, height


def _visual_feature_lookup(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    return {
        key: value for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video" and "depth" not in key.lower()
    }


def _feature_key_for_video_path(video_path: Path, feature_keys: list[str]) -> str | None:
    text = video_path.as_posix()
    for key in sorted(feature_keys, key=len, reverse=True):
        short_name = key.rsplit(".", 1)[-1]
        if key in text or short_name in video_path.stem:
            return key
    if len(feature_keys) == 1:
        return feature_keys[0]
    return None


def _stream_label(video_path: Path, feature_key: str | None) -> str:
    if feature_key:
        return feature_key.rsplit(".", 1)[-1]
    return video_path.stem


def _video_clip_bounds(episode_meta: dict[str, Any], video_key: str | None) -> tuple[float | None, float | None]:
    if not video_key:
        return _fallback_video_clip_bounds(episode_meta)
    prefix = f"videos/{video_key}/"
    clip_start = safe_float(episode_meta.get(f"{prefix}from_timestamp"))
    clip_end = safe_float(episode_meta.get(f"{prefix}to_timestamp"))
    if clip_start is None and clip_end is None:
        return _fallback_video_clip_bounds(episode_meta)
    return clip_start, clip_end


def _fallback_video_clip_bounds(episode_meta: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        safe_float(episode_meta.get("video_from_timestamp")),
        safe_float(episode_meta.get("video_to_timestamp")),
    )


def _decode_image_like(value: Any) -> np.ndarray | None:
    from PIL import Image
    if value is None:
        return None
    if isinstance(value, dict):
        if "bytes" in value and isinstance(value["bytes"], (bytes, bytearray)):
            try:
                return np.array(Image.open(io.BytesIO(value["bytes"])))
            except Exception:
                return None
        if "path" in value:
            return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return np.array(Image.open(io.BytesIO(value)))
        except Exception:
            return None
    if hasattr(value, "shape"):
        try:
            return np.array(value)
        except Exception:
            return None
    return None


def _iter_visual_parquet_frames(rows: list[dict[str, Any]], max_samples: int = 10) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    if not rows:
        return frames
    sample_count = _row_sample_count(rows, max_samples)
    sample_step = max(1, len(rows) // sample_count)
    for row in rows[::sample_step]:
        for key, value in row.items():
            if "observation.images" not in key or "depth" in key.lower():
                continue
            decoded = _decode_image_like(value)
            if decoded is None:
                continue
            frames.append((key, decoded))
            if len(frames) >= sample_count:
                return frames
    return frames


def _iter_depth_parquet_frames(rows: list[dict[str, Any]], max_samples: int = 10) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    if not rows:
        return frames
    sample_count = _row_sample_count(rows, max_samples)
    sample_step = max(1, len(rows) // sample_count)
    for row in rows[::sample_step]:
        for key, value in row.items():
            if "depth" not in key.lower():
                continue
            decoded = _decode_image_like(value)
            if decoded is None:
                continue
            frames.append((key, decoded))
            if len(frames) >= sample_count:
                return frames
    return frames


def _row_sample_count(rows: list[dict[str, Any]], min_samples: int) -> int:
    return min(max(min_samples, 1), len(rows))


def _compute_visual_frame_stats(frame: np.ndarray) -> dict[str, float]:
    gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame.astype(np.float32)
    result = {
        "overexposure": float(np.mean(gray > 250)),
        "underexposure": float(np.mean(gray < 5)),
        "black": float(np.mean(gray < 2)),
        "white": float(np.mean(gray > 253)),
        "color_shift": 0.0,
    }
    if frame.ndim == 3:
        means = np.mean(frame, axis=(0, 1))
        result["color_shift"] = float(np.std(means) / 255.0)
    return result


def _compute_depth_invalid_ratio(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 1.0
    if frame.dtype == np.uint8:
        return float(np.mean(frame == 0))
    return float(np.mean((frame == 0) | np.isnan(frame)))


def _compute_depth_continuity(current: np.ndarray, previous: np.ndarray | None) -> float | None:
    if previous is None or previous.shape != current.shape:
        return None
    if current.dtype == np.uint8:
        valid_current = current > 0
        valid_previous = previous > 0
    else:
        valid_current = (current > 0) & (~np.isnan(current))
        valid_previous = (previous > 0) & (~np.isnan(previous))
    union = np.sum(valid_current | valid_previous)
    if union <= 0:
        return None
    overlap = np.sum(valid_current & valid_previous)
    return float(overlap / union)


def validate_visual_assets(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "visual"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    video_files = data["video_files"]
    rows = data["rows"]
    info = data["info"]
    episode_meta = data.get("episode_meta", {}) or {}
    issues: list[dict[str, Any]] = []
    min_video_count = int(thresholds["visual_min_video_count"])
    visual_features = _visual_feature_lookup(info)
    visual_feature_keys = list(visual_features)

    if not visual_feature_keys:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="visual_streams",
            passed=True,
            message="No visual streams declared in dataset metadata, skipping visual validation",
            level="info",
            value={"visual_streams": 0},
        ))
        return finalize_validator(operator_name, issues, details={"visual_streams": 0, "skipped": True})

    issues.append(make_issue(
        operator_name=operator_name,
        check_name="video_count",
        passed=len(video_files) >= min_video_count,
        message=f"Video file count {len(video_files)}",
        level="major" if len(video_files) < min_video_count else "minor",
        value={"video_count": len(video_files)},
    ))

    non_depth = [f for f in video_files if "depth" not in f.stem.lower()]
    sample = non_depth or video_files
    accessible = sum(1 for f in sample if f.exists() and f.stat().st_size > 0)
    if sample:
        accessible_ratio = accessible / len(sample)
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="video_accessibility",
            passed=accessible_ratio >= thresholds["visual_min_accessible_ratio"],
            message=f"Accessible videos {accessible}/{len(sample)}",
            level="major",
            value={
                "accessible_videos": accessible,
                "sample_size": len(sample),
                "accessible_ratio": accessible_ratio,
            },
        ))

    sampled_frames: list[np.ndarray] = []
    video_metrics: list[dict[str, float]] = []
    for video_path in sample:
        feature_key = _feature_key_for_video_path(video_path, visual_feature_keys)
        stream = _stream_label(video_path, feature_key)
        clip_start_s, clip_end_s = _video_clip_bounds(episode_meta, feature_key)
        frames, fps, width, height, _frame_count = _sample_video_frames(
            video_path,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
        )
        if fps <= 0 or width <= 0 or height <= 0:
            meta_fps, meta_width, meta_height = _video_metadata_from_feature(
                visual_features.get(feature_key)
            )
            fps = fps or meta_fps
            width = width or meta_width
            height = height or meta_height
        sampled_frames.extend(frames)
        _check_video_shape_and_rate(issues, operator_name, thresholds, stream, fps, width, height)

    if not sampled_frames:
        sampled_frames = [frame for _, frame in _iter_visual_parquet_frames(rows)]

    for frame in sampled_frames:
        video_metrics.append(_compute_visual_frame_stats(frame))

    if video_metrics:
        _check_visual_metrics(issues, operator_name, video_metrics, thresholds)

    return finalize_validator(operator_name, issues, details={"sample_size": len(sample)})


def _check_video_shape_and_rate(
    issues: list[dict[str, Any]],
    operator_name: str,
    thresholds: dict[str, float],
    stream: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    min_width = thresholds["visual_min_resolution_width"]
    min_height = thresholds["visual_min_resolution_height"]
    if width and height:
        resolution_passed = width >= min_width and height >= min_height
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="video_resolution",
            passed=resolution_passed,
            message=f"{stream}: {width}x{height} (min {int(min_width)}x{int(min_height)})",
            level="major",
            value={"stream": stream, "width": width, "height": height},
        ))
    elif min_width > 0 or min_height > 0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="video_resolution",
            passed=False,
            message=f"{stream}: video resolution unavailable (min {int(min_width)}x{int(min_height)})",
            level="major",
            value={"stream": stream, "width": None, "height": None},
        ))

    min_fps = thresholds["visual_min_frame_rate"]
    if fps > 0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="video_fps",
            passed=fps >= min_fps,
            message=f"{stream}: {fps:.1f} Hz (min {min_fps:.1f} Hz)",
            level="major",
            value={"stream": stream, "fps": fps},
        ))
    elif min_fps > 0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="video_fps",
            passed=False,
            message=f"{stream}: video frame rate unavailable (min {min_fps:.1f} Hz)",
            level="major",
            value={"stream": stream, "fps": None},
        ))


def _check_visual_metrics(
    issues: list[dict[str, Any]],
    operator_name: str,
    video_metrics: list[dict[str, float]],
    thresholds: dict[str, float],
) -> None:
    avg_over = statistics.fmean(metric["overexposure"] for metric in video_metrics)
    avg_under = statistics.fmean(metric["underexposure"] for metric in video_metrics)
    avg_black = statistics.fmean(metric["black"] for metric in video_metrics)
    avg_white = statistics.fmean(metric["white"] for metric in video_metrics)
    avg_shift = statistics.fmean(metric["color_shift"] for metric in video_metrics)

    issues.extend([
        make_issue(
            operator_name=operator_name,
            check_name="overexposure_ratio",
            passed=avg_over <= thresholds["visual_overexposure_ratio_max"],
            message=f"Overexposed pixels {avg_over * 100:.1f}% (max {thresholds['visual_overexposure_ratio_max'] * 100:.0f}%)",
            level="major",
            value={"ratio": avg_over},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="underexposure_ratio",
            passed=avg_under <= thresholds["visual_underexposure_ratio_max"],
            message=f"Underexposed pixels {avg_under * 100:.1f}% (max {thresholds['visual_underexposure_ratio_max'] * 100:.0f}%)",
            level="major",
            value={"ratio": avg_under},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="abnormal_frame_ratio",
            passed=avg_black < thresholds["visual_abnormal_black_ratio_max"] and avg_white < thresholds["visual_abnormal_white_ratio_max"],
            message=f"Black {avg_black * 100:.1f}%, white {avg_white * 100:.1f}%",
            level="major",
            value={"black_ratio": avg_black, "white_ratio": avg_white},
        ),
        make_issue(
            operator_name=operator_name,
            check_name="color_shift",
            passed=avg_shift <= thresholds["visual_color_shift_max"],
            message=f"Color shift {avg_shift * 100:.1f}% (max {thresholds['visual_color_shift_max'] * 100:.0f}%)",
            level="major",
            value={"color_shift": avg_shift},
        ),
    ])


def validate_depth_assets(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    operator_name = "depth"
    thresholds = _merge_threshold_overrides(threshold_overrides)
    video_files = data["video_files"]
    rows = data["rows"]
    info = data["info"]
    depth_files = [f for f in video_files if "depth" in f.stem.lower()]
    issues: list[dict[str, Any]] = []
    min_stream_count = int(thresholds["depth_min_stream_count"])
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    declared_depth_features = [
        key for key in features
        if "depth" in key.lower()
    ]

    if not declared_depth_features and min_stream_count <= 0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_streams",
            passed=True,
            message="No depth streams declared in dataset metadata, skipping depth validation",
            level="info",
            value={"depth_streams": 0},
        ))
        return finalize_validator(operator_name, issues, details={"depth_streams": 0, "skipped": True})

    if len(depth_files) < min_stream_count:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_streams",
            passed=False,
            message=f"Depth streams {len(depth_files)}",
            level="major",
            value={"depth_streams": len(depth_files)},
        ))
        return finalize_validator(operator_name, issues, details={"depth_streams": len(depth_files)})

    sample = depth_files[:2]
    accessible = sum(1 for f in sample if f.exists() and f.stat().st_size > 0)
    accessible_ratio = (accessible / len(sample)) if sample else 0.0
    issues.append(make_issue(
        operator_name=operator_name,
        check_name="depth_accessibility",
        passed=accessible_ratio >= thresholds["depth_min_accessible_ratio"],
        message=f"Accessible depth assets {accessible}/{len(sample)}",
        level="major",
        value={
            "accessible_depth_assets": accessible,
            "sample_size": len(sample),
            "accessible_ratio": accessible_ratio,
        },
    ))

    depth_frames = [frame for _, frame in _iter_depth_parquet_frames(rows)]
    invalid_ratios = [_compute_depth_invalid_ratio(frame) for frame in depth_frames]
    continuity_ratios: list[float] = []
    previous_frame: np.ndarray | None = None
    for frame in depth_frames:
        continuity = _compute_depth_continuity(frame, previous_frame)
        if continuity is not None:
            continuity_ratios.append(continuity)
        previous_frame = frame

    if invalid_ratios:
        avg_invalid = statistics.fmean(invalid_ratios)
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_invalid_ratio",
            passed=avg_invalid <= thresholds["depth_invalid_pixel_max"],
            message=f"Invalid depth pixels {avg_invalid * 100:.1f}% (max {thresholds['depth_invalid_pixel_max'] * 100:.0f}%)",
            level="major",
            value={"invalid_ratio": avg_invalid},
        ))
    elif thresholds["depth_invalid_pixel_max"] < 1.0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_invalid_ratio",
            passed=False,
            message=(
                "Depth invalid-pixel ratio unavailable "
                f"(max {thresholds['depth_invalid_pixel_max'] * 100:.0f}%)"
            ),
            level="major",
            value={"invalid_ratio": None},
        ))

    if continuity_ratios:
        avg_continuity = statistics.fmean(continuity_ratios)
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_continuity",
            passed=avg_continuity >= thresholds["depth_continuity_min"],
            message=f"Depth continuity {avg_continuity * 100:.1f}% (min {thresholds['depth_continuity_min'] * 100:.0f}%)",
            level="major",
            value={"continuity": avg_continuity},
        ))
    elif thresholds["depth_continuity_min"] > 0:
        issues.append(make_issue(
            operator_name=operator_name,
            check_name="depth_continuity",
            passed=False,
            message=f"Depth continuity unavailable (min {thresholds['depth_continuity_min'] * 100:.0f}%)",
            level="major",
            value={"continuity": None},
        ))

    return finalize_validator(operator_name, issues, details={"depth_streams": len(depth_files)})
