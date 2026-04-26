from __future__ import annotations

import io
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .validators import finalize_validator, make_issue, safe_float, _merge_threshold_overrides


def _sample_video_frames(video_path: Path, max_samples: int = 10) -> tuple[list[np.ndarray], float, int, int, int]:
    try:
        import av
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        frame_count = int(stream.frames or 0)
        sample_step = max(1, frame_count // max(max_samples, 1)) if frame_count > 0 else 1
        frames: list[np.ndarray] = []
        for index, frame in enumerate(container.decode(stream)):
            if index % sample_step != 0:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_samples:
                break
        container.close()
        return frames, fps, width, height, frame_count
    except Exception:
        pass

    try:
        import cv2
    except Exception:
        return [], 0.0, 0, 0, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0, 0, 0, 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    if frame_count <= 0:
        cap.release()
        return frames, fps, width, height, frame_count
    sample_step = max(1, frame_count // max(max_samples, 1))
    for index in range(0, frame_count, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
        if len(frames) >= max_samples:
            break
    cap.release()
    return frames, fps, width, height, frame_count


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
    sample_step = max(1, len(rows) // max(max_samples, 1))
    for row in rows[::sample_step]:
        for key, value in row.items():
            if "observation.images" not in key or "depth" in key.lower():
                continue
            decoded = _decode_image_like(value)
            if decoded is None:
                continue
            frames.append((key, decoded))
            if len(frames) >= max_samples:
                return frames
    return frames


def _iter_depth_parquet_frames(rows: list[dict[str, Any]], max_samples: int = 10) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    if not rows:
        return frames
    sample_step = max(1, len(rows) // max(max_samples, 1))
    for row in rows[::sample_step]:
        for key, value in row.items():
            if "depth" not in key.lower():
                continue
            decoded = _decode_image_like(value)
            if decoded is None:
                continue
            frames.append((key, decoded))
            if len(frames) >= max_samples:
                return frames
    return frames


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
    issues: list[dict[str, Any]] = []
    min_video_count = int(thresholds["visual_min_video_count"])
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    visual_feature_keys = [
        key for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video" and "depth" not in key.lower()
    ]

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
        level="major" if not video_files else "minor",
        value={"video_count": len(video_files)},
    ))

    non_depth = [f for f in video_files if "depth" not in f.stem.lower()]
    sample = non_depth[:2] or video_files[:2]
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
    if sample:
        frames, fps, width, height, _frame_count = _sample_video_frames(sample[0])
        if (fps <= 0 or width <= 0 or height <= 0) and visual_feature_keys:
            meta_fps, meta_width, meta_height = _video_metadata_from_feature(
                features.get(visual_feature_keys[0])
            )
            fps = fps or meta_fps
            width = width or meta_width
            height = height or meta_height
        sampled_frames = frames
        if width and height:
            issues.append(make_issue(
                operator_name=operator_name,
                check_name="video_resolution",
                passed=width >= thresholds["visual_min_resolution_width"] and height >= thresholds["visual_min_resolution_height"],
                message=f"{width}x{height} (min {int(thresholds['visual_min_resolution_width'])}x{int(thresholds['visual_min_resolution_height'])})",
                level="major",
                value={"width": width, "height": height},
            ))
        if fps > 0:
            issues.append(make_issue(
                operator_name=operator_name,
                check_name="video_fps",
                passed=fps >= thresholds["visual_min_frame_rate"],
                message=f"{fps:.1f} Hz (min {thresholds['visual_min_frame_rate']:.1f} Hz)",
                level="major",
                value={"fps": fps},
            ))

    if not sampled_frames:
        sampled_frames = [frame for _, frame in _iter_visual_parquet_frames(rows)]

    for frame in sampled_frames:
        video_metrics.append(_compute_visual_frame_stats(frame))

    if video_metrics:
        _check_visual_metrics(issues, operator_name, video_metrics, thresholds)

    return finalize_validator(operator_name, issues, details={"sample_size": len(sample)})


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

    return finalize_validator(operator_name, issues, details={"depth_streams": len(depth_files)})
