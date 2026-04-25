from __future__ import annotations

from typing import Any

from .dtw import dtw_alignment
from .features import (
    clamp,
    mean,
    normalize_scalar_series,
    percentile,
    resolve_action_vector,
    resolve_state_vector,
    resolve_timestamp,
    sample_indices,
)

# ---------------------------------------------------------------------------
# Quality tags
# ---------------------------------------------------------------------------


def derive_quality_tags(
    issues: list[dict[str, Any]],
    *,
    overall_score: float,
) -> list[str]:
    tags: set[str] = set()
    failed_issues = [issue for issue in issues if not issue.get("passed")]
    failed_critical_or_major = any(
        issue.get("level") in {"critical", "major"}
        for issue in failed_issues
    )

    if failed_critical_or_major:
        tags.add("quality-risk")
    elif failed_issues or overall_score < 85:
        tags.add("quality-watch")
    else:
        tags.add("quality-pass")

    for issue in issues:
        if issue.get("passed"):
            continue
        _tag_from_operator(tags, str(issue.get("operator_name", "unknown")).lower())

    return sorted(tags)


def _tag_from_operator(tags: set[str], operator_name: str) -> None:
    mapping = {
        "metadata": "metadata-risk",
        "timing": "timing-risk",
        "action": "motion-risk",
        "visual": "visual-risk",
        "depth": "depth-risk",
    }
    for token, tag in mapping.items():
        if token in operator_name:
            tags.add(tag)
            return
    tags.add("quality-risk")


# ---------------------------------------------------------------------------
# Phase progress
# ---------------------------------------------------------------------------


def build_phase_progress(
    spans: list[dict[str, Any]],
    *,
    duration_s: float,
) -> list[dict[str, Any]]:
    safe_duration = max(duration_s, 1.0)
    progress: list[dict[str, Any]] = []
    for span in spans:
        start_time = float(span.get("startTime", 0.0))
        end_time = float(span.get("endTime") if span.get("endTime") is not None else start_time)
        progress.append({
            "label": span.get("label", "Annotation"),
            "start_progress": clamp(start_time / safe_duration, 0.0, 1.0),
            "end_progress": clamp(end_time / safe_duration, 0.0, 1.0),
        })
    return progress


# ---------------------------------------------------------------------------
# Confidence payload
# ---------------------------------------------------------------------------


def build_confidence_payload(
    *,
    annotation_count: int,
    quality_score: float,
    prototype_score: float,
) -> dict[str, float]:
    annotation_signal = min(annotation_count / 4.0, 1.0)
    quality_signal = clamp(quality_score / 100.0, 0.0, 1.0)
    prototype_signal = clamp(prototype_score, 0.0, 1.0)
    overall = mean([annotation_signal, quality_signal, prototype_signal])
    return {
        "overall": round(overall, 4),
        "annotation_signal": round(annotation_signal, 4),
        "quality_signal": round(quality_signal, 4),
        "prototype_signal": round(prototype_signal, 4),
    }


# ---------------------------------------------------------------------------
# Annotation propagation
# ---------------------------------------------------------------------------

def build_alignment_series(
    rows: list[dict[str, Any]],
    *,
    max_dims: int = 6,
    max_points: int = 90,
) -> tuple[list[list[float]], list[float]]:
    """Build a normalized per-row vector series plus a relative time axis.

    This intentionally mirrors ``build_episode_sequence`` behavior but also
    returns a monotonic time axis so annotations can be aligned with DTW.
    """
    if max_dims <= 0:
        return [[0.0]], [0.0]

    raw_vectors: list[list[float]] = []
    raw_times: list[float] = []

    for fallback_index, row in enumerate(rows):
        state = resolve_state_vector(row)
        action = resolve_action_vector(row)
        source = state or action
        if not source:
            continue

        capped: list[float] = []
        capped_size = min(max_dims, len(source))
        for index in range(capped_size):
            value = source[index]
            try:
                capped.append(float(value) if value is not None else 0.0)
            except (TypeError, ValueError):
                capped.append(0.0)

        # If the source has more dims, prefer keeping the last dim (often gripper)
        # by replacing the last slot. This helps alignment across different robots.
        if len(source) > max_dims and capped:
            try:
                capped[-1] = float(source[-1]) if source[-1] is not None else 0.0
            except (TypeError, ValueError):
                capped[-1] = 0.0

        if not capped:
            continue
        raw_vectors.append(capped)

        timestamp = resolve_timestamp(row)
        raw_times.append(float(timestamp) if timestamp is not None else float(fallback_index))

    if len(raw_vectors) < 2:
        return [[0.0] * max_dims], [0.0]

    indices = sample_indices(len(raw_vectors), max_points=max_points)
    sampled_vectors = [raw_vectors[index] for index in indices]
    sampled_times = [raw_times[index] for index in indices]

    base_time = sampled_times[0] if sampled_times else 0.0
    rel_times = [max(float(t) - float(base_time), 0.0) for t in sampled_times]
    # Enforce monotonicity for downstream interpolation.
    for index in range(1, len(rel_times)):
        if rel_times[index] < rel_times[index - 1]:
            rel_times[index] = rel_times[index - 1]

    dim_count = max(len(vector) for vector in sampled_vectors)
    normalized_dimensions: list[list[float]] = []
    for dim_index in range(dim_count):
        dim_values = [
            vector[dim_index] if dim_index < len(vector) else 0.0
            for vector in sampled_vectors
        ]
        normalized_dimensions.append(normalize_scalar_series(dim_values))

    normalized_sequence: list[list[float]] = []
    for row_index in range(len(sampled_vectors)):
        normalized_sequence.append([
            normalized_dimensions[dim_index][row_index]
            for dim_index in range(dim_count)
        ])

    return normalized_sequence, rel_times


def _build_monotonic_index_map(
    path: list[tuple[int, int]],
    source_len: int,
    target_len: int,
) -> list[int]:
    if source_len <= 0 or target_len <= 0:
        return []
    buckets: list[list[int]] = [[] for _ in range(source_len)]
    for left_index, right_index in path:
        if 0 <= left_index < source_len and 0 <= right_index < target_len:
            buckets[left_index].append(right_index)

    mapping: list[int] = [0 for _ in range(source_len)]
    last = 0
    for index, right_indices in enumerate(buckets):
        if right_indices:
            mapped = int(round(sum(right_indices) / len(right_indices)))
        else:
            mapped = last
        mapped = max(mapped, last)  # enforce monotonic non-decreasing
        mapped = min(max(mapped, 0), target_len - 1)
        mapping[index] = mapped
        last = mapped
    return mapping


def _map_time_by_alignment(
    source_times: list[float],
    target_times: list[float],
    index_map: list[int],
    source_time: float,
) -> float:
    if not source_times or not target_times or not index_map:
        return 0.0

    safe_source_time = clamp(float(source_time), 0.0, float(source_times[-1] or 0.0))
    if safe_source_time <= float(source_times[0]):
        return float(target_times[index_map[0]])
    if safe_source_time >= float(source_times[-1]):
        return float(target_times[index_map[-1]])

    left = 0
    right = len(source_times) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if float(source_times[mid]) < safe_source_time:
            left = mid
        else:
            right = mid

    t0 = float(source_times[left])
    t1 = float(source_times[right])
    if t1 <= t0:
        return float(target_times[index_map[left]])
    weight = (safe_source_time - t0) / (t1 - t0)
    j0 = index_map[left]
    j1 = index_map[right]
    y0 = float(target_times[j0])
    y1 = float(target_times[j1])
    return y0 + (y1 - y0) * weight


def propagate_annotation_spans(
    spans: list[dict[str, Any]],
    *,
    source_duration: float,
    target_duration: float,
    target_record_key: str,
    prototype_score: float,
    source_time_axis: list[float] | None = None,
    target_time_axis: list[float] | None = None,
    alignment_path: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    safe_source_duration = max(source_duration, 1e-6)
    scale = max(target_duration, 0.0) / safe_source_duration

    index_map: list[int] | None = None
    if (
        alignment_path
        and source_time_axis
        and target_time_axis
        and len(source_time_axis) > 1
        and len(target_time_axis) > 1
    ):
        index_map = _build_monotonic_index_map(
            alignment_path,
            source_len=len(source_time_axis),
            target_len=len(target_time_axis),
        )

    propagated: list[dict[str, Any]] = []
    for span in spans:
        raw_start = float(span.get("startTime", 0.0))
        if index_map and source_time_axis and target_time_axis:
            start_time = _map_time_by_alignment(source_time_axis, target_time_axis, index_map, raw_start)
        else:
            start_time = raw_start * scale
        raw_end = span.get("endTime")
        if raw_end is None:
            end_time = None
        else:
            end_source = float(raw_end)
            if index_map and source_time_axis and target_time_axis:
                end_time = _map_time_by_alignment(source_time_axis, target_time_axis, index_map, end_source)
            else:
                end_time = end_source * scale

        if end_time is not None and end_time < start_time:
            end_time = start_time

        propagated.append({
            **span,
            "startTime": round(start_time, 4),
            "endTime": round(end_time, 4) if end_time is not None else None,
            "target_record_key": target_record_key,
            "propagated": True,
            "source": "propagated",
            "prototype_score": round(prototype_score, 4),
        })
    return propagated


# ---------------------------------------------------------------------------
# HF annotation rows
# ---------------------------------------------------------------------------


def build_hf_annotation_rows(
    *,
    dataset: str,
    record_key: str,
    record_key_field: str,
    spans: list[dict[str, Any]],
    quality_tags: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, span in enumerate(spans, start=1):
        rows.append({
            "dataset": dataset,
            "record_key_field": record_key_field,
            "record_key": record_key,
            "annotation_index": index,
            "label": span.get("label", "Annotation"),
            "text": span.get("text", ""),
            "category": span.get("category", "movement"),
            "start_time": span.get("startTime"),
            "end_time": span.get("endTime"),
            "tags": span.get("tags", []),
            "quality_tags": quality_tags,
        })
    return rows


# ---------------------------------------------------------------------------
# Grasp / place event detection
# ---------------------------------------------------------------------------


def detect_grasp_place_events(
    *,
    rows: list[dict[str, Any]],
    action_names: list[str],
    state_names: list[str],
    duration_s: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    gripper_index = _find_gripper_index(action_names, state_names)
    series, timestamps = _extract_gripper_series(rows, gripper_index)

    if len(series) < 5:
        return []

    lower = percentile(series, 0.1)
    upper = percentile(series, 0.9)
    if abs(upper - lower) < 1e-6:
        return []
    midpoint = (lower + upper) / 2.0

    close_index, open_index = _find_crossings(series, midpoint)
    return _build_grasp_place_annotations(close_index, open_index, timestamps, duration_s)


def _find_gripper_index(
    action_names: list[str],
    state_names: list[str],
) -> int | None:
    candidate_names = list(action_names or []) + list(state_names or [])
    lowered_names = [name.lower() for name in candidate_names]
    return next(
        (
            index
            for index, name in enumerate(lowered_names)
            if any(token in name for token in ("gripper", "claw", "finger", "hand"))
        ),
        None,
    )


def _extract_gripper_series(
    rows: list[dict[str, Any]],
    gripper_index: int | None,
) -> tuple[list[float], list[float]]:
    series: list[float] = []
    timestamps: list[float] = []
    current_gripper_index = gripper_index

    for row in rows:
        action = resolve_action_vector(row)
        state = resolve_state_vector(row)
        values = action or state
        if current_gripper_index is None and values:
            current_gripper_index = len(values) - 1
        if current_gripper_index is None or current_gripper_index >= len(values):
            continue
        value = values[current_gripper_index]
        if value is None:
            continue
        timestamp = resolve_timestamp(row)
        if timestamp is None:
            continue
        series.append(float(value))
        timestamps.append(timestamp)

    return series, timestamps


def _find_crossings(
    series: list[float],
    midpoint: float,
) -> tuple[int | None, int | None]:
    close_index = None
    open_index = None
    for index in range(1, len(series)):
        if close_index is None and series[index - 1] >= midpoint and series[index] < midpoint:
            close_index = index
        elif close_index is not None and series[index - 1] <= midpoint and series[index] > midpoint:
            open_index = index
            break
    return close_index, open_index


def _build_grasp_place_annotations(
    close_index: int | None,
    open_index: int | None,
    timestamps: list[float],
    duration_s: float,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    event_specs = [
        ("Grasp", close_index, "grasp", "#ff8a5b"),
        ("Place", open_index, "placement", "#44d7ff"),
    ]
    for label, index, category, color in event_specs:
        if index is None:
            continue
        event_time = max(timestamps[index] - timestamps[0], 0.0)
        window = min(max(duration_s * 0.04, 0.5), 1.6)
        annotations.append({
            "label": label,
            "text": f"Auto-detected {label.lower()} event from gripper state transition.",
            "category": category,
            "color": color,
            "startTime": round(event_time, 4),
            "endTime": round(min(event_time + window, duration_s), 4),
            "tags": ["Auto-Seed", "Gripper"],
        })
    return annotations
