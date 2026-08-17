from __future__ import annotations

from typing import Any

from .features import (
    clamp,
    mean,
    percentile,
    resolve_action_vector,
    resolve_state_vector,
    resolve_timestamp,
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


def propagate_annotation_spans(
    spans: list[dict[str, Any]],
    *,
    source_duration: float,
    target_duration: float,
    target_record_key: str,
    prototype_score: float,
) -> list[dict[str, Any]]:
    safe_source_duration = max(source_duration, 1e-6)
    scale = max(target_duration, 0.0) / safe_source_duration

    propagated: list[dict[str, Any]] = []
    for span in spans:
        start_time = float(span.get("startTime", 0.0)) * scale
        raw_end = span.get("endTime")
        end_time = float(raw_end) * scale if raw_end is not None else None
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
