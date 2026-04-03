from __future__ import annotations

import math
import statistics
from typing import Any

# ---------------------------------------------------------------------------
# Field candidate lists for resolving vectors from episode row dicts
# ---------------------------------------------------------------------------

STATE_FIELD_CANDIDATES = [
    "observation.state",
    "observation.base_state",
    "observation.robot_state",
    "state",
    "robot_state",
]

ACTION_FIELD_CANDIDATES = [
    "action",
    "action.base",
    "action_base",
    "actions",
    "control",
    "command",
]

TIMESTAMP_FIELD_CANDIDATES = [
    "timestamp",
    "timestamp_utc",
    "observation.timestamp",
    "time",
]

FRAME_INDEX_FIELD_CANDIDATES = [
    "frame_index",
    "index",
    "row_idx",
]

TASK_FIELD_CANDIDATES = [
    "task_id",
    "task",
    "instruction",
]

# ---------------------------------------------------------------------------
# Row resolution helpers
# ---------------------------------------------------------------------------


def first_present_value(row: dict[str, Any], candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return None


def coerce_vector(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return []


def resolve_state_vector(row: dict[str, Any]) -> list[Any]:
    return coerce_vector(first_present_value(row, STATE_FIELD_CANDIDATES))


def resolve_action_vector(row: dict[str, Any]) -> list[Any]:
    return coerce_vector(first_present_value(row, ACTION_FIELD_CANDIDATES))


def resolve_timestamp(row: dict[str, Any]) -> float | None:
    value = first_present_value(row, TIMESTAMP_FIELD_CANDIDATES)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_frame_index(row: dict[str, Any], fallback: int) -> int:
    value = first_present_value(row, FRAME_INDEX_FIELD_CANDIDATES)
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def resolve_task_value(row: dict[str, Any]) -> Any:
    return first_present_value(row, TASK_FIELD_CANDIDATES)


# ---------------------------------------------------------------------------
# Scalar statistics
# ---------------------------------------------------------------------------


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = clamp(ratio, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# Series summarization
# ---------------------------------------------------------------------------


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "stdev": 0.0,
            "range": 0.0,
            "delta": 0.0,
            "mean_abs_step": 0.0,
        }

    steps = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
    return {
        "mean": mean(values),
        "stdev": stdev(values),
        "range": max(values) - min(values),
        "delta": values[-1] - values[0],
        "mean_abs_step": mean(steps),
    }


# ---------------------------------------------------------------------------
# Episode feature vector
# ---------------------------------------------------------------------------


def build_episode_feature_vector(
    joint_trajectory: dict[str, Any] | None,
    *,
    max_joints: int = 6,
) -> dict[str, Any]:
    joint_entries = (joint_trajectory or {}).get("joint_trajectories", [])
    vector: list[float] = []
    features: list[dict[str, Any]] = []

    for joint in joint_entries[:max_joints]:
        values = [
            float(value)
            for value in (
                joint.get("state_values")
                or joint.get("action_values")
                or []
            )
            if value is not None
        ]
        summary = summarize_series(values)
        vector.extend([
            summary["mean"],
            summary["stdev"],
            summary["range"],
            summary["delta"],
            summary["mean_abs_step"],
        ])
        features.append({
            "joint_name": joint.get("joint_name") or joint.get("state_name") or joint.get("action_name"),
            **summary,
        })

    if not vector:
        vector = [0.0, 0.0, 0.0, 0.0, 0.0]

    return {
        "vector": vector,
        "joint_features": features,
    }


# ---------------------------------------------------------------------------
# Sequence sampling and normalization
# ---------------------------------------------------------------------------


def sample_sequence(values: list[Any], max_points: int) -> list[Any]:
    if len(values) <= max_points or max_points <= 1:
        return values[:]

    sampled: list[Any] = []
    for sample_index in range(max_points):
        source_index = round(sample_index * (len(values) - 1) / (max_points - 1))
        sampled.append(values[source_index])
    return sampled


def normalize_scalar_series(values: list[float]) -> list[float]:
    if not values:
        return []
    center = statistics.median(values)
    spread = percentile([abs(value - center) for value in values], 0.5) or 1.0
    return [(value - center) / spread for value in values]


def build_episode_sequence(
    rows: list[dict[str, Any]],
    *,
    max_dims: int = 6,
    max_points: int = 80,
) -> list[list[float]]:
    raw_sequence: list[list[float]] = []
    for row in rows:
        state = resolve_state_vector(row)
        action = resolve_action_vector(row)
        source = state or action
        if not source:
            continue
        vector = _build_capped_vector(source, max_dims)
        if vector:
            raw_sequence.append(vector)

    if not raw_sequence:
        return [[0.0] * max_dims]

    sampled = sample_sequence(raw_sequence, max_points=max_points)
    dimension_count = max(len(vector) for vector in sampled)
    normalized_dimensions = _normalize_sampled_dimensions(sampled, dimension_count)

    normalized_sequence: list[list[float]] = []
    for row_index in range(len(sampled)):
        normalized_sequence.append([
            normalized_dimensions[dim_index][row_index]
            for dim_index in range(dimension_count)
        ])
    return normalized_sequence


def _build_capped_vector(source: list[Any], max_dims: int) -> list[float]:
    vector: list[float] = []
    for index in range(min(max_dims, len(source))):
        value = source[index]
        vector.append(float(value) if value is not None else 0.0)
    return vector


def _normalize_sampled_dimensions(
    sampled: list[list[float]],
    dimension_count: int,
) -> list[list[float]]:
    normalized_dimensions: list[list[float]] = []
    for dimension_index in range(dimension_count):
        dimension_values = [
            vector[dimension_index] if dimension_index < len(vector) else 0.0
            for vector in sampled
        ]
        normalized_dimensions.append(normalize_scalar_series(dimension_values))
    return normalized_dimensions


# ---------------------------------------------------------------------------
# Joint name utilities
# ---------------------------------------------------------------------------


def extract_joint_names(feature_names: Any) -> list[str]:
    if isinstance(feature_names, list):
        return [str(name) for name in feature_names if str(name).strip()]

    if isinstance(feature_names, dict):
        flattened_names: list[str] = []
        for value in feature_names.values():
            flattened_names.extend(extract_joint_names(value))
        return flattened_names

    return []


def sample_indices(total_points: int, max_points: int) -> list[int]:
    if total_points <= 0:
        return []
    if total_points <= max_points or max_points <= 1:
        return list(range(total_points))

    indices: list[int] = []
    for sample_index in range(max_points):
        point_index = round(sample_index * (total_points - 1) / (max_points - 1))
        if not indices or point_index != indices[-1]:
            indices.append(point_index)

    if indices[-1] != total_points - 1:
        indices.append(total_points - 1)
    return indices


def normalize_joint_names(
    feature_config: dict[str, Any] | None,
    fallback_size: int,
) -> list[str]:
    names = feature_config.get("names", []) if isinstance(feature_config, dict) else []
    normalized_names = extract_joint_names(names)

    if len(normalized_names) >= fallback_size:
        return normalized_names[:fallback_size]

    for index in range(len(normalized_names), fallback_size):
        normalized_names.append(f"joint_{index + 1}")
    return normalized_names


def build_joint_trajectory_payload(
    rows: list[dict[str, Any]],
    action_names: list[str],
    state_names: list[str],
    *,
    max_points: int = 180,
) -> dict[str, Any]:
    if not rows:
        return _empty_trajectory_payload()

    first_action = resolve_action_vector(rows[0])
    first_state = resolve_state_vector(rows[0])
    joint_count = max(len(first_action), len(first_state), len(action_names), len(state_names))
    if joint_count == 0:
        return {**_empty_trajectory_payload(), "total_points": len(rows)}

    normalized_action_names = normalize_joint_names({"names": action_names}, joint_count)
    normalized_state_names = normalize_joint_names({"names": state_names}, joint_count)
    sample_points = sample_indices(len(rows), max_points)
    sampled_rows = [rows[index] for index in sample_points]
    time_values = _extract_time_values(sampled_rows)
    frame_values = [resolve_frame_index(row, index) for index, row in enumerate(sampled_rows)]

    trajectories = _build_trajectories(
        sampled_rows, joint_count, normalized_action_names, normalized_state_names,
    )

    return {
        "x_axis_key": "timestamp",
        "x_values": time_values,
        "time_values": time_values,
        "frame_values": frame_values,
        "joint_trajectories": trajectories,
        "sampled_points": len(sampled_rows),
        "total_points": len(rows),
    }


def _empty_trajectory_payload() -> dict[str, Any]:
    return {
        "x_axis_key": "timestamp",
        "x_values": [],
        "time_values": [],
        "frame_values": [],
        "joint_trajectories": [],
        "sampled_points": 0,
        "total_points": 0,
    }


def _extract_time_values(sampled_rows: list[dict[str, Any]]) -> list[float]:
    return [
        resolve_timestamp(row) if resolve_timestamp(row) is not None else float(index)
        for index, row in enumerate(sampled_rows)
    ]


def _build_trajectories(
    sampled_rows: list[dict[str, Any]],
    joint_count: int,
    action_names: list[str],
    state_names: list[str],
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for joint_index in range(joint_count):
        action_values, state_values, has_values = _collect_joint_values(
            sampled_rows, joint_index,
        )
        if not has_values:
            continue
        trajectories.append({
            "joint_name": action_names[joint_index] or state_names[joint_index],
            "action_name": action_names[joint_index],
            "state_name": state_names[joint_index],
            "action_values": action_values,
            "state_values": state_values,
        })
    return trajectories


def _collect_joint_values(
    sampled_rows: list[dict[str, Any]],
    joint_index: int,
) -> tuple[list[float | None], list[float | None], bool]:
    action_values: list[float | None] = []
    state_values: list[float | None] = []
    has_values = False

    for row in sampled_rows:
        action = resolve_action_vector(row)
        state = resolve_state_vector(row)
        action_value = action[joint_index] if joint_index < len(action) else None
        state_value = state[joint_index] if joint_index < len(state) else None

        action_values.append(float(action_value) if action_value is not None else None)
        state_values.append(float(state_value) if state_value is not None else None)
        if action_value is not None or state_value is not None:
            has_values = True

    return action_values, state_values, has_values
