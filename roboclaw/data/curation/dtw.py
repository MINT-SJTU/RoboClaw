from __future__ import annotations

import math
from typing import Any, Callable

from .features import mean

# ---------------------------------------------------------------------------
# DTW configuration constants
# ---------------------------------------------------------------------------

CARTESIAN_20D_GROUP_WEIGHTS = {
    "eef_pos": 1.0,
    "eef_rot6d": 0.8,
    "gripper": 1.5,
    "delta_pos": 0.5,
    "delta_rot6d": 0.3,
    "delta_gripper": 0.8,
}
CARTESIAN_20D_WINDOW_RATIO = 0.15
DEFAULT_DTW_HUBER_DELTA = 1.0

# ---------------------------------------------------------------------------
# Distance primitives
# ---------------------------------------------------------------------------


def euclidean_distance(left: list[float], right: list[float]) -> float:
    length = max(len(left), len(right))
    padded_left = left + [0.0] * (length - len(left))
    padded_right = right + [0.0] * (length - len(right))
    return math.sqrt(
        sum((padded_left[index] - padded_right[index]) ** 2 for index in range(length))
    )


def vector_distance(left: list[float], right: list[float]) -> float:
    return euclidean_distance(left, right)


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimension_count = max(len(vector) for vector in vectors)
    averaged: list[float] = []
    for dimension_index in range(dimension_count):
        values = [
            vector[dimension_index] if dimension_index < len(vector) else 0.0
            for vector in vectors
        ]
        averaged.append(mean(values))
    return averaged


def huber_loss(value: float, delta: float = DEFAULT_DTW_HUBER_DELTA) -> float:
    absolute = abs(value)
    if absolute <= delta:
        return 0.5 * absolute * absolute
    return delta * (absolute - (0.5 * delta))


# ---------------------------------------------------------------------------
# Grouped Huber distance
# ---------------------------------------------------------------------------


def grouped_huber_distance(
    left: list[float],
    right: list[float],
    *,
    groups: dict[str, list[int]] | None = None,
    group_weights: dict[str, float] | None = None,
    huber_delta: float = DEFAULT_DTW_HUBER_DELTA,
) -> float:
    if not groups:
        return vector_distance(left, right)

    length = max(len(left), len(right))
    padded_left = left + [0.0] * (length - len(left))
    padded_right = right + [0.0] * (length - len(right))
    covered_indices: set[int] = set()
    total_cost = 0.0

    for group_name, group_indices in groups.items():
        valid_indices = [index for index in group_indices if 0 <= index < length]
        if not valid_indices:
            continue
        covered_indices.update(valid_indices)
        squared_norm = sum(
            (padded_left[index] - padded_right[index]) ** 2
            for index in valid_indices
        )
        weight = float(group_weights.get(group_name, 1.0) if group_weights else 1.0)
        total_cost += weight * huber_loss(math.sqrt(squared_norm), huber_delta)

    for index in range(length):
        if index in covered_indices:
            continue
        total_cost += huber_loss(padded_left[index] - padded_right[index], huber_delta)

    return total_cost


# ---------------------------------------------------------------------------
# DTW configuration resolver
# ---------------------------------------------------------------------------


def resolve_dtw_configuration(
    *,
    left_mode: str | None = None,
    right_mode: str | None = None,
    left_groups: dict[str, list[int]] | None = None,
    right_groups: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    normalized_left_groups = left_groups or {}
    normalized_right_groups = right_groups or {}
    if left_mode != "cartesian_20d" or right_mode != "cartesian_20d":
        return {}
    if not normalized_left_groups or normalized_left_groups != normalized_right_groups:
        return {}
    return {
        "groups": normalized_left_groups,
        "group_weights": CARTESIAN_20D_GROUP_WEIGHTS,
        "window_ratio": CARTESIAN_20D_WINDOW_RATIO,
        "huber_delta": DEFAULT_DTW_HUBER_DELTA,
    }


# ---------------------------------------------------------------------------
# DTW internals
# ---------------------------------------------------------------------------


def _validate_dtw_distance(distance: float, left_length: int, right_length: int) -> float:
    if math.isnan(distance) or math.isinf(distance):
        return float(max(left_length, right_length))
    return max(distance, 0.0)


def _resolve_dtw_window(
    left_length: int,
    right_length: int,
    window_ratio: float | None,
) -> int | None:
    if window_ratio is None:
        return None
    safe_ratio = max(float(window_ratio), 0.0)
    return max(
        abs(left_length - right_length),
        int(math.ceil(max(left_length, right_length) * safe_ratio)),
    )


def _compute_dtw_cost_matrix(
    left: list[list[float]],
    right: list[list[float]],
    *,
    groups: dict[str, list[int]] | None = None,
    group_weights: dict[str, float] | None = None,
    window_ratio: float | None = None,
    huber_delta: float = DEFAULT_DTW_HUBER_DELTA,
) -> tuple[list[list[float]], list[list[int]]]:
    left_length = len(left)
    right_length = len(right)
    matrix = [
        [math.inf for _ in range(right_length + 1)]
        for _ in range(left_length + 1)
    ]
    steps = [
        [0 for _ in range(right_length + 1)]
        for _ in range(left_length + 1)
    ]
    matrix[0][0] = 0.0

    window = _resolve_dtw_window(left_length, right_length, window_ratio)

    for left_index in range(1, left_length + 1):
        right_start, right_end = _window_bounds(left_index, right_length, window)
        _fill_cost_row(
            matrix, steps, left, right,
            left_index, right_start, right_end,
            groups=groups, group_weights=group_weights, huber_delta=huber_delta,
        )

    if window is not None and math.isinf(matrix[left_length][right_length]):
        return _compute_dtw_cost_matrix(
            left, right,
            groups=groups,
            group_weights=group_weights,
            window_ratio=None,
            huber_delta=huber_delta,
        )

    return matrix, steps


def _window_bounds(
    left_index: int,
    right_length: int,
    window: int | None,
) -> tuple[int, int]:
    if window is None:
        return 1, right_length
    return max(1, left_index - window), min(right_length, left_index + window)


def _fill_cost_row(
    matrix: list[list[float]],
    steps: list[list[int]],
    left: list[list[float]],
    right: list[list[float]],
    left_index: int,
    right_start: int,
    right_end: int,
    *,
    groups: dict[str, list[int]] | None,
    group_weights: dict[str, float] | None,
    huber_delta: float,
) -> None:
    for right_index in range(right_start, right_end + 1):
        cost = grouped_huber_distance(
            left[left_index - 1],
            right[right_index - 1],
            groups=groups,
            group_weights=group_weights,
            huber_delta=huber_delta,
        )
        candidates = [
            (matrix[left_index - 1][right_index], steps[left_index - 1][right_index]),
            (matrix[left_index][right_index - 1], steps[left_index][right_index - 1]),
            (matrix[left_index - 1][right_index - 1], steps[left_index - 1][right_index - 1]),
        ]
        best_cost, best_steps = min(candidates, key=lambda item: (item[0], item[1]))
        if math.isinf(best_cost):
            continue
        matrix[left_index][right_index] = cost + best_cost
        steps[left_index][right_index] = best_steps + 1


# ---------------------------------------------------------------------------
# Public DTW functions
# ---------------------------------------------------------------------------


def dtw_distance(
    left: list[list[float]],
    right: list[list[float]],
    *,
    groups: dict[str, list[int]] | None = None,
    group_weights: dict[str, float] | None = None,
    window_ratio: float | None = None,
    huber_delta: float = DEFAULT_DTW_HUBER_DELTA,
) -> float:
    if not left or not right:
        return math.inf
    left_length = len(left)
    right_length = len(right)
    matrix, steps = _compute_dtw_cost_matrix(
        left, right,
        groups=groups,
        group_weights=group_weights,
        window_ratio=window_ratio,
        huber_delta=huber_delta,
    )
    normalizer = max(steps[left_length][right_length], 1)
    distance = matrix[left_length][right_length] / normalizer
    return _validate_dtw_distance(distance, left_length, right_length)


def dtw_alignment(
    left: list[list[float]],
    right: list[list[float]],
    *,
    groups: dict[str, list[int]] | None = None,
    group_weights: dict[str, float] | None = None,
    window_ratio: float | None = None,
    huber_delta: float = DEFAULT_DTW_HUBER_DELTA,
) -> tuple[float, list[tuple[int, int]]]:
    if not left or not right:
        return math.inf, []

    left_length = len(left)
    right_length = len(right)
    matrix, steps = _compute_dtw_cost_matrix(
        left, right,
        groups=groups,
        group_weights=group_weights,
        window_ratio=window_ratio,
        huber_delta=huber_delta,
    )

    path = _traceback_alignment(matrix, left_length, right_length)
    distance = matrix[left_length][right_length] / max(steps[left_length][right_length], 1)
    return _validate_dtw_distance(distance, left_length, right_length), path


def _traceback_alignment(
    matrix: list[list[float]],
    left_length: int,
    right_length: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    left_index = left_length
    right_index = right_length

    while left_index > 0 or right_index > 0:
        path.append((max(left_index - 1, 0), max(right_index - 1, 0)))

        if left_index == 0:
            right_index -= 1
            continue
        if right_index == 0:
            left_index -= 1
            continue

        candidates = [
            (matrix[left_index - 1][right_index - 1], left_index - 1, right_index - 1),
            (matrix[left_index - 1][right_index], left_index - 1, right_index),
            (matrix[left_index][right_index - 1], left_index, right_index - 1),
        ]
        _, left_index, right_index = min(candidates, key=lambda item: item[0])

    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Distance matrix builders
# ---------------------------------------------------------------------------


def build_distance_matrix(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return build_distance_matrix_with_progress(entries)[0]


def build_distance_matrix_with_progress(
    entries: list[dict[str, Any]],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, float]], int]:
    entries = [entry for entry in entries if entry.get("sequence")]
    distances: dict[str, dict[str, float]] = {
        entry["record_key"]: {}
        for entry in entries
    }
    total_pairs = (len(entries) * (len(entries) - 1)) // 2
    completed_pairs = 0

    if progress_callback is not None:
        progress_callback(completed_pairs, total_pairs)

    for left_index, entry in enumerate(entries):
        key = entry["record_key"]
        distances[key][key] = 0.0
        for other in entries[left_index + 1:]:
            other_key = other["record_key"]
            dtw_configuration = resolve_dtw_configuration(
                left_mode=entry.get("canonical_mode"),
                right_mode=other.get("canonical_mode"),
                left_groups=entry.get("canonical_groups"),
                right_groups=other.get("canonical_groups"),
            )
            distance = dtw_distance(entry["sequence"], other["sequence"], **dtw_configuration)
            distances[key][other_key] = distance
            distances[other_key][key] = distance
            completed_pairs += 1
            if progress_callback is not None:
                progress_callback(completed_pairs, total_pairs)
    return distances, total_pairs
