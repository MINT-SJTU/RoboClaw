from __future__ import annotations

import importlib.util
import math
import os
from functools import lru_cache
from typing import Any, Callable

from .features import mean

# ---------------------------------------------------------------------------
# DTW configuration constants
# ---------------------------------------------------------------------------

CARTESIAN_20D_GROUP_WEIGHTS = {
    "eef_pos": 1.0,
    "eef_rot6d": 0.7,
    "gripper": 1.2,
    "delta_pos": 0.5,
    "delta_rot6d": 0.3,
    "delta_gripper": 0.8,
}
CARTESIAN_20D_WINDOW_RATIO = 0.15
DEFAULT_DTW_HUBER_DELTA = 1.0
CUDA_DTW_TARGET_CELLS_PER_BATCH = 4_000_000

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
    if _cuda_alignment_supported(
        left,
        right,
        groups=groups,
        group_weights=group_weights,
        window_ratio=window_ratio,
    ):
        return _cuda_dtw_alignment(left, right)

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
    distances, total_pairs, _stats = build_distance_matrix_with_progress_and_stats(
        entries,
        progress_callback=progress_callback,
    )
    return distances, total_pairs


def build_distance_matrix_with_progress_and_stats(
    entries: list[dict[str, Any]],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, float]], int, dict[str, Any]]:
    entries = [entry for entry in entries if entry.get("sequence")]
    distances: dict[str, dict[str, float]] = {
        entry["record_key"]: {}
        for entry in entries
    }
    total_pairs = (len(entries) * (len(entries) - 1)) // 2
    completed_pairs = 0
    stats = _empty_distance_stats()

    if progress_callback is not None:
        progress_callback(completed_pairs, total_pairs)

    for entry in entries:
        distances[entry["record_key"]][entry["record_key"]] = 0.0

    cuda_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    cpu_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    use_cuda = _cuda_dtw_enabled()

    for left_index, entry in enumerate(entries):
        for other in entries[left_index + 1:]:
            dtw_configuration = resolve_dtw_configuration(
                left_mode=entry.get("canonical_mode"),
                right_mode=other.get("canonical_mode"),
                left_groups=entry.get("canonical_groups"),
                right_groups=other.get("canonical_groups"),
            )
            if use_cuda and _pair_supports_cuda(entry, other, dtw_configuration):
                cuda_pairs.append((entry, other))
            else:
                cpu_pairs.append((entry, other, dtw_configuration))

    if cuda_pairs:
        cuda_result = _fill_cuda_distance_pairs(
            distances,
            cuda_pairs,
            completed_pairs=completed_pairs,
            total_pairs=total_pairs,
            progress_callback=progress_callback,
        )
        completed_pairs = cuda_result["completed_pairs"]
        stats.update(cuda_result["stats"])

    for entry, other, dtw_configuration in cpu_pairs:
        key = entry["record_key"]
        other_key = other["record_key"]
        distance = dtw_distance(entry["sequence"], other["sequence"], **dtw_configuration)
        distances[key][other_key] = distance
        distances[other_key][key] = distance
        completed_pairs += 1
        stats["cpu_pair_count"] += 1
        if progress_callback is not None:
            progress_callback(completed_pairs, total_pairs)

    stats["pair_count"] = total_pairs
    stats["backend"] = _resolve_distance_backend(stats)
    return distances, total_pairs, stats


def _empty_distance_stats() -> dict[str, Any]:
    return {
        "backend": "cpu",
        "pair_count": 0,
        "cuda_pair_count": 0,
        "cpu_pair_count": 0,
        "cuda_batch_count": 0,
        "cuda_device": None,
        "cuda_torch_version": None,
    }


def _cuda_dtw_enabled() -> bool:
    if os.environ.get("ROBOCLAW_DTW_BACKEND", "").strip().lower() == "cpu":
        return False
    return _cuda_available()


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _cuda_alignment_supported(
    left: list[list[float]],
    right: list[list[float]],
    *,
    groups: dict[str, list[int]] | None,
    group_weights: dict[str, float] | None,
    window_ratio: float | None,
) -> bool:
    if groups or group_weights or window_ratio is not None:
        return False
    return _cuda_dtw_enabled() and _sequence_is_numeric(left) and _sequence_is_numeric(right)


def _cuda_dtw_alignment(
    left: list[list[float]],
    right: list[list[float]],
) -> tuple[float, list[tuple[int, int]]]:
    import torch

    left_length = len(left)
    right_length = len(right)
    dimension_count = max(
        max((len(frame) for frame in left), default=0),
        max((len(frame) for frame in right), default=0),
    )
    device = torch.device("cuda")
    left_tensor = torch.zeros((1, left_length, dimension_count), device=device)
    right_tensor = torch.zeros((1, right_length, dimension_count), device=device)
    _fill_sequence_tensor(left_tensor, 0, left)
    _fill_sequence_tensor(right_tensor, 0, right)

    local_costs = _cuda_pairwise_euclidean_costs(left_tensor, right_tensor)
    matrix, steps = _cuda_accumulate_dtw(local_costs)
    distance = matrix[0, left_length, right_length] / torch.clamp(
        steps[0, left_length, right_length],
        min=1,
    ).to(torch.float32)
    distance_value = _validate_dtw_distance(float(distance.detach().cpu()), left_length, right_length)
    path = _traceback_alignment(
        matrix[0].detach().cpu().tolist(),
        left_length,
        right_length,
    )
    return distance_value, path


def _pair_supports_cuda(
    left: dict[str, Any],
    right: dict[str, Any],
    dtw_configuration: dict[str, Any],
) -> bool:
    if dtw_configuration:
        return False
    return _sequence_is_numeric(left.get("sequence")) and _sequence_is_numeric(right.get("sequence"))


def _sequence_is_numeric(sequence: Any) -> bool:
    if not isinstance(sequence, list) or not sequence:
        return False
    for frame in sequence:
        if not isinstance(frame, list):
            return False
        for value in frame:
            if isinstance(value, bool):
                return False
            try:
                float(value)
            except (TypeError, ValueError):
                return False
    return True


def _resolve_distance_backend(stats: dict[str, Any]) -> str:
    cuda_pairs = int(stats.get("cuda_pair_count", 0) or 0)
    cpu_pairs = int(stats.get("cpu_pair_count", 0) or 0)
    if cuda_pairs and cpu_pairs:
        return "mixed_cuda_cpu"
    if cuda_pairs:
        return "cuda"
    return "cpu"


def _fill_cuda_distance_pairs(
    distances: dict[str, dict[str, float]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    completed_pairs: int,
    total_pairs: int,
    progress_callback: Callable[[int, int], None] | None,
) -> dict[str, Any]:
    import torch

    stats = {
        "cuda_pair_count": 0,
        "cuda_batch_count": 0,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_torch_version": torch.__version__,
    }
    for batch in _iter_cuda_batches(pairs):
        batch_distances = _cuda_batch_dtw_distances(batch)
        for (entry, other), distance in zip(batch, batch_distances, strict=True):
            key = entry["record_key"]
            other_key = other["record_key"]
            distances[key][other_key] = distance
            distances[other_key][key] = distance
        completed_pairs += len(batch)
        stats["cuda_pair_count"] += len(batch)
        stats["cuda_batch_count"] += 1
        if progress_callback is not None:
            progress_callback(completed_pairs, total_pairs)
    return {"completed_pairs": completed_pairs, "stats": stats}


def _iter_cuda_batches(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    batches: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    batch: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sorted_pairs = sorted(pairs, key=_pair_shape_key)
    for pair in sorted_pairs:
        candidate_cells = _batch_padded_cell_count([*batch, pair])
        if batch and candidate_cells > CUDA_DTW_TARGET_CELLS_PER_BATCH:
            batches.append(batch)
            batch = []
        batch.append(pair)
    if batch:
        batches.append(batch)
    return batches


def _pair_shape_key(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int]:
    left, right = pair
    return max(len(left["sequence"]), len(right["sequence"])), min(
        len(left["sequence"]),
        len(right["sequence"]),
    )


def _batch_padded_cell_count(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> int:
    max_left = max(len(left["sequence"]) for left, _right in pairs)
    max_right = max(len(right["sequence"]) for _left, right in pairs)
    return len(pairs) * (max_left + 1) * (max_right + 1)


def _cuda_batch_dtw_distances(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[float]:
    import torch

    device = torch.device("cuda")
    left_lengths = [len(left["sequence"]) for left, _right in pairs]
    right_lengths = [len(right["sequence"]) for _left, right in pairs]
    max_left = max(left_lengths)
    max_right = max(right_lengths)
    dimension_count = max(
        max((len(frame) for frame in entry["sequence"]), default=0)
        for pair in pairs
        for entry in pair
    )
    left_tensor = torch.zeros((len(pairs), max_left, dimension_count), device=device)
    right_tensor = torch.zeros((len(pairs), max_right, dimension_count), device=device)

    for batch_index, (left, right) in enumerate(pairs):
        _fill_sequence_tensor(left_tensor, batch_index, left["sequence"])
        _fill_sequence_tensor(right_tensor, batch_index, right["sequence"])

    local_costs = _cuda_pairwise_euclidean_costs(left_tensor, right_tensor)
    matrix, steps = _cuda_accumulate_dtw(local_costs)
    batch_indices = torch.arange(len(pairs), device=device)
    left_index = torch.tensor(left_lengths, device=device)
    right_index = torch.tensor(right_lengths, device=device)
    distances = matrix[batch_indices, left_index, right_index]
    normalizers = torch.clamp(steps[batch_indices, left_index, right_index], min=1).to(torch.float32)
    normalized = distances / normalizers
    values = normalized.detach().cpu().tolist()
    return [
        _validate_dtw_distance(float(value), left_length, right_length)
        for value, left_length, right_length in zip(values, left_lengths, right_lengths, strict=True)
    ]


def _fill_sequence_tensor(tensor: Any, batch_index: int, sequence: list[list[float]]) -> None:
    import torch

    values = [
        [float(value) for value in frame]
        for frame in sequence
    ]
    if not values:
        return
    width = max(len(frame) for frame in values)
    padded = [
        frame + [0.0] * (width - len(frame))
        for frame in values
    ]
    tensor[batch_index, :len(padded), :width] = torch.tensor(
        padded,
        dtype=torch.float32,
        device=tensor.device,
    )


def _cuda_pairwise_euclidean_costs(left_tensor: Any, right_tensor: Any) -> Any:
    import torch

    left_norm = (left_tensor * left_tensor).sum(dim=2, keepdim=True)
    right_norm = (right_tensor * right_tensor).sum(dim=2).unsqueeze(1)
    cross = torch.bmm(left_tensor, right_tensor.transpose(1, 2))
    return torch.clamp(left_norm + right_norm - (2.0 * cross), min=0.0).sqrt()


def _cuda_accumulate_dtw(local_costs: Any) -> tuple[Any, Any]:
    import torch

    batch_count, left_length, right_length = local_costs.shape
    device = local_costs.device
    matrix = torch.full(
        (batch_count, left_length + 1, right_length + 1),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    steps = torch.zeros(
        (batch_count, left_length + 1, right_length + 1),
        dtype=torch.int32,
        device=device,
    )
    matrix[:, 0, 0] = 0.0
    max_step = torch.iinfo(torch.int32).max

    for diagonal in range(2, left_length + right_length + 1):
        left_start = max(1, diagonal - right_length)
        left_end = min(left_length, diagonal - 1)
        if left_start > left_end:
            continue
        left_indices = torch.arange(left_start, left_end + 1, device=device)
        right_indices = diagonal - left_indices

        candidate_costs = torch.stack(
            (
                matrix[:, left_indices - 1, right_indices],
                matrix[:, left_indices, right_indices - 1],
                matrix[:, left_indices - 1, right_indices - 1],
            ),
            dim=2,
        )
        candidate_steps = torch.stack(
            (
                steps[:, left_indices - 1, right_indices],
                steps[:, left_indices, right_indices - 1],
                steps[:, left_indices - 1, right_indices - 1],
            ),
            dim=2,
        )
        best_cost = candidate_costs.min(dim=2).values
        eligible_steps = torch.where(
            candidate_costs == best_cost.unsqueeze(2),
            candidate_steps,
            torch.full_like(candidate_steps, max_step),
        )
        best_steps = eligible_steps.min(dim=2).values
        finite = torch.isfinite(best_cost)
        matrix[:, left_indices, right_indices] = torch.where(
            finite,
            local_costs[:, left_indices - 1, right_indices - 1] + best_cost,
            torch.full_like(best_cost, float("inf")),
        )
        steps[:, left_indices, right_indices] = torch.where(
            finite,
            best_steps.clamp_max(max_step - 1) + 1,
            torch.zeros_like(best_steps),
        )
    return matrix, steps
