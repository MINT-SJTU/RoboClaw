from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .dtw import dtw_alignment, vector_distance
from .features import mean, percentile
from .validators import finalize_validator, make_issue

TRAJECTORY_DTW_VALIDATOR = "trajectory_dtw"
REFERENCE_TUBE_BACKEND = "windowed_exact_dtw"


@dataclass(frozen=True)
class ReferenceTube:
    mean_sequence: list[list[float]]
    aligned_references: list[list[list[float]]]
    velocity_references: list[list[list[float]]]
    distance_threshold: float
    velocity_threshold: float
    anchor_record_key: str
    backend: str


class ReferenceTubeBuilder:
    def __init__(
        self,
        *,
        thresholds: dict[str, float],
        dtw_config: dict[str, Any] | None = None,
    ) -> None:
        self._thresholds = thresholds
        self._dtw_config = dtw_config or {}

    def build(self, entries: list[dict[str, Any]]) -> ReferenceTube | None:
        usable_entries = [entry for entry in entries if _has_sequence(entry)]
        if not usable_entries:
            return None

        anchor = _median_length_anchor(usable_entries)
        anchor_sequence = _entry_sequence(anchor)
        aligned_references: list[list[list[float]]] = []
        aligned_velocity_references: list[list[list[float]]] = []
        for entry in usable_entries:
            aligned, aligned_velocity = _align_sequence_and_velocity_to_anchor(
                anchor_sequence,
                _entry_sequence(entry),
                self._dtw_config,
            )
            if aligned:
                aligned_references.append(aligned)
                aligned_velocity_references.append(aligned_velocity)

        if not aligned_references:
            return None

        mean_sequence = [
            _average_frame([aligned[step_index] for aligned in aligned_references])
            for step_index in range(len(anchor_sequence))
        ]
        distance_threshold = _distance_threshold(
            aligned_references,
            mean_sequence,
            floor=self._thresholds["trajectory_dtw_position_floor"],
            quantile=self._thresholds["trajectory_dtw_distance_quantile"],
        )
        velocity_threshold = _velocity_threshold(
            aligned_velocity_references,
            floor=self._thresholds["trajectory_dtw_velocity_floor"],
            quantile=self._thresholds["trajectory_dtw_velocity_quantile"],
        )

        return ReferenceTube(
            mean_sequence=mean_sequence,
            aligned_references=aligned_references,
            velocity_references=aligned_velocity_references,
            distance_threshold=distance_threshold,
            velocity_threshold=velocity_threshold,
            anchor_record_key=str(anchor.get("record_key", "")),
            backend=REFERENCE_TUBE_BACKEND,
        )


class ReferenceTubeEvaluator:
    def __init__(
        self,
        tube: ReferenceTube,
        *,
        thresholds: dict[str, float],
        dtw_config: dict[str, Any] | None = None,
    ) -> None:
        self._tube = tube
        self._thresholds = thresholds
        self._dtw_config = dtw_config or {}

    def evaluate(self, entry: dict[str, Any]) -> dict[str, Any]:
        sequence = _entry_sequence(entry)
        if not sequence:
            return _single_issue_result(
                check_name="trajectory_available",
                passed=False,
                message="Trajectory sequence unavailable",
                level="major",
                value={"reason": "empty_sequence"},
            )

        distance, path = dtw_alignment(
            self._tube.mean_sequence,
            sequence,
            **self._dtw_config,
        )
        if not math.isfinite(distance) or not path:
            return _single_issue_result(
                check_name="trajectory_comparable",
                passed=False,
                message="Trajectory is not comparable with reference tube",
                level="major",
                value={"distance": None if math.isinf(distance) else distance},
            )

        time_axis = _entry_time_axis(entry, len(sequence))
        anomalies = []
        anomalies.extend(self._deviation_points(sequence, path))
        anomalies.extend(self._hesitation_points(sequence))
        issues = self._segment_issues(anomalies, time_axis)
        issues.extend(self._stall_issues(path, time_axis))

        if not issues:
            issues.append(make_issue(
                operator_name=TRAJECTORY_DTW_VALIDATOR,
                check_name="reference_tube_alignment",
                passed=True,
                message="Trajectory matches reference tube",
                level="info",
                value=self._details(entry),
            ))

        return finalize_validator(
            TRAJECTORY_DTW_VALIDATOR,
            issues,
            details=self._details(entry),
        )

    def _deviation_points(
        self,
        sequence: list[list[float]],
        path: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        threshold = (
            self._tube.distance_threshold
            * self._thresholds["trajectory_dtw_deviation_multiplier"]
        )
        points: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for tube_index, candidate_index in path:
            if (tube_index, candidate_index) in seen_pairs:
                continue
            seen_pairs.add((tube_index, candidate_index))
            if tube_index >= len(self._tube.mean_sequence) or candidate_index >= len(sequence):
                continue
            candidate = sequence[candidate_index]
            nearest = min(
                vector_distance(reference[tube_index], candidate)
                for reference in self._tube.aligned_references
            )
            if nearest <= threshold:
                continue
            component_index, component_delta = _largest_component_delta(
                candidate,
                self._tube.mean_sequence[tube_index],
            )
            points.append({
                "type": "Deviation",
                "frame_index": candidate_index,
                "distance": nearest,
                "threshold": threshold,
                "component_index": component_index,
                "component_delta": component_delta,
            })
        return points

    def _hesitation_points(self, sequence: list[list[float]]) -> list[dict[str, Any]]:
        threshold = (
            self._tube.velocity_threshold
            * self._thresholds["trajectory_dtw_hesitation_multiplier"]
        )
        points: list[dict[str, Any]] = []
        velocities = _velocity_sequence(sequence)
        for frame_index, velocity in enumerate(velocities):
            if frame_index == 0:
                continue
            variance = _variance(velocity)
            if variance <= threshold:
                continue
            points.append({
                "type": "Hesitate",
                "frame_index": frame_index,
                "velocity_variance": variance,
                "threshold": threshold,
            })
        return points

    def _segment_issues(
        self,
        anomalies: list[dict[str, Any]],
        time_axis: list[float],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for anomaly_type in ("Deviation", "Hesitate"):
            frames = sorted(
                int(item["frame_index"])
                for item in anomalies
                if item.get("type") == anomaly_type
            )
            for group in _contiguous_groups(frames):
                if not _segment_is_long_enough(
                    group,
                    time_axis,
                    self._thresholds["trajectory_dtw_min_segment_s"],
                ):
                    continue
                samples = [
                    item for item in anomalies
                    if item.get("type") == anomaly_type and int(item["frame_index"]) in group
                ]
                evidence = max(
                    samples,
                    key=lambda item: float(item.get("distance") or item.get("velocity_variance") or 0.0),
                )
                issues.append(make_issue(
                    operator_name=TRAJECTORY_DTW_VALIDATOR,
                    check_name=f"dtw_{anomaly_type.lower()}",
                    passed=False,
                    message=_anomaly_message(anomaly_type, group, time_axis, evidence),
                    level="major",
                    value={
                        "anomaly_type": anomaly_type,
                        "start_frame": group[0],
                        "end_frame": group[-1],
                        "start_time_s": _time_at(time_axis, group[0]),
                        "end_time_s": _time_at(time_axis, group[-1]),
                        "evidence": evidence,
                    },
                ))
        return issues

    def _stall_issues(
        self,
        path: list[tuple[int, int]],
        time_axis: list[float],
    ) -> list[dict[str, Any]]:
        buckets: dict[int, list[int]] = {}
        for tube_index, candidate_index in path:
            buckets.setdefault(tube_index, []).append(candidate_index)

        issues: list[dict[str, Any]] = []
        frame_threshold = int(self._thresholds["trajectory_dtw_stall_frame_threshold"])
        for tube_index, frame_indices in sorted(buckets.items()):
            unique_frames = sorted(set(frame_indices))
            if len(unique_frames) <= frame_threshold:
                continue
            duration = _time_at(time_axis, unique_frames[-1]) - _time_at(time_axis, unique_frames[0])
            issues.append(make_issue(
                operator_name=TRAJECTORY_DTW_VALIDATOR,
                check_name="dtw_stall",
                passed=False,
                message=(
                    "Stall near reference frame "
                    f"{tube_index}: {len(unique_frames)} candidate frames over {duration:.3f}s"
                ),
                level="major",
                value={
                    "anomaly_type": "Stall",
                    "tube_frame": tube_index,
                    "matched_frame_count": len(unique_frames),
                    "start_frame": unique_frames[0],
                    "end_frame": unique_frames[-1],
                    "start_time_s": _time_at(time_axis, unique_frames[0]),
                    "end_time_s": _time_at(time_axis, unique_frames[-1]),
                    "duration_s": duration,
                    "frame_threshold": frame_threshold,
                },
            ))
        return issues

    def _details(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": self._tube.backend,
            "anchor_record_key": self._tube.anchor_record_key,
            "record_key": str(entry.get("record_key", "")),
            "distance_threshold": self._tube.distance_threshold,
            "velocity_threshold": self._tube.velocity_threshold,
            "reference_count": len(self._tube.aligned_references),
        }


def skipped_result(check_name: str, message: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
    return _single_issue_result(
        check_name=check_name,
        passed=True,
        message=message,
        level="info",
        value=value or {},
    )


def unavailable_result(message: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
    return _single_issue_result(
        check_name="trajectory_available",
        passed=False,
        message=message,
        level="major",
        value=value or {},
    )


def _single_issue_result(
    *,
    check_name: str,
    passed: bool,
    message: str,
    level: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    issue = make_issue(
        operator_name=TRAJECTORY_DTW_VALIDATOR,
        check_name=check_name,
        passed=passed,
        message=message,
        level=level,
        value=value,
    )
    return finalize_validator(
        TRAJECTORY_DTW_VALIDATOR,
        [issue],
        details=value,
    )


def _has_sequence(entry: dict[str, Any]) -> bool:
    return bool(_entry_sequence(entry))


def _entry_sequence(entry: dict[str, Any]) -> list[list[float]]:
    sequence = entry.get("sequence")
    if not isinstance(sequence, list):
        return []
    normalized: list[list[float]] = []
    for frame in sequence:
        if not isinstance(frame, list) or not frame:
            continue
        values: list[float] = []
        for value in frame:
            try:
                number = float(value)
            except (TypeError, ValueError):
                values = []
                break
            if not math.isfinite(number):
                values = []
                break
            values.append(number)
        if values:
            normalized.append(values)
    return normalized


def _entry_time_axis(entry: dict[str, Any], sequence_length: int) -> list[float]:
    raw_time_axis = entry.get("time_axis")
    if isinstance(raw_time_axis, list) and len(raw_time_axis) == sequence_length:
        values: list[float] = []
        for index, value in enumerate(raw_time_axis):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float(index)
            values.append(number if math.isfinite(number) else float(index))
        return values
    return [float(index) for index in range(sequence_length)]


def _median_length_anchor(entries: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(_entry_sequence(entry)) for entry in entries]
    median_length = statistics.median(lengths)
    return min(entries, key=lambda entry: abs(len(_entry_sequence(entry)) - median_length))


def _align_sequence_and_velocity_to_anchor(
    anchor_sequence: list[list[float]],
    sequence: list[list[float]],
    dtw_config: dict[str, Any],
) -> tuple[list[list[float]], list[list[float]]]:
    distance, path = dtw_alignment(anchor_sequence, sequence, **dtw_config)
    if not math.isfinite(distance) or not path:
        return [], []

    aligned = _align_path_frames(anchor_sequence, sequence, path)
    aligned_velocity = _align_path_frames(
        _velocity_sequence(anchor_sequence),
        _velocity_sequence(sequence),
        path,
    )
    return aligned, aligned_velocity


def _align_path_frames(
    anchor_fallback: list[list[float]],
    sequence_frames: list[list[float]],
    path: list[tuple[int, int]],
) -> list[list[float]]:
    buckets: list[list[list[float]]] = [[] for _ in anchor_fallback]
    for anchor_index, sequence_index in path:
        if 0 <= anchor_index < len(anchor_fallback) and 0 <= sequence_index < len(sequence_frames):
            buckets[anchor_index].append(sequence_frames[sequence_index])

    aligned: list[list[float]] = []
    previous: list[float] | None = None
    for index, frames in enumerate(buckets):
        if frames:
            current = _average_frame(frames)
        elif previous is not None:
            current = previous[:]
        else:
            current = anchor_fallback[index][:]
        aligned.append(current)
        previous = current
    return aligned


def _average_frame(frames: list[list[float]]) -> list[float]:
    if not frames:
        return []
    dimension_count = max(len(frame) for frame in frames)
    averaged: list[float] = []
    for dim_index in range(dimension_count):
        values = [
            frame[dim_index] if dim_index < len(frame) else 0.0
            for frame in frames
        ]
        averaged.append(mean(values))
    return averaged


def _velocity_sequence(sequence: list[list[float]]) -> list[list[float]]:
    if not sequence:
        return []
    velocities = [[0.0 for _ in range(len(sequence[0]))]]
    for index in range(1, len(sequence)):
        current = sequence[index]
        previous = sequence[index - 1]
        dimension_count = max(len(current), len(previous))
        velocities.append([
            (current[dim] if dim < len(current) else 0.0)
            - (previous[dim] if dim < len(previous) else 0.0)
            for dim in range(dimension_count)
        ])
    return velocities


def _distance_threshold(
    aligned_references: list[list[list[float]]],
    mean_sequence: list[list[float]],
    *,
    floor: float,
    quantile: float,
) -> float:
    distances = [
        vector_distance(frame, mean_sequence[index])
        for reference in aligned_references
        for index, frame in enumerate(reference)
        if index < len(mean_sequence)
    ]
    return max(percentile(distances, quantile), float(floor))


def _velocity_threshold(
    velocity_references: list[list[list[float]]],
    *,
    floor: float,
    quantile: float,
) -> float:
    variances = [
        _variance(velocity)
        for reference in velocity_references
        for velocity in reference
    ]
    return max(percentile(variances, quantile), float(floor))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def _largest_component_delta(left: list[float], right: list[float]) -> tuple[int, float]:
    dimension_count = max(len(left), len(right))
    if dimension_count <= 0:
        return 0, 0.0
    deltas = [
        abs((left[index] if index < len(left) else 0.0) - (right[index] if index < len(right) else 0.0))
        for index in range(dimension_count)
    ]
    component_index = max(range(dimension_count), key=lambda index: deltas[index])
    return component_index, deltas[component_index]


def _contiguous_groups(frames: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for frame in frames:
        if not groups or frame != groups[-1][-1] + 1:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return groups


def _segment_is_long_enough(
    group: list[int],
    time_axis: list[float],
    minimum_duration: float,
) -> bool:
    if not group:
        return False
    if len(group) > 1:
        return (_time_at(time_axis, group[-1]) - _time_at(time_axis, group[0])) >= minimum_duration
    return minimum_duration <= 0.0


def _time_at(time_axis: list[float], index: int) -> float:
    if not time_axis:
        return float(index)
    bounded = min(max(index, 0), len(time_axis) - 1)
    return float(time_axis[bounded])


def _anomaly_message(
    anomaly_type: str,
    group: list[int],
    time_axis: list[float],
    evidence: dict[str, Any],
) -> str:
    start_time = _time_at(time_axis, group[0])
    end_time = _time_at(time_axis, group[-1])
    if anomaly_type == "Deviation":
        return (
            f"Deviation from reference tube over {start_time:.3f}-{end_time:.3f}s; "
            f"component_{evidence.get('component_index')} delta "
            f"{float(evidence.get('component_delta', 0.0)):.3f}"
        )
    return (
        f"Hesitate near {start_time:.3f}-{end_time:.3f}s; "
        f"velocity variance {float(evidence.get('velocity_variance', 0.0)):.3f}"
    )
