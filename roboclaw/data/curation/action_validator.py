from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import percentile


def validate_action(
    data: dict[str, Any],
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    from .validators import _extract_timestamps, _merge_threshold_overrides
    from .validators import finalize_validator, make_issue

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


def _safe_float(value: Any) -> float | None:
    from .validators import safe_float

    return safe_float(value)


def _extract_numeric_components(value: Any) -> list[float | None]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [_safe_float(value.item())]
        if value.ndim == 1:
            return [_safe_float(item) for item in value.tolist()]
        return []
    if isinstance(value, (list, tuple)):
        return [_safe_float(item) for item in value]
    return [_safe_float(value)]


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
    from .validators import make_issue

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
    from .validators import make_issue

    velocities: list[float] = []
    absolute_values = [
        abs(v)
        for vals in primary_series.values()
        for v in vals
        if v is not None
    ]
    total_value_count = sum(len(values) for values in primary_series.values())
    valid_value_count = sum(
        1
        for values in primary_series.values()
        for value in values
        if value is not None
    )
    nan_like_count = max(total_value_count - valid_value_count, 0)
    for values in primary_series.values():
        limit = min(len(values), len(timestamps))
        for index in range(1, limit):
            cv = values[index]
            pv = values[index - 1]
            if cv is None or pv is None:
                continue
            dt = max(timestamps[index] - timestamps[index - 1], 1e-6)
            velocities.append(abs(cv - pv) / dt)

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
