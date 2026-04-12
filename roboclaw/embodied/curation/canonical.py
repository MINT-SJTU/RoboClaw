from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .features import (
    build_episode_feature_vector,
    build_episode_sequence,
    percentile,
    resolve_action_vector,
    resolve_state_vector,
    resolve_timestamp,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_GROUP_SLICES = {
    "eef_pos": list(range(0, 3)),
    "eef_rot6d": list(range(3, 9)),
    "gripper": [9],
    "delta_pos": list(range(10, 13)),
    "delta_rot6d": list(range(13, 19)),
    "delta_gripper": [19],
}

ALOHA_ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


@dataclass(frozen=True)
class CanonicalTrajectory:
    mode: str
    sequence: list[list[float]]
    feature_vector: dict[str, Any]
    groups: dict[str, list[int]]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    center = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[center]
    return (ordered[center - 1] + ordered[center]) / 2.0


def _extract_joint_names(joint_trajectory: dict[str, Any] | None) -> list[str]:
    joint_entries = (joint_trajectory or {}).get("joint_trajectories", [])
    names: list[str] = []
    for entry in joint_entries:
        name = entry.get("joint_name") or entry.get("state_name") or entry.get("action_name")
        if name:
            names.append(str(name))
    return names


def _group_joint_indices(joint_names: list[str], dim_count: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "left_arm": [],
        "left_gripper": [],
        "right_arm": [],
        "right_gripper": [],
        "other": [],
    }

    for index, name in enumerate(joint_names[:dim_count]):
        lowered = name.lower()
        if lowered.startswith("left_"):
            _classify_left_right(lowered, index, groups, "left")
        elif lowered.startswith("right_"):
            _classify_left_right(lowered, index, groups, "right")
        else:
            groups["other"].append(index)

    return {key: value for key, value in groups.items() if value}


def _classify_left_right(
    lowered: str,
    index: int,
    groups: dict[str, list[int]],
    side: str,
) -> None:
    if any(token in lowered for token in ("gripper", "claw", "finger")):
        groups[f"{side}_gripper"].append(index)
    else:
        groups[f"{side}_arm"].append(index)


def _infer_joint_dim(rows: list[dict[str, Any]]) -> int:
    for row in rows:
        state = resolve_state_vector(row)
        action = resolve_action_vector(row)
        source = state or action
        if source:
            return len(source)
    return 0


def _coerce_numeric_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None

    vector: list[float] = []
    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return None
    return vector


# ---------------------------------------------------------------------------
# Pose / gripper extraction
# ---------------------------------------------------------------------------


def _extract_row_pose(row: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    for key in ("observation.state.cartesian_position", "action.cartesian_position"):
        vector = _coerce_numeric_vector(row.get(key))
        if vector is None or len(vector) < 6:
            continue
        position = vector[:3]
        rotation = _rotation_matrix_to_6d(_rotation_from_euler_xyz(vector[3:6]))
        return position, rotation
    return None


def _extract_row_gripper(row: dict[str, Any]) -> float | None:
    for key in ("observation.state.gripper_position", "action.gripper_position"):
        vector = _coerce_numeric_vector(row.get(key))
        if vector:
            return float(vector[0])

    state = resolve_state_vector(row)
    action = resolve_action_vector(row)
    source = state or action
    if source:
        try:
            return float(source[-1])
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _rotation_from_euler_xyz(euler_xyz: list[float]) -> list[list[float]]:
    roll, pitch, yaw = euler_xyz[:3]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_matrix_to_6d(matrix: list[list[float]]) -> list[float]:
    return [
        matrix[0][0], matrix[1][0], matrix[2][0],
        matrix[0][1], matrix[1][1], matrix[2][1],
    ]


# ---------------------------------------------------------------------------
# Aloha vx300s bridge detection
# ---------------------------------------------------------------------------


def _looks_like_aloha_vx300s_joint_names(joint_names: list[str]) -> bool:
    lowered_names = [name.lower() for name in joint_names]
    for arm_label in ("left", "right"):
        required = {f"{arm_label}_{token}" for token in ALOHA_ARM_JOINT_ORDER}
        required.add(f"{arm_label}_gripper")
        if required.issubset(lowered_names):
            return True
    return False


def _detect_aloha_vx300s_bridge(
    rows: list[dict[str, Any]],
    joint_trajectory: dict[str, Any] | None,
) -> dict[str, Any] | None:
    joint_names = _extract_joint_names(joint_trajectory)
    source_dim = len(joint_names) or _infer_joint_dim(rows) or 0
    looks_like_aloha = _looks_like_aloha_vx300s_joint_names(joint_names) or source_dim == 14
    if not looks_like_aloha:
        return None

    pose_source = None
    gripper_source = None
    for row in rows:
        pose_source, gripper_source = _probe_bridge_sources(row, pose_source, gripper_source)
        if pose_source and gripper_source:
            break

    return {
        "bridge_schema": "aloha_vx300s",
        "source_joint_dim": source_dim,
        "joint_names": joint_names[:source_dim],
        "bridge_pose_source": pose_source,
        "bridge_gripper_source": gripper_source,
    }


def _probe_bridge_sources(
    row: dict[str, Any],
    pose_source: str | None,
    gripper_source: str | None,
) -> tuple[str | None, str | None]:
    if pose_source is None:
        for candidate in ("observation.state.cartesian_position", "action.cartesian_position"):
            vector = _coerce_numeric_vector(row.get(candidate))
            if vector is not None and len(vector) >= 6:
                pose_source = candidate
                break
    if gripper_source is None:
        for candidate in ("observation.state.gripper_position", "action.gripper_position"):
            vector = _coerce_numeric_vector(row.get(candidate))
            if vector:
                gripper_source = candidate
                break
    return pose_source, gripper_source


# ---------------------------------------------------------------------------
# Resampling and normalization
# ---------------------------------------------------------------------------


def _linear_sample(
    time_axis: list[float],
    values: list[list[float]],
    target_time: float,
) -> list[float]:
    if target_time <= time_axis[0]:
        return values[0][:]
    if target_time >= time_axis[-1]:
        return values[-1][:]

    for index in range(1, len(time_axis)):
        left_time = time_axis[index - 1]
        right_time = time_axis[index]
        if right_time < target_time:
            continue
        if right_time <= left_time:
            return values[index][:]
        weight = (target_time - left_time) / (right_time - left_time)
        return [
            values[index - 1][dim] + (values[index][dim] - values[index - 1][dim]) * weight
            for dim in range(len(values[index]))
        ]
    return values[-1][:]


def _resample_cartesian_rows(
    timestamps: list[float],
    rows: list[list[float]],
    *,
    max_points: int,
) -> list[list[float]]:
    if len(rows) <= max_points or max_points <= 1:
        return [row[:] for row in rows]

    duration = max(timestamps[-1] - timestamps[0], 0.0)
    if duration <= 0:
        return [row[:] for row in rows[:max_points]]

    sampled: list[list[float]] = []
    for sample_index in range(max_points):
        ratio = sample_index / max(max_points - 1, 1)
        target_time = timestamps[0] + duration * ratio
        sampled.append(_linear_sample(timestamps, rows, target_time))
    return sampled


def _robust_normalize_features(
    rows: list[list[float]],
) -> tuple[list[list[float]], dict[str, Any]]:
    if not rows:
        return [], {"median": [], "iqr": []}

    dimension_count = len(rows[0])
    normalized_columns: list[list[float]] = []
    medians: list[float] = []
    iqrs: list[float] = []

    for index in range(dimension_count):
        column = [float(row[index]) for row in rows]
        median = _median(column)
        q75 = percentile(column, 0.75)
        q25 = percentile(column, 0.25)
        iqr = q75 - q25
        safe_iqr = iqr if abs(iqr) >= 1e-8 else 1.0
        medians.append(median)
        iqrs.append(safe_iqr)
        normalized_columns.append([(value - median) / safe_iqr for value in column])

    normalized_rows: list[list[float]] = []
    for row_index in range(len(rows)):
        normalized_rows.append([
            normalized_columns[dim][row_index] for dim in range(dimension_count)
        ])

    return normalized_rows, {"median": medians, "iqr": iqrs}


# ---------------------------------------------------------------------------
# Cartesian feature building
# ---------------------------------------------------------------------------


def build_cartesian_feature_rows(rows: list[dict[str, Any]]) -> list[list[float]]:
    pose_rows: list[list[float]] = []
    timestamps: list[float] = []
    for fallback_index, row in enumerate(rows):
        pose = _extract_row_pose(row)
        gripper = _extract_row_gripper(row)
        if pose is None or gripper is None:
            continue
        position, rotation = pose
        pose_rows.append([*position, *rotation, gripper])
        timestamp = resolve_timestamp(row)
        timestamps.append(float(timestamp) if timestamp is not None else float(fallback_index))

    if len(pose_rows) < 2:
        return []

    resampled = _resample_cartesian_rows(timestamps, pose_rows, max_points=80)
    feature_rows: list[list[float]] = []
    previous = [0.0] * 10
    for index, row in enumerate(resampled):
        deltas = [0.0] * 10 if index == 0 else [row[dim] - previous[dim] for dim in range(10)]
        feature_rows.append(row + deltas)
        previous = row
    return feature_rows


# ---------------------------------------------------------------------------
# Public trajectory builders
# ---------------------------------------------------------------------------


def build_cartesian_canonical_trajectory(
    rows: list[dict[str, Any]],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> CanonicalTrajectory | None:
    feature_rows = build_cartesian_feature_rows(rows)
    if len(feature_rows) < 2:
        return None

    normalized_rows, normalization = _robust_normalize_features(feature_rows)
    feature_vector = {
        "vector": [_median([row[index] for row in feature_rows]) for index in range(20)],
        "feature_dim": 20,
        "normalization": normalization,
    }

    return CanonicalTrajectory(
        mode="cartesian_20d",
        sequence=normalized_rows,
        feature_vector=feature_vector,
        groups=CANONICAL_GROUP_SLICES,
        metadata={
            **(extra_metadata or {}),
            "feature_dim": 20,
            "group_keys": list(CANONICAL_GROUP_SLICES.keys()),
            "source": "cartesian_position+gripper",
        },
    )


def build_joint_canonical_trajectory(
    rows: list[dict[str, Any]],
    joint_trajectory: dict[str, Any] | None,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> CanonicalTrajectory:
    joint_names = _extract_joint_names(joint_trajectory)
    source_dim = len(joint_names) or _infer_joint_dim(rows) or 0
    sequence = build_episode_sequence(rows)
    feature_vector = build_episode_feature_vector(joint_trajectory)
    feature_dim = len(sequence[0]) if sequence else 0
    groups = _group_joint_indices(joint_names, feature_dim)

    return CanonicalTrajectory(
        mode="joint_canonical",
        sequence=sequence,
        feature_vector=feature_vector,
        groups=groups,
        metadata={
            **(extra_metadata or {}),
            "feature_dim": feature_dim,
            "source_joint_dim": source_dim,
            "joint_names": joint_names[:feature_dim],
            "group_keys": list(groups.keys()),
            "source": "joint_state_action",
        },
    )


def build_canonical_trajectory(
    rows: list[dict[str, Any]],
    joint_trajectory: dict[str, Any] | None,
) -> CanonicalTrajectory:
    aloha_bridge = _detect_aloha_vx300s_bridge(rows, joint_trajectory)
    cartesian_metadata: dict[str, Any] | None = None
    if aloha_bridge is not None:
        cartesian_metadata = {
            **aloha_bridge,
            "bridge_status": (
                "cartesian_pose_source"
                if aloha_bridge.get("bridge_pose_source")
                else "joint_only_fallback"
            ),
        }

    cartesian = build_cartesian_canonical_trajectory(rows, extra_metadata=cartesian_metadata)
    if cartesian is not None:
        return cartesian

    if aloha_bridge is not None:
        return build_joint_canonical_trajectory(
            rows,
            joint_trajectory,
            extra_metadata={
                **aloha_bridge,
                "bridge_status": "joint_only_fallback",
                "bridge_warning": "Aloha/vx300s episode has no cartesian pose source; using joint_canonical fallback.",
            },
        )

    return build_joint_canonical_trajectory(rows, joint_trajectory)
