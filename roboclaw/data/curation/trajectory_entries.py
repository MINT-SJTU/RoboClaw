from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import build_canonical_trajectory
from .dtw import resolve_dtw_configuration
from .features import (
    build_joint_trajectory_payload,
    extract_action_names,
    extract_state_names,
    resolve_task_value,
    resolve_timestamp,
)
from .validators import load_episode_data


def build_prototype_entry(
    dataset_path: Path,
    episode_index: int,
    *,
    quality: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episode_data = data or load_episode_data(dataset_path, episode_index)
    rows = episode_data["rows"]
    canonical = _build_canonical(episode_data)
    return {
        "record_key": str(episode_index),
        "episode_index": episode_index,
        "sequence": canonical.sequence,
        "feature_vector": canonical.feature_vector,
        "canonical_mode": canonical.mode,
        "canonical_groups": canonical.groups,
        "quality": quality or {},
        "task_key": _episode_task_key(episode_data),
        "robot_type": str(episode_data["info"].get("robot_type", "")),
        "time_axis": canonical_time_axis(rows, canonical.sequence),
    }


def build_propagation_entry(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    data = load_episode_data(dataset_path, episode_index)
    rows = data["rows"]
    canonical = _build_canonical(data)
    return {
        "sequence": canonical.sequence,
        "canonical_mode": canonical.mode,
        "canonical_groups": canonical.groups,
        "time_axis": canonical_time_axis(rows, canonical.sequence),
    }


def propagation_dtw_config(
    source_entry: dict[str, Any],
    target_entry: dict[str, Any],
) -> dict[str, Any]:
    cfg = resolve_dtw_configuration(
        left_mode=source_entry.get("canonical_mode"),
        right_mode=target_entry.get("canonical_mode"),
        left_groups=source_entry.get("canonical_groups"),
        right_groups=target_entry.get("canonical_groups"),
    )
    if cfg:
        cfg["window_ratio"] = max(float(cfg.get("window_ratio", 0.0)), 0.20)
    return cfg


def _build_canonical(data: dict[str, Any]):
    joint_traj = build_joint_trajectory_payload(
        data["rows"],
        extract_action_names(data["info"]),
        extract_state_names(data["info"]),
    )
    return build_canonical_trajectory(data["rows"], joint_traj)


def _episode_task_key(data: dict[str, Any]) -> str:
    episode_meta = data.get("episode_meta") or {}
    for key in ("task", "task_label", "instruction"):
        value = episode_meta.get(key)
        if value not in (None, ""):
            return str(value)
    for row in data.get("rows", []):
        value = resolve_task_value(row)
        if value not in (None, ""):
            return str(value)
    return "unknown-task"


def canonical_time_axis(rows: list[dict[str, Any]], sequence: list[list[float]]) -> list[float]:
    if not sequence:
        return []
    timestamps = [
        timestamp
        for row in rows
        if (timestamp := resolve_timestamp(row)) is not None
    ]
    if len(timestamps) < 2:
        return [float(index) for index in range(len(sequence))]
    duration = max(timestamps[-1] - timestamps[0], 0.0)
    if len(sequence) == 1:
        return [0.0]
    return [
        duration * index / (len(sequence) - 1)
        for index in range(len(sequence))
    ]
