"""Serializers — transform internal curation data structures to API response format."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from roboclaw.data.curation.exports import (
    dataset_quality_parquet_path,
    dataset_text_annotations_parquet_path,
    workflow_quality_parquet_path,
)
from roboclaw.data.curation.features import (
    build_joint_trajectory_payload,
    extract_action_names,
    extract_state_names,
    resolve_task_value,
    resolve_timestamp,
)
from roboclaw.data.curation.state import (
    load_annotations,
    load_propagation_results,
)
from roboclaw.data.curation.validators import load_episode_data


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def coerce_int(value: Any) -> int | None:
    """Attempt to coerce *value* to an ``int``, returning ``None`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def episode_time_bounds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Return ``(start, end)`` timestamps from a list of episode rows."""
    timestamps = [
        timestamp
        for row in rows
        if (timestamp := resolve_timestamp(row)) is not None
    ]
    if not timestamps:
        return None, None
    return timestamps[0], timestamps[-1]


def derive_task_value(data: dict[str, Any]) -> str:
    """Extract a human-readable task label from episode data."""
    episode_meta = data.get("episode_meta") or {}
    for key in ("task", "task_label", "instruction"):
        value = episode_meta.get(key)
        if value not in (None, ""):
            return str(value)

    for row in data.get("rows", []):
        value = resolve_task_value(row)
        if value not in (None, ""):
            return str(value)

    return ""


# ---------------------------------------------------------------------------
# Result serializers
# ---------------------------------------------------------------------------


def serialize_quality_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize quality-validation results for the API response."""
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "overall_score": 0.0,
            "selected_validators": [],
            "episodes": [],
        }

    return {
        **results,
        "overall_score": float(results.get("overall_score", 0.0) or 0.0),
    }


def serialize_prototype_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize prototype-discovery results for the API response."""
    if not results:
        return {
            "candidate_count": 0,
            "entry_count": 0,
            "cluster_count": 0,
            "anchor_record_keys": [],
            "clusters": [],
        }

    refinement = results.get("refinement", {})
    clustering = results.get("clustering", {})
    raw_clusters = refinement.get("clusters") or clustering.get("clusters") or []
    clusters: list[dict[str, Any]] = []

    for index, cluster in enumerate(raw_clusters):
        members = []
        for member in cluster.get("members", []):
            members.append({
                **member,
                "episode_index": coerce_int(member.get("record_key")),
            })

        clusters.append({
            "cluster_index": cluster.get("cluster_index", index),
            "prototype_record_key": str(
                cluster.get("prototype_record_key")
                or cluster.get("anchor_record_key")
                or ""
            ),
            "anchor_record_key": str(
                cluster.get("anchor_record_key")
                or cluster.get("prototype_record_key")
                or ""
            ),
            "member_count": int(cluster.get("member_count", len(members)) or len(members)),
            "average_distance": cluster.get("average_distance"),
            "anchor_distance_to_barycenter": cluster.get("anchor_distance_to_barycenter"),
            "members": members,
        })

    anchor_record_keys = refinement.get("anchor_record_keys") or [
        cluster["anchor_record_key"]
        for cluster in clusters
        if cluster["anchor_record_key"]
    ]

    return {
        "candidate_count": int(results.get("candidate_count", 0) or 0),
        "entry_count": int(results.get("entry_count", 0) or 0),
        "cluster_count": int(results.get("cluster_count", len(clusters)) or len(clusters)),
        "anchor_record_keys": anchor_record_keys,
        "clusters": clusters,
    }


def serialize_propagation_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize semantic-propagation results for the API response."""
    if not results:
        return {
            "source_episode_index": None,
            "target_count": 0,
            "propagated": [],
        }
    return results


# ---------------------------------------------------------------------------
# Workspace payload builder
# ---------------------------------------------------------------------------


def build_workspace_payload(
    dataset: str,
    dataset_path: Path,
    episode_index: int,
) -> dict[str, Any]:
    """Assemble the full annotation-workspace payload for a single episode."""
    data = load_episode_data(dataset_path, episode_index)
    info = data.get("info", {})
    rows = data.get("rows", [])
    start_timestamp, end_timestamp = episode_time_bounds(rows)
    duration_s = 0.0
    if start_timestamp is not None and end_timestamp is not None:
        duration_s = max(end_timestamp - start_timestamp, 0.0)

    action_names = extract_action_names(info)
    state_names = extract_state_names(info)
    joint_trajectory = build_joint_trajectory_payload(rows, action_names, state_names)
    relative_videos = [
        video_path.relative_to(dataset_path).as_posix()
        for video_path in data.get("video_files", [])
    ]
    task_value = derive_task_value(data)
    saved_annotations = load_annotations(dataset_path, episode_index) or {
        "episode_index": episode_index,
        "task_context": {},
        "annotations": [],
        "version_number": 0,
    }
    propagation = load_propagation_results(dataset_path)
    latest_propagation = None
    if propagation and propagation.get("source_episode_index") == episode_index:
        latest_propagation = propagation

    return {
        "episode_index": episode_index,
        "summary": {
            "episode_index": episode_index,
            "record_key": str(episode_index),
            "task_value": task_value,
            "task_label": task_value,
            "fps": info.get("fps", 0),
            "robot_type": info.get("robot_type", ""),
            "row_count": len(rows),
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "duration_s": duration_s,
            "video_count": len(relative_videos),
        },
        "videos": [
            {
                "path": relative_path,
                "url": f"/api/curation/video/{quote(relative_path, safe='/')}?dataset={quote(dataset, safe='')}",
                "stream": Path(relative_path).stem,
                "from_timestamp": 0,
                "to_timestamp": duration_s if duration_s > 0 else None,
            }
            for relative_path in relative_videos
        ],
        "joint_trajectory": joint_trajectory,
        "annotations": saved_annotations,
        "latest_propagation": latest_propagation,
    }
