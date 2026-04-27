"""Serializers — transform internal curation data structures to API response format."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
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
    load_quality_results,
)
from roboclaw.data.curation.validators import load_episode_data, safe_float


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


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    """Return LeRobot feature keys that point to video streams."""
    features = info.get("features", {})
    keys: list[str] = []
    for name, config in features.items():
        if isinstance(config, dict) and config.get("dtype") == "video":
            keys.append(str(name))
    return keys


def video_key_from_relative_path(relative_path: str, info: dict[str, Any]) -> str | None:
    """Resolve the LeRobot video feature key represented by a repo-relative path."""
    normalized_path = relative_path.replace("\\", "/")
    for video_key in sorted(video_feature_keys(info), key=len, reverse=True):
        if normalized_path.startswith(f"videos/{video_key}/"):
            return video_key

    parts = PurePosixPath(normalized_path).parts
    if len(parts) >= 2 and parts[0] == "videos":
        candidate = parts[1]
        if not candidate.startswith(("chunk-", "episode_")):
            return candidate
    return None


def stream_name_from_video_key(video_key: str | None, relative_path: str) -> str:
    """Build a concise stream label from a video key or legacy video path."""
    if video_key:
        if video_key.startswith("observation.images."):
            return video_key.split("observation.images.", 1)[1]
        if "." in video_key:
            return video_key.rsplit(".", 1)[-1]
        return video_key
    return Path(relative_path).stem


def video_clip_bounds(
    episode_meta: dict[str, Any],
    video_key: str | None,
    duration_s: float,
) -> tuple[float, float | None]:
    """Return absolute video clip bounds for the episode's stream."""
    from_timestamp = None
    to_timestamp = None
    if video_key:
        prefix = f"videos/{video_key}/"
        from_timestamp = safe_float(episode_meta.get(f"{prefix}from_timestamp"))
        to_timestamp = safe_float(episode_meta.get(f"{prefix}to_timestamp"))

    if from_timestamp is None:
        from_timestamp = safe_float(episode_meta.get("video_from_timestamp"))
    if to_timestamp is None:
        to_timestamp = safe_float(episode_meta.get("video_to_timestamp"))

    clip_start = from_timestamp if from_timestamp is not None else 0.0
    clip_end = to_timestamp
    if clip_end is None and duration_s > 0:
        clip_end = clip_start + duration_s

    return clip_start, clip_end


def serialize_workspace_video(
    dataset: str,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    relative_path: str,
    duration_s: float,
) -> dict[str, Any]:
    """Serialize one episode video with its absolute shared-video clip window."""
    video_key = video_key_from_relative_path(relative_path, info)
    clip_start, clip_end = video_clip_bounds(episode_meta, video_key, duration_s)
    return {
        "path": relative_path,
        "url": f"/api/curation/video/{quote(relative_path, safe='/')}?dataset={quote(dataset, safe='')}",
        "stream": stream_name_from_video_key(video_key, relative_path),
        "from_timestamp": clip_start,
        "to_timestamp": clip_end,
    }


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
        "quality_filter_mode": str(results.get("quality_filter_mode", "passed") or "passed"),
        "selected_episode_indices": [
            coerce_int(value)
            for value in results.get("selected_episode_indices", [])
            if coerce_int(value) is not None
        ],
        "clusters": clusters,
    }


def serialize_propagation_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize semantic-propagation results for the API response."""
    if not results:
        return {
            "source_episode_index": None,
            "source_episode_indices": [],
            "target_count": 0,
            "propagated": [],
        }
    payload = dict(results)
    source_episode_indices = [
        index
        for value in payload.get("source_episode_indices", [])
        if (index := coerce_int(value)) is not None
    ]
    source_episode_index = coerce_int(payload.get("source_episode_index"))
    if source_episode_index is not None and source_episode_index not in source_episode_indices:
        source_episode_indices.append(source_episode_index)
    payload["source_episode_indices"] = sorted(source_episode_indices)
    return payload


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
    episode_meta = data.get("episode_meta") or {}
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
    quality = load_quality_results(dataset_path) or {}
    quality_entry = next(
        (
            episode
            for episode in quality.get("episodes", [])
            if coerce_int(episode.get("episode_index")) == episode_index
        ),
        None,
    )
    failed_validators = []
    quality_tags = []
    if isinstance(quality_entry, dict):
        validators = quality_entry.get("validators", {}) or {}
        failed_validators = [
            str(name)
            for name, validator in validators.items()
            if isinstance(validator, dict) and not validator.get("passed", False)
        ]
        quality_tags = ["quality-pass" if quality_entry.get("passed") else "quality-risk"]

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
            "quality_status": (
                "passed"
                if quality_entry and quality_entry.get("passed")
                else "failed"
                if quality_entry
                else "unvalidated"
            ),
            "quality_score": float(quality_entry.get("score", 0.0) or 0.0)
            if isinstance(quality_entry, dict)
            else None,
        },
        "videos": [
            serialize_workspace_video(
                dataset,
                info,
                episode_meta,
                relative_path,
                duration_s,
            )
            for relative_path in relative_videos
        ],
        "joint_trajectory": joint_trajectory,
        "annotations": saved_annotations,
        "latest_propagation": latest_propagation,
        "quality": {
            "validated": bool(quality_entry),
            "passed": bool(quality_entry.get("passed")) if isinstance(quality_entry, dict) else None,
            "score": float(quality_entry.get("score", 0.0) or 0.0)
            if isinstance(quality_entry, dict)
            else None,
            "failed_validators": failed_validators,
            "quality_tags": quality_tags,
            "issues": quality_entry.get("issues", []) if isinstance(quality_entry, dict) else [],
        },
    }
