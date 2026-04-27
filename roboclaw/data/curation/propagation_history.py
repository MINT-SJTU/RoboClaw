from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def coerce_episode_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_propagated_source_episodes(
    annotation_stage: dict[str, Any],
    previous_results: dict[str, Any] | None,
    source_episode_index: int,
) -> list[int]:
    sources = {
        index
        for value in annotation_stage.get("propagated_source_episodes", [])
        if (index := coerce_episode_index(value)) is not None
    }
    if previous_results:
        previous_source = coerce_episode_index(previous_results.get("source_episode_index"))
        if previous_source is not None:
            sources.add(previous_source)
        sources.update(
            index
            for value in previous_results.get("source_episode_indices", [])
            if (index := coerce_episode_index(value)) is not None
        )
    sources.add(source_episode_index)
    return sorted(sources)


def collect_propagation_targets(
    prototype_results: dict[str, Any] | None,
    source_episode_index: int,
) -> list[dict[str, Any]]:
    if prototype_results is None:
        return []

    refinement = prototype_results.get("refinement", {})
    clusters = refinement.get("clusters", [])
    if not clusters:
        clusters = prototype_results.get("clustering", {}).get("clusters", [])

    targets: list[dict[str, Any]] = []
    source_key = str(source_episode_index)
    for cluster in clusters:
        member_keys = [str(m.get("record_key", "")) for m in cluster.get("members", [])]
        if source_key not in member_keys:
            continue
        for member in cluster.get("members", []):
            member_key = str(member.get("record_key", ""))
            if member_key == source_key:
                continue
            targets.append({
                "episode_index": int(member_key),
                "prototype_score": 1.0 - float(
                    member.get(
                        "distance_to_barycenter",
                        member.get("distance_to_prototype", 0.0),
                    ),
                ),
            })
    return targets


def recover_propagated_source_episodes(dataset_path: Path) -> list[int]:
    sources: set[int] = set()
    annotation_dir = dataset_path / ".workflow" / "annotations"
    for annotation_path in annotation_dir.glob("ep_*.json"):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        task_context = payload.get("task_context") or {}
        if task_context.get("source") != "propagation":
            continue
        source_episode_index = coerce_episode_index(task_context.get("source_episode_index"))
        if source_episode_index is not None:
            sources.add(source_episode_index)
    return sorted(sources)


def reconcile_propagated_source_episodes(
    dataset_path: Path,
    state: dict[str, Any],
    propagation_results: dict[str, Any] | None,
) -> bool:
    annotation_stage = state["stages"]["annotation"]
    sources = {
        index
        for value in annotation_stage.get("propagated_source_episodes", [])
        if (index := coerce_episode_index(value)) is not None
    }
    if propagation_results:
        latest_source = coerce_episode_index(propagation_results.get("source_episode_index"))
        if latest_source is not None:
            sources.add(latest_source)
        sources.update(
            index
            for value in propagation_results.get("source_episode_indices", [])
            if (index := coerce_episode_index(value)) is not None
        )
    sources.update(recover_propagated_source_episodes(dataset_path))
    next_sources = sorted(sources)
    if annotation_stage.get("propagated_source_episodes") == next_sources:
        return False
    annotation_stage["propagated_source_episodes"] = next_sources
    return True
