from __future__ import annotations

from typing import Any, Callable

from .clustering import discover_prototype_clusters, refine_clusters_with_dba


def discover_grouped_prototypes(
    entries: list[dict[str, Any]],
    *,
    cluster_count: int | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    grouped_entries = _group_entries(entries)
    clustering_clusters: list[dict[str, Any]] = []
    refined_clusters: list[dict[str, Any]] = []
    prototype_record_keys: list[str] = []
    anchor_record_keys: list[str] = []
    distance_matrices: dict[str, Any] = {}
    distance_pair_count = 0
    distance_backend_counts: dict[str, int] = {}
    distance_backend_details: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    cluster_counts = _allocate_cluster_counts(grouped_entries, cluster_count)

    for group_index, group in enumerate(grouped_entries):
        group_clustering = discover_prototype_clusters(
            group["entries"],
            cluster_count=cluster_counts.get(group["bucket_key"]),
            progress_callback=_group_progress(progress_callback, group),
        )
        group_refined = refine_clusters_with_dba(
            group["entries"],
            clusters=group_clustering.get("clusters", []),
            progress_callback=_group_progress(progress_callback, group),
        )

        tagged_clustering = _tag_clusters(group_clustering.get("clusters", []), group, len(clustering_clusters))
        tagged_refined = _tag_clusters(group_refined.get("clusters", []), group, len(refined_clusters))
        clustering_clusters.extend(tagged_clustering)
        refined_clusters.extend(tagged_refined)
        prototype_record_keys.extend(str(key) for key in group_clustering.get("prototype_record_keys", []))
        anchor_record_keys.extend(str(key) for key in group_refined.get("anchor_record_keys", []))
        distance_matrices[group["bucket_key"]] = group_clustering.get("distance_matrix", {})
        distance_pair_count += int(group_clustering.get("distance_pair_count", 0) or 0)
        backend = str(group_clustering.get("distance_backend") or "cpu")
        distance_backend_counts[backend] = distance_backend_counts.get(backend, 0) + 1
        distance_backend_detail = dict(group_clustering.get("distance_backend_detail") or {})
        distance_backend_detail["bucket_key"] = group["bucket_key"]
        distance_backend_details.append(distance_backend_detail)
        group_summaries.append(_group_summary(group_index, group, group_clustering, group_refined))

    distance_backend = _summarize_distance_backend(distance_backend_counts)
    return {
        "clustering": {
            "cluster_count": len(clustering_clusters),
            "clusters": clustering_clusters,
            "prototype_record_keys": prototype_record_keys,
            "distance_matrix": distance_matrices,
            "distance_matrices": distance_matrices,
            "distance_pair_count": distance_pair_count,
            "distance_backend": distance_backend,
            "distance_backend_detail": {
                "backend": distance_backend,
                "groups": distance_backend_details,
            },
            "selection_mode": "grouped_fixed" if cluster_count is not None else "grouped_auto",
            "groups": group_summaries,
        },
        "refinement": {
            "clusters": refined_clusters,
            "cluster_count": len(refined_clusters),
            "anchor_record_keys": anchor_record_keys,
            "groups": group_summaries,
        },
        "group_count": len(grouped_entries),
    }


def _group_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task_key = _stable_key(entry.get("task_key"), "unknown-task")
        robot_type = _stable_key(entry.get("robot_type"), "unknown-robot")
        canonical_mode = _stable_key(entry.get("canonical_mode"), "unknown-mode")
        bucket_key = f"{task_key}::{robot_type}::{canonical_mode}"
        group = groups.setdefault(
            bucket_key,
            {
                "bucket_key": bucket_key,
                "task_key": task_key,
                "robot_type": robot_type,
                "canonical_mode": canonical_mode,
                "entries": [],
            },
        )
        group["entries"].append(entry)
    return sorted(groups.values(), key=lambda group: group["bucket_key"])


def _allocate_cluster_counts(
    groups: list[dict[str, Any]],
    requested_count: int | None,
) -> dict[str, int | None]:
    if requested_count is None:
        return {group["bucket_key"]: None for group in groups}
    budget = min(max(int(requested_count), len(groups)), sum(len(group["entries"]) for group in groups))
    counts = {group["bucket_key"]: 1 for group in groups}
    remaining = budget - len(groups)
    while remaining > 0:
        expandable = [
            group for group in groups
            if counts[group["bucket_key"]] < len(group["entries"])
        ]
        if not expandable:
            break
        target = max(expandable, key=lambda group: len(group["entries"]) / counts[group["bucket_key"]])
        counts[target["bucket_key"]] += 1
        remaining -= 1
    return counts


def _stable_key(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _group_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    group: dict[str, Any],
) -> Callable[[dict[str, Any]], None] | None:
    if callback is None:
        return None

    def report(payload: dict[str, Any]) -> None:
        callback({
            **payload,
            "bucket_key": group["bucket_key"],
            "task_key": group["task_key"],
            "robot_type": group["robot_type"],
        })

    return report


def _tag_clusters(
    clusters: list[dict[str, Any]],
    group: dict[str, Any],
    start_index: int,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for offset, cluster in enumerate(clusters):
        tagged.append({
            **cluster,
            "cluster_index": start_index + offset,
            "bucket_key": group["bucket_key"],
            "task_key": group["task_key"],
            "robot_type": group["robot_type"],
            "canonical_mode": group["canonical_mode"],
        })
    return tagged


def _group_summary(
    group_index: int,
    group: dict[str, Any],
    clustering: dict[str, Any],
    refinement: dict[str, Any],
) -> dict[str, Any]:
    return {
        "group_index": group_index,
        "bucket_key": group["bucket_key"],
        "task_key": group["task_key"],
        "robot_type": group["robot_type"],
        "canonical_mode": group["canonical_mode"],
        "entry_count": len(group["entries"]),
        "cluster_count": int(refinement.get("cluster_count", clustering.get("cluster_count", 0)) or 0),
        "selection_mode": str(clustering.get("selection_mode") or ""),
        "distance_pair_count": int(clustering.get("distance_pair_count", 0) or 0),
        "distance_backend": str(clustering.get("distance_backend") or "cpu"),
        "selection_diagnostics": clustering.get("selection_diagnostics"),
    }


def _summarize_distance_backend(backend_counts: dict[str, int]) -> str:
    active = {backend for backend, count in backend_counts.items() if count > 0}
    if not active:
        return "cpu"
    if len(active) == 1:
        return next(iter(active))
    if any("cuda" in backend for backend in active):
        return "mixed_cuda_cpu"
    return "mixed"
