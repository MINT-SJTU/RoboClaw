from __future__ import annotations

from pathlib import Path
from typing import Any

from .serializers import (
    coerce_int,
    serialize_propagation_results,
    serialize_prototype_results,
)
from .state import (
    load_annotations,
    load_propagation_results,
    load_prototype_results,
    load_quality_results,
)


def build_alignment_overview(dataset_path: Path) -> dict[str, Any]:
    quality = load_quality_results(dataset_path) or {}
    prototype = serialize_prototype_results(load_prototype_results(dataset_path))
    propagation = serialize_propagation_results(load_propagation_results(dataset_path))
    quality_rows = quality.get("episodes", []) or []
    annotated_lookup = _build_annotation_lookup(dataset_path)
    propagated_lookup = _build_propagation_lookup(propagation)

    issue_distribution: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    aligned_count = 0
    annotated_count = 0

    for episode in quality_rows:
        episode_index = coerce_int(episode.get("episode_index"))
        if episode_index is None:
            continue
        _collect_issue_distribution(issue_distribution, episode.get("issues", []) or [])

        annotation_meta = annotated_lookup.get(episode_index, {})
        propagation_meta = propagated_lookup.get(episode_index, {})
        annotation_count = int(annotation_meta.get("annotation_count", 0) or 0)
        propagated_count = int(propagation_meta.get("propagated_count", 0) or 0)
        if annotation_count > 0 or propagated_count > 0:
            aligned_count += 1
        if annotation_count > 0:
            annotated_count += 1

        rows.append(_build_alignment_row(
            episode,
            episode_index,
            annotation_meta,
            propagation_meta,
            annotation_count,
            propagated_count,
        ))

    return {
        "summary": _build_summary(rows, prototype, aligned_count, annotated_count),
        "distribution": _build_distribution(rows, issue_distribution),
        "rows": rows,
    }


def _build_annotation_lookup(dataset_path: Path) -> dict[int, dict[str, Any]]:
    annotated_lookup: dict[int, dict[str, Any]] = {}
    annotations_dir = dataset_path / ".workflow" / "annotations"
    if not annotations_dir.exists():
        return annotated_lookup
    for annotation_path in sorted(annotations_dir.glob("ep_*.json")):
        payload = _load_annotation_file(dataset_path, annotation_path)
        if not payload:
            continue
        episode_index = coerce_int(payload.get("episode_index"))
        if episode_index is None:
            continue
        spans = payload.get("annotations", []) or []
        annotated_lookup[episode_index] = {
            "annotation_count": len(spans),
            "updated_at": payload.get("updated_at") or payload.get("created_at") or "",
            "has_manual_annotation": len(spans) > 0,
        }
    return annotated_lookup


def _load_annotation_file(dataset_path: Path, annotation_path: Path) -> dict[str, Any] | None:
    try:
        episode_index = int(annotation_path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return load_annotations(dataset_path, episode_index)


def _build_propagation_lookup(propagation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    propagated_lookup: dict[int, dict[str, Any]] = {}
    for item in propagation.get("propagated", []) or []:
        episode_index = coerce_int(item.get("episode_index"))
        if episode_index is None:
            continue
        spans = item.get("spans", []) or []
        propagated_lookup[episode_index] = {
            "propagated_count": len(spans),
            "prototype_score": item.get("prototype_score"),
        }
    return propagated_lookup


def _collect_issue_distribution(issue_distribution: dict[str, int], issues: list[dict[str, Any]]) -> None:
    for issue in issues:
        if issue.get("passed") is True:
            continue
        issue_name = str(issue.get("check_name") or "").strip()
        if issue_name:
            issue_distribution[issue_name] = issue_distribution.get(issue_name, 0) + 1


def _build_alignment_row(
    episode: dict[str, Any],
    episode_index: int,
    annotation_meta: dict[str, Any],
    propagation_meta: dict[str, Any],
    annotation_count: int,
    propagated_count: int,
) -> dict[str, Any]:
    alignment_status = "not_started"
    if propagated_count > 0:
        alignment_status = "propagated"
    elif annotation_count > 0:
        alignment_status = "annotated"

    validators = episode.get("validators", {}) or {}
    return {
        "episode_index": episode_index,
        "record_key": str(episode_index),
        "task": "",
        "quality_passed": bool(episode.get("passed", False)),
        "quality_score": float(episode.get("score", 0.0) or 0.0),
        "quality_status": "passed" if episode.get("passed") else "failed",
        "validator_scores": {
            name: float(value.get("score", 0.0) or 0.0)
            for name, value in validators.items()
            if isinstance(value, dict)
        },
        "failed_validators": [
            str(name)
            for name, value in validators.items()
            if isinstance(value, dict) and not value.get("passed", False)
        ],
        "issues": episode.get("issues", []) or [],
        "alignment_status": alignment_status,
        "annotation_count": annotation_count,
        "propagated_count": propagated_count,
        "prototype_score": propagation_meta.get("prototype_score"),
        "updated_at": annotation_meta.get("updated_at", ""),
    }


def _build_summary(
    rows: list[dict[str, Any]],
    prototype: dict[str, Any],
    aligned_count: int,
    annotated_count: int,
) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["quality_passed"])
    perfect = sum(1 for row in rows if row["quality_score"] >= 99.95)
    return {
        "total_checked": total,
        "passed_count": passed,
        "failed_count": total - passed,
        "perfect_ratio": round((perfect / max(total, 1)) * 100, 1) if total else 0.0,
        "aligned_count": aligned_count,
        "annotated_count": annotated_count,
        "propagated_count": sum(1 for row in rows if row["alignment_status"] == "propagated"),
        "prototype_cluster_count": prototype.get("cluster_count", 0),
        "quality_filter_mode": prototype.get("quality_filter_mode", "passed"),
    }


def _build_distribution(
    rows: list[dict[str, Any]],
    issue_distribution: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "issue_types": [
            {"label": label, "count": count}
            for label, count in sorted(issue_distribution.items(), key=lambda item: item[1], reverse=True)
        ],
        "alignment_status": [
            {"label": "not_started", "count": sum(1 for row in rows if row["alignment_status"] == "not_started")},
            {"label": "annotated", "count": sum(1 for row in rows if row["alignment_status"] == "annotated")},
            {"label": "propagated", "count": sum(1 for row in rows if row["alignment_status"] == "propagated")},
        ],
    }
