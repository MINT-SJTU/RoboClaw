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
    propagated_lookup = _build_propagation_lookup(dataset_path, propagation)
    _merge_saved_propagation_annotations(dataset_path, annotated_lookup, propagated_lookup)

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
        task_context = payload.get("task_context", {}) or {}
        if not isinstance(task_context, dict):
            task_context = {}
        annotated_lookup[episode_index] = {
            "annotation_count": len(spans),
            "annotation_spans": _simplify_spans(spans),
            "raw_spans": [span for span in spans if isinstance(span, dict)],
            "task_context": task_context,
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


def _build_propagation_lookup(dataset_path: Path, propagation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    propagated_lookup: dict[int, dict[str, Any]] = {}
    source_episode_index = coerce_int(propagation.get("source_episode_index"))
    source_spans = _load_source_spans(dataset_path, source_episode_index)
    for item in propagation.get("propagated", []) or []:
        episode_index = coerce_int(item.get("episode_index"))
        if episode_index is None:
            continue
        spans = item.get("spans", []) or []
        item_source_episode_index = coerce_int(item.get("source_episode_index"))
        if item_source_episode_index is None:
            item_source_episode_index = source_episode_index
        item_source_spans = source_spans
        if item_source_episode_index != source_episode_index:
            item_source_spans = _load_source_spans(dataset_path, item_source_episode_index)
        propagated_lookup[episode_index] = {
            "propagated_count": len(spans),
            "prototype_score": item.get("prototype_score"),
            "source_episode_index": item_source_episode_index,
            "alignment_method": item.get("alignment_method") or _infer_alignment_method(spans),
            "spans": _simplify_propagated_spans(spans, item_source_spans),
        }
    return propagated_lookup


def _merge_saved_propagation_annotations(
    dataset_path: Path,
    annotated_lookup: dict[int, dict[str, Any]],
    propagated_lookup: dict[int, dict[str, Any]],
) -> None:
    for episode_index, annotation_meta in annotated_lookup.items():
        existing = propagated_lookup.get(episode_index)
        if existing and int(existing.get("propagated_count", 0) or 0) > 0:
            continue

        raw_spans = annotation_meta.get("raw_spans", [])
        if not isinstance(raw_spans, list):
            continue
        propagated_spans = [
            span
            for span in raw_spans
            if isinstance(span, dict) and _is_saved_propagated_span(span)
        ]
        if not propagated_spans:
            continue

        task_context = annotation_meta.get("task_context", {})
        if not isinstance(task_context, dict):
            task_context = {}
        source_episode_index = coerce_int(task_context.get("source_episode_index"))
        source_spans = _load_source_spans(dataset_path, source_episode_index)
        simplified_spans = _simplify_propagated_spans(propagated_spans, source_spans)
        propagated_lookup[episode_index] = {
            "propagated_count": len(simplified_spans),
            "prototype_score": _first_prototype_score(propagated_spans),
            "source_episode_index": source_episode_index,
            "alignment_method": _infer_alignment_method(propagated_spans),
            "spans": simplified_spans,
        }


def _is_saved_propagated_span(span: dict[str, Any]) -> bool:
    if span.get("propagated") is True:
        return True
    return str(span.get("source") or "") in {"dtw_propagated", "duration_scaled"}


def _first_prototype_score(spans: list[dict[str, Any]]) -> float | None:
    for span in spans:
        score = _coerce_float(span.get("prototype_score"))
        if score is not None:
            return score
    return None


def _load_source_spans(dataset_path: Path, episode_index: int | None) -> list[dict[str, Any]]:
    if episode_index is None:
        return []
    annotations = load_annotations(dataset_path, episode_index) or {}
    spans = annotations.get("annotations", []) or []
    return [span for span in spans if isinstance(span, dict)]


def _infer_alignment_method(spans: list[dict[str, Any]]) -> str:
    sources = {
        str(span.get("source") or "")
        for span in spans
        if isinstance(span, dict)
    }
    if "dtw_propagated" in sources:
        return "dtw"
    if "duration_scaled" in sources:
        return "scale"
    return ""


def _simplify_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _simplify_span(span)
        for span in spans
        if isinstance(span, dict)
    ]


def _simplify_propagated_spans(
    spans: list[dict[str, Any]],
    source_spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_id = {
        str(span.get("id")): span
        for span in source_spans
        if span.get("id") not in (None, "")
    }
    simplified: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        source_span = source_by_id.get(str(span.get("id"))) if span.get("id") not in (None, "") else None
        if source_span is None and index < len(source_spans):
            source_span = source_spans[index]
        simplified.append(_simplify_span(span, source_span))
    return simplified


def _simplify_span(
    span: dict[str, Any],
    source_span: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_time = _coerce_float(span.get("startTime"))
    end_time = _coerce_float(span.get("endTime"))
    source_start_time = _coerce_float(source_span.get("startTime")) if source_span else None
    source_end_time = _coerce_float(source_span.get("endTime")) if source_span else None
    duration_delta = None
    if start_time is not None and end_time is not None and source_start_time is not None and source_end_time is not None:
        duration_delta = (end_time - start_time) - (source_end_time - source_start_time)

    return {
        "id": span.get("id"),
        "label": span.get("label"),
        "text": span.get("text"),
        "category": span.get("category"),
        "startTime": start_time,
        "endTime": end_time,
        "source": span.get("source"),
        "target_record_key": span.get("target_record_key"),
        "prototype_score": _coerce_float(span.get("prototype_score")),
        "source_start_time": source_start_time,
        "source_end_time": source_end_time,
        "dtw_start_delay_s": _subtract_optional(start_time, source_start_time),
        "dtw_end_delay_s": _subtract_optional(end_time, source_end_time),
        "duration_delta_s": _round_optional(duration_delta),
    }


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _round_optional(left - right)


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


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
    annotation_spans = annotation_meta.get("annotation_spans", [])
    propagation_spans = propagation_meta.get("spans", [])
    original_task = _episode_task_text(episode)
    semantic_task = (
        _semantic_task_text(propagation_spans)
        or _semantic_task_text(annotation_spans)
        or _task_context_text(annotation_meta.get("task_context"))
    )
    task_is_supplemental = not original_task and bool(semantic_task)
    task = original_task or semantic_task or ""
    return {
        "episode_index": episode_index,
        "record_key": str(episode_index),
        "task": task,
        "task_source": "dataset" if original_task else "semantic_supplement" if semantic_task else "",
        "task_is_supplemental": task_is_supplemental,
        "semantic_task_text": semantic_task or "",
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
        "annotation_spans": annotation_spans,
        "propagation_source_episode_index": propagation_meta.get("source_episode_index"),
        "propagation_alignment_method": propagation_meta.get("alignment_method", ""),
        "propagation_spans": propagation_spans,
        "prototype_score": propagation_meta.get("prototype_score"),
        "updated_at": annotation_meta.get("updated_at", ""),
    }


def _episode_task_text(episode: dict[str, Any]) -> str:
    return _first_text(
        episode.get("task"),
        episode.get("task_label"),
        episode.get("task_value"),
        episode.get("instruction"),
        episode.get("language_instruction"),
        episode.get("language_instruction_2"),
        episode.get("language_instruction_3"),
    )


def _task_context_text(task_context: Any) -> str:
    if not isinstance(task_context, dict):
        return ""
    return _first_text(
        task_context.get("text"),
        task_context.get("label"),
        task_context.get("instruction"),
        task_context.get("language_instruction"),
    )


def _semantic_task_text(spans: list[dict[str, Any]]) -> str:
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = _first_text(span.get("text"), span.get("label"), span.get("category"))
        if text:
            return text
    return ""


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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
