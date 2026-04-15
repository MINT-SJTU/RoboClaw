from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from .bridge import write_parquet_rows
from .propagation import build_hf_annotation_rows
from .state import load_annotations, load_dataset_info, load_prototype_results, load_quality_results
from .validators import load_episode_data


def workflow_quality_dir(dataset_path: Path) -> Path:
    return dataset_path / ".workflow" / "quality"


def workflow_quality_parquet_path(dataset_path: Path) -> Path:
    return workflow_quality_dir(dataset_path) / "quality_results.parquet"


def dataset_quality_parquet_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "quality_results.parquet"


def dataset_text_annotations_parquet_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "text_annotations.parquet"


_load_info = load_dataset_info


def _load_episode_meta_map(dataset_path: Path) -> dict[int, dict[str, Any]]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {}

    by_index: dict[int, dict[str, Any]] = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            episode_index = int(entry.get("episode_index"))
        except (TypeError, ValueError):
            continue
        by_index[episode_index] = entry
    return by_index


def build_quality_result_rows(dataset_name: str, dataset_path: Path) -> list[dict[str, Any]]:
    info = _load_info(dataset_path)
    episode_meta_map = _load_episode_meta_map(dataset_path)
    quality = load_quality_results(dataset_path) or {}
    rows: list[dict[str, Any]] = []

    for episode in quality.get("episodes", []):
        episode_index = int(episode.get("episode_index", -1))
        validators = episode.get("validators", {}) or {}
        issues = episode.get("issues", []) or []
        issue_types = sorted(
            {
                str(issue.get("check_name"))
                for issue in issues
                if issue.get("check_name") not in (None, "")
            },
        )
        validator_names = sorted(str(name) for name in validators.keys())
        failed_validator_count = sum(
            1
            for validator in validators.values()
            if isinstance(validator, dict) and not validator.get("passed", False)
        )
        episode_meta = episode_meta_map.get(episode_index, {})

        row = {
            "source_dataset": dataset_name,
            "source_revision": "",
            "episode_index": episode_index,
            "record_key": str(episode_index),
            "task": str(
                episode_meta.get("task")
                or episode_meta.get("task_label")
                or episode_meta.get("instruction")
                or ""
            ),
            "robot_type": str(info.get("robot_type", "")),
            "fps": info.get("fps", 0),
            "is_valid": bool(episode.get("passed", False)),
            "overall_score": float(episode.get("score", 0.0) or 0.0),
            "metadata_score": _validator_score(validators, "metadata"),
            "timing_score": _validator_score(validators, "timing"),
            "action_score": _validator_score(validators, "action"),
            "visual_score": _validator_score(validators, "visual"),
            "depth_score": _validator_score(validators, "depth"),
            "ee_trajectory_score": _validator_score(validators, "ee_trajectory"),
            "issue_count": len(issues),
            "failed_validator_count": failed_validator_count,
            "validator_names": json.dumps(validator_names, ensure_ascii=False),
            "issue_types": json.dumps(issue_types, ensure_ascii=False),
            "issues_json": json.dumps(issues, ensure_ascii=False),
            "validated_at": quality.get("validated_at", ""),
            "run_id": quality.get("run_id", ""),
        }
        rows.append(row)

    return rows


def _validator_score(validators: dict[str, Any], key: str) -> float | None:
    validator = validators.get(key)
    if not isinstance(validator, dict):
        return None
    score = validator.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def save_working_quality_parquet(dataset_name: str, dataset_path: Path) -> dict[str, Any]:
    rows = build_quality_result_rows(dataset_name, dataset_path)
    output_path = workflow_quality_parquet_path(dataset_path)
    result = write_parquet_rows(output_path, rows)
    return {
        **result,
        "path": str(output_path),
        "row_count": len(rows),
    }


def publish_quality_metadata_parquet(dataset_name: str, dataset_path: Path) -> dict[str, Any]:
    rows = build_quality_result_rows(dataset_name, dataset_path)
    output_path = dataset_quality_parquet_path(dataset_path)
    result = write_parquet_rows(output_path, rows)
    return {
        **result,
        "path": str(output_path),
        "row_count": len(rows),
    }


def export_quality_csv(dataset_name: str, dataset_path: Path, *, failed_only: bool = False) -> str:
    rows = build_quality_result_rows(dataset_name, dataset_path)
    if failed_only:
        rows = [row for row in rows if not row.get("is_valid", False)]

    output = StringIO()
    fieldnames = [
        "source_dataset",
        "episode_index",
        "record_key",
        "task",
        "robot_type",
        "fps",
        "is_valid",
        "overall_score",
        "metadata_score",
        "timing_score",
        "action_score",
        "visual_score",
        "depth_score",
        "ee_trajectory_score",
        "issue_count",
        "failed_validator_count",
        "validator_names",
        "issue_types",
        "issues_json",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    return output.getvalue()


def publish_text_annotations_metadata_parquet(
    dataset_name: str,
    dataset_path: Path,
) -> dict[str, Any]:
    rows = build_text_annotation_rows(dataset_name, dataset_path)
    output_path = dataset_text_annotations_parquet_path(dataset_path)
    result = write_parquet_rows(output_path, rows)
    return {
        **result,
        "path": str(output_path),
        "row_count": len(rows),
    }


def build_text_annotation_rows(dataset_name: str, dataset_path: Path) -> list[dict[str, Any]]:
    prototype_results = load_prototype_results(dataset_path)
    cluster_lookup = _build_cluster_lookup(prototype_results)
    info = _load_info(dataset_path)
    rows: list[dict[str, Any]] = []

    annotations_dir = dataset_path / ".workflow" / "annotations"
    for annotation_path in sorted(annotations_dir.glob("ep_*.json")):
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        episode_index = int(payload.get("episode_index", -1))
        task_context = payload.get("task_context", {}) or {}
        spans = payload.get("annotations", []) or []
        version_number = int(payload.get("version_number", 0) or 0)
        updated_at = payload.get("updated_at") or payload.get("created_at") or ""

        cluster_meta = cluster_lookup.get(episode_index, {})
        quality_tags = []
        quality = load_quality_results(dataset_path) or {}
        for episode in quality.get("episodes", []):
            if int(episode.get("episode_index", -1)) == episode_index:
                quality_tags = [
                    "quality-pass" if episode.get("passed") else "quality-risk",
                ]
                break

        hf_rows = build_hf_annotation_rows(
            dataset=dataset_name,
            record_key=str(episode_index),
            record_key_field="episode_index",
            spans=spans,
            quality_tags=quality_tags,
        )
        for index, row in enumerate(hf_rows):
            span = spans[index] if index < len(spans) else {}
            start_time = _coerce_float(row.get("start_time"))
            end_time = _coerce_float(row.get("end_time"))
            fps = _coerce_float(info.get("fps")) or 0
            rows.append({
                "source_dataset": dataset_name,
                "episode_index": episode_index,
                "record_key": str(episode_index),
                "cluster_index": cluster_meta.get("cluster_index"),
                "anchor_episode_index": cluster_meta.get("anchor_episode_index"),
                "annotation_id": str(
                    span.get("id")
                    or f"{episode_index}:{row.get('annotation_index', index + 1)}"
                ),
                "label": row.get("label", ""),
                "text": row.get("text", ""),
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": _time_to_frame(start_time, fps),
                "end_frame": _time_to_frame(end_time, fps),
                "source": span.get("source", "user"),
                "confidence": _coerce_float(span.get("prototype_score"))
                or (1.0 if span.get("source") == "user" else None),
                "propagated": bool(span.get("propagated", False)),
                "version_number": version_number,
                "updated_at": updated_at,
                "task_label": task_context.get("label", ""),
                "task_text": task_context.get("text", ""),
            })

    return rows


def _build_cluster_lookup(prototype_results: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    if not prototype_results:
        return lookup

    refinement = prototype_results.get("refinement", {})
    clusters = refinement.get("clusters") or prototype_results.get("clustering", {}).get("clusters", [])
    for cluster_index, cluster in enumerate(clusters):
        try:
            anchor_episode_index = int(
                cluster.get("anchor_record_key")
                or cluster.get("prototype_record_key"),
            )
        except (TypeError, ValueError):
            anchor_episode_index = None
        members = cluster.get("members", []) or []
        for member in members:
            try:
                episode_index = int(member.get("record_key"))
            except (TypeError, ValueError):
                continue
            lookup[episode_index] = {
                "cluster_index": int(cluster.get("cluster_index", cluster_index)),
                "anchor_episode_index": anchor_episode_index,
            }
    return lookup


def _time_to_frame(value: float | None, fps: float) -> int | None:
    if value is None or fps <= 0:
        return None
    return int(round(value * fps))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
