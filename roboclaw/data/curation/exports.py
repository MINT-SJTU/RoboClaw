from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .bridge import read_parquet_rows, write_parquet_rows
from .propagation import build_hf_annotation_rows
from .state import (
    load_annotations,
    load_dataset_info,
    load_prototype_results,
    load_quality_results,
    save_quality_results,
)
from .validators import load_episode_data


def workflow_quality_dir(dataset_path: Path) -> Path:
    return dataset_path / ".workflow" / "quality"


def workflow_quality_parquet_path(dataset_path: Path) -> Path:
    return workflow_quality_dir(dataset_path) / "quality_results.parquet"


def dataset_quality_parquet_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "quality_results.parquet"


def dataset_text_annotations_parquet_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "text_annotations.parquet"


def dataset_instruction_override_manifest_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "instruction_override_manifest.json"


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
            "decision_label": str(episode.get("decision_label", "")),
            "training_weight": float(episode.get("training_weight", 0.0) or 0.0),
            "overall_score": float(episode.get("score", 0.0) or 0.0),
            "metadata_score": _validator_score(validators, "metadata"),
            "timing_score": _validator_score(validators, "timing"),
            "action_score": _validator_score(validators, "action"),
            "visual_score": _validator_score(validators, "visual"),
            "depth_score": _validator_score(validators, "depth"),
            "ee_trajectory_score": _validator_score(validators, "ee_trajectory"),
            "trajectory_dtw_score": _validator_score(validators, "trajectory_dtw"),
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
        "decision_label",
        "training_weight",
        "overall_score",
        "metadata_score",
        "timing_score",
        "action_score",
        "visual_score",
        "depth_score",
        "ee_trajectory_score",
        "trajectory_dtw_score",
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


def publish_text_annotations_as_training_tasks(
    dataset_name: str,
    dataset_path: Path,
) -> dict[str, Any]:
    """Apply text-alignment results to the LeRobot task files used by training."""
    rows = build_text_annotation_rows(dataset_name, dataset_path)
    if not rows:
        rows = read_parquet_rows(dataset_text_annotations_parquet_path(dataset_path))

    overrides = _build_episode_instruction_overrides(rows)
    manifest_path = dataset_instruction_override_manifest_path(dataset_path)
    applied_at = datetime.now(timezone.utc).isoformat()
    backup_dir = (
        dataset_path
        / ".workflow"
        / "backups"
        / "instruction-overrides"
        / _filesystem_timestamp()
    )

    if not overrides:
        return {
            "status": "no_annotations",
            "path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "backup_dir": str(backup_dir),
            "updated_episode_count": 0,
            "updated_episode_file_count": 0,
            "updated_data_file_count": 0,
            "updated_task_file_count": 0,
            "updated_info_file_count": 0,
            "task_count": 0,
            "unmatched_episode_indices": [],
        }

    episode_files = _episodes_parquet_files(dataset_path)
    if not episode_files:
        raise ValueError("No episode metadata parquet files found under meta/episodes")

    tasks_df = _load_tasks_dataframe(dataset_path)
    task_lookup = _task_index_lookup(tasks_df)
    episode_original_tasks, episode_to_task = _build_episode_task_plan(
        episode_files,
        task_lookup,
        overrides,
    )
    applied_episode_indices = sorted(set(overrides).intersection(episode_to_task))
    unmatched_episode_indices = sorted(set(overrides) - set(episode_to_task))
    if not applied_episode_indices:
        return {
            "status": "no_matching_episodes",
            "path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "backup_dir": str(backup_dir),
            "updated_episode_count": 0,
            "updated_episode_file_count": 0,
            "updated_data_file_count": 0,
            "updated_task_file_count": 0,
            "updated_info_file_count": 0,
            "task_count": len(set(episode_to_task.values())),
            "unmatched_episode_indices": unmatched_episode_indices,
        }

    task_to_index, task_list = _build_task_index(episode_to_task)
    backed_up_files: set[str] = set()

    def backup_file(path: Path) -> None:
        if not path.exists():
            return
        relative_path = path.relative_to(dataset_path)
        target = backup_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        backed_up_files.add(relative_path.as_posix())

    updated_episode_files = _write_episode_task_overrides(
        dataset_path,
        episode_files,
        episode_to_task,
        backup_file,
    )
    updated_data_files = _write_data_task_indices(
        dataset_path,
        episode_to_task,
        task_to_index,
        backup_file,
    )
    updated_task_file_count = _write_tasks_parquet(
        dataset_path,
        task_list,
        backup_file,
    )
    updated_info_file_count = _write_total_tasks_to_info(
        dataset_path,
        len(task_list),
        backup_file,
    )
    synced_quality_episode_count = _sync_quality_episode_tasks(
        dataset_path,
        episode_to_task,
        applied_episode_indices,
    )

    manifest = {
        "dataset": dataset_name,
        "applied_at": applied_at,
        "source": "text_alignment",
        "backup_dir": str(backup_dir),
        "task_count": len(task_list),
        "updated_episode_count": len(applied_episode_indices),
        "updated_episode_file_count": len(updated_episode_files),
        "updated_data_file_count": len(updated_data_files),
        "updated_task_file_count": updated_task_file_count,
        "updated_info_file_count": updated_info_file_count,
        "synced_quality_episode_count": synced_quality_episode_count,
        "unmatched_episode_indices": unmatched_episode_indices,
        "files": {
            "tasks": "meta/tasks.parquet",
            "episodes": updated_episode_files,
            "data": updated_data_files,
            "info": "meta/info.json" if updated_info_file_count else "",
        },
        "backed_up_files": sorted(backed_up_files),
        "episodes": [
            {
                "episode_index": episode_index,
                "old_task": episode_original_tasks.get(episode_index, ""),
                "new_task": episode_to_task[episode_index],
                "annotation_count": overrides[episode_index]["annotation_count"],
                "version_number": overrides[episode_index]["version_number"],
                "updated_at": overrides[episode_index]["updated_at"],
                "source": overrides[episode_index]["source"],
            }
            for episode_index in applied_episode_indices
        ],
    }
    backup_file(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "status": "applied",
        "path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "backup_dir": str(backup_dir),
        "updated_episode_count": len(applied_episode_indices),
        "updated_episode_file_count": len(updated_episode_files),
        "updated_data_file_count": len(updated_data_files),
        "updated_task_file_count": updated_task_file_count,
        "updated_info_file_count": updated_info_file_count,
        "synced_quality_episode_count": synced_quality_episode_count,
        "task_count": len(task_list),
        "unmatched_episode_indices": unmatched_episode_indices,
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
                "quality_passed": "quality-pass" in quality_tags,
                "quality_tags": json.dumps(quality_tags, ensure_ascii=False),
                "version_number": version_number,
                "updated_at": updated_at,
                "task_label": task_context.get("label", ""),
                "task_text": task_context.get("text", ""),
            })

    return rows


def _filesystem_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _episodes_parquet_files(dataset_path: Path) -> list[Path]:
    episodes_root = dataset_path / "meta" / "episodes"
    if not episodes_root.exists():
        return []
    return sorted(episodes_root.rglob("*.parquet"))


def _build_episode_instruction_overrides(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            episode_index = int(row.get("episode_index"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(episode_index, []).append(row)

    overrides: dict[int, dict[str, Any]] = {}
    for episode_index, episode_rows in grouped.items():
        ordered_rows = sorted(
            episode_rows,
            key=lambda item: (
                _coerce_float(item.get("start_time")) is None,
                _coerce_float(item.get("start_time")) or 0.0,
                str(item.get("annotation_id") or ""),
            ),
        )
        span_texts = _unique_texts(
            _first_nonempty_text(row.get("text"), row.get("label"))
            for row in ordered_rows
        )
        task_context_text = _first_nonempty_text(
            ordered_rows[0].get("task_text"),
            ordered_rows[0].get("task_label"),
        )
        instruction = " ; ".join(span_texts) if span_texts else task_context_text
        if not instruction:
            continue
        overrides[episode_index] = {
            "instruction": instruction,
            "annotation_count": len(ordered_rows),
            "version_number": max(
                _coerce_int(row.get("version_number")) or 0
                for row in ordered_rows
            ),
            "updated_at": max(
                (str(row.get("updated_at") or "") for row in ordered_rows),
                default="",
            ),
            "source": _instruction_source(ordered_rows),
        }
    return overrides


def _load_tasks_dataframe(dataset_path: Path) -> pd.DataFrame | None:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    tasks_df = pd.read_parquet(tasks_path)
    if tasks_df.index.name == "task":
        return tasks_df
    if "__index_level_0__" in tasks_df.columns:
        tasks_df = tasks_df.set_index("__index_level_0__")
        tasks_df.index.name = "task"
        return tasks_df
    if "task" in tasks_df.columns:
        tasks_df = tasks_df.set_index("task")
        tasks_df.index.name = "task"
        return tasks_df
    tasks_df.index.name = "task"
    return tasks_df


def _task_index_lookup(tasks_df: pd.DataFrame | None) -> dict[int, str]:
    if tasks_df is None or "task_index" not in tasks_df.columns:
        return {}
    lookup: dict[int, str] = {}
    for task, row in tasks_df.iterrows():
        task_index = _coerce_int(row.get("task_index"))
        if task_index is None:
            continue
        task_text = str(task).strip()
        if task_text:
            lookup[task_index] = task_text
    return lookup


def _build_episode_task_plan(
    episode_files: list[Path],
    task_lookup: dict[int, str],
    overrides: dict[int, dict[str, Any]],
) -> tuple[dict[int, str], dict[int, str]]:
    original_tasks: dict[int, str] = {}
    episode_to_task: dict[int, str] = {}
    for episode_file in episode_files:
        rows = pd.read_parquet(episode_file).to_dict("records")
        for row in rows:
            episode_index = _coerce_int(row.get("episode_index"))
            if episode_index is None:
                continue
            original_task = _episode_row_task_text(row, task_lookup)
            replacement = overrides.get(episode_index, {}).get("instruction")
            if not original_task and not replacement:
                raise ValueError(f"Episode {episode_index} has no task text to keep or replace")
            original_tasks.setdefault(episode_index, original_task)
            episode_to_task[episode_index] = str(replacement or original_task)
    return original_tasks, episode_to_task


def _build_task_index(episode_to_task: dict[int, str]) -> tuple[dict[str, int], list[str]]:
    task_to_index: dict[str, int] = {}
    task_list: list[str] = []
    for episode_index in sorted(episode_to_task):
        task = episode_to_task[episode_index]
        if task in task_to_index:
            continue
        task_to_index[task] = len(task_list)
        task_list.append(task)
    return task_to_index, task_list


def _write_episode_task_overrides(
    dataset_path: Path,
    episode_files: list[Path],
    episode_to_task: dict[int, str],
    backup_file: Callable[[Path], None],
) -> list[str]:
    updated_files: list[str] = []
    for episode_file in episode_files:
        df = pd.read_parquet(episode_file)
        if "episode_index" not in df.columns:
            continue
        changed = False
        tasks_column: list[list[str]] = []
        for _, row in df.iterrows():
            episode_index = _coerce_int(row.get("episode_index"))
            if episode_index is None or episode_index not in episode_to_task:
                tasks_column.append(_as_task_list(row.get("tasks")))
                continue
            new_tasks = [episode_to_task[episode_index]]
            tasks_column.append(new_tasks)
            if _as_task_list(row.get("tasks")) != new_tasks:
                changed = True
        if not changed:
            continue
        backup_file(episode_file)
        df["tasks"] = pd.Series(tasks_column, index=df.index, dtype="object")
        df.to_parquet(episode_file, index=False)
        updated_files.append(episode_file.relative_to(dataset_path).as_posix())
    return updated_files


def _write_data_task_indices(
    dataset_path: Path,
    episode_to_task: dict[int, str],
    task_to_index: dict[str, int],
    backup_file: Callable[[Path], None],
) -> list[str]:
    data_root = dataset_path / "data"
    if not data_root.exists():
        return []
    updated_files: list[str] = []
    for data_file in sorted(data_root.rglob("*.parquet")):
        df = pd.read_parquet(data_file)
        if "episode_index" not in df.columns:
            raise ValueError(f"Data parquet '{data_file}' has no episode_index column")
        new_indices: list[int] = []
        for value in df["episode_index"].tolist():
            episode_index = _coerce_int(value)
            if episode_index is None or episode_index not in episode_to_task:
                raise ValueError(f"Data parquet '{data_file}' references unknown episode {value!r}")
            new_indices.append(task_to_index[episode_to_task[episode_index]])

        current_indices = (
            [_coerce_int(value) for value in df["task_index"].tolist()]
            if "task_index" in df.columns
            else []
        )
        if current_indices == new_indices:
            continue
        backup_file(data_file)
        df["task_index"] = pd.Series(new_indices, index=df.index, dtype="int64")
        df.to_parquet(data_file, index=False)
        updated_files.append(data_file.relative_to(dataset_path).as_posix())
    return updated_files


def _write_tasks_parquet(
    dataset_path: Path,
    task_list: list[str],
    backup_file: Callable[[Path], None],
) -> int:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    backup_file(tasks_path)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(task_list)))},
        index=pd.Index(task_list, name="task"),
    )
    tasks_df.to_parquet(tasks_path)
    return 1


def _write_total_tasks_to_info(
    dataset_path: Path,
    task_count: int,
    backup_file: Callable[[Path], None],
) -> int:
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    if info.get("total_tasks") == task_count:
        return 0
    backup_file(info_path)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info["total_tasks"] = task_count
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 1


def _sync_quality_episode_tasks(
    dataset_path: Path,
    episode_to_task: dict[int, str],
    applied_episode_indices: list[int],
) -> int:
    quality = load_quality_results(dataset_path)
    if not quality:
        return 0
    applied = set(applied_episode_indices)
    changed = 0
    for episode in quality.get("episodes", []) or []:
        episode_index = _coerce_int(episode.get("episode_index"))
        if episode_index is None or episode_index not in applied:
            continue
        new_task = episode_to_task[episode_index]
        if episode.get("task") == new_task:
            continue
        episode["task"] = new_task
        episode["task_value"] = new_task
        changed += 1
    if changed:
        save_quality_results(dataset_path, quality)
    return changed


def _episode_row_task_text(row: dict[str, Any], task_lookup: dict[int, str]) -> str:
    task_text = _first_nonempty_text(
        _first_from_sequence(row.get("tasks")),
        row.get("task"),
        row.get("task_label"),
        row.get("instruction"),
        row.get("language_instruction"),
        row.get("language_instruction_2"),
        row.get("language_instruction_3"),
    )
    if task_text:
        return task_text
    task_index = _coerce_int(row.get("task_index"))
    if task_index is not None:
        return task_lookup.get(task_index, "")
    return ""


def _instruction_source(rows: list[dict[str, Any]]) -> str:
    sources = {str(row.get("source") or "") for row in rows}
    if any("propagat" in source for source in sources):
        return "semantic_propagation"
    if "user" in sources:
        return "manual_annotation"
    return "text_alignment"


def _as_task_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _first_from_sequence(value: Any) -> str:
    task_values = _as_task_list(value)
    return task_values[0] if task_values else ""


def _unique_texts(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip() if value is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
