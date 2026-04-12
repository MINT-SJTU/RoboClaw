"""FastAPI routes for the curation quality/prototype/annotation pipeline."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from huggingface_hub import snapshot_download
from loguru import logger
from pydantic import BaseModel

from roboclaw.http.dashboard_datasets import (
    datasets_root,
    extract_action_names,
    extract_state_names,
    get_dataset_info,
    list_datasets,
    resolve_dataset_path,
)
from roboclaw.http.remote_explorer import build_remote_dataset_info
from roboclaw.embodied.curation.exports import (
    dataset_quality_parquet_path,
    dataset_text_annotations_parquet_path,
    export_quality_csv,
    publish_quality_metadata_parquet,
    publish_text_annotations_metadata_parquet,
    workflow_quality_parquet_path,
)
from roboclaw.embodied.curation.features import (
    build_joint_trajectory_payload,
    resolve_task_value,
    resolve_timestamp,
)
from roboclaw.embodied.curation.service import CurationService
from roboclaw.embodied.curation.state import (
    load_annotations,
    load_propagation_results,
    load_prototype_results,
    load_quality_results,
    load_workflow_state,
    save_annotations,
    save_workflow_state,
    set_stage_pause_requested,
)
from roboclaw.embodied.curation.validators import load_episode_data

router = APIRouter(prefix="/api/curation")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QualityRunRequest(BaseModel):
    dataset: str
    selected_validators: list[str]
    episode_indices: list[int] | None = None
    threshold_overrides: dict[str, float] | None = None


class PrototypeRunRequest(BaseModel):
    dataset: str
    cluster_count: int | None = None
    candidate_limit: int = 50

    def model_post_init(self, _context: Any) -> None:
        if self.candidate_limit > 200:
            self.candidate_limit = 200


class AnnotationSaveRequest(BaseModel):
    dataset: str
    episode_index: int
    task_context: dict[str, Any]
    annotations: list[dict[str, Any]]


class PropagationRunRequest(BaseModel):
    dataset: str
    source_episode_index: int


class HFDatasetImportRequest(BaseModel):
    dataset_id: str
    include_videos: bool = True
    force: bool = False


class DatasetPublishRequest(BaseModel):
    dataset: str


_IMPORT_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_WORKFLOW_TASKS: dict[tuple[str, str], asyncio.Task[Any]] = {}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_in_background(coro: Any, dataset_path: Path, stage_key: str) -> None:
    """Wrapper so background tasks log errors and update state on failure."""
    task_key = (str(dataset_path.resolve()), stage_key)
    try:
        await coro
    except asyncio.CancelledError:
        state = load_workflow_state(dataset_path)
        state["stages"][stage_key]["status"] = "error"
        save_workflow_state(dataset_path, state)
        raise
    except Exception:
        logger.exception("Background workflow task failed")
        state = load_workflow_state(dataset_path)
        state["stages"][stage_key]["status"] = "error"
        save_workflow_state(dataset_path, state)
    finally:
        _ACTIVE_WORKFLOW_TASKS.pop(task_key, None)


def _register_workflow_task(dataset_path: Path, stage_key: str, coro: Any) -> None:
    task_key = (str(dataset_path.resolve()), stage_key)
    existing = _ACTIVE_WORKFLOW_TASKS.get(task_key)
    if existing is not None and not existing.done():
        existing.cancel()
    task = asyncio.create_task(_run_in_background(coro, dataset_path, stage_key))
    _ACTIVE_WORKFLOW_TASKS[task_key] = task


def _reconcile_stale_running_state(dataset_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    resolved_dataset = str(dataset_path.resolve())
    changed = False
    for stage_key, stage in state.get("stages", {}).items():
        if stage.get("status") != "running":
            continue
        active_task = _ACTIVE_WORKFLOW_TASKS.get((resolved_dataset, stage_key))
        if active_task is not None and not active_task.done():
            continue
        stage["status"] = "error"
        summary = stage.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        summary["warning"] = "Previous run was interrupted before completion."
        stage["summary"] = summary
        changed = True
    if changed:
        save_workflow_state(dataset_path, state)
    return state


def _import_allow_patterns(include_videos: bool) -> list[str]:
    patterns = [
        "meta/**",
        "README*",
    ]
    if include_videos:
        patterns.append("videos/**")
    return patterns


def _record_import_job(
    job_id: str,
    *,
    dataset_id: str,
    status: str,
    include_videos: bool,
    message: str = "",
    imported_dataset: str | None = None,
    local_path: str | None = None,
) -> None:
    _IMPORT_JOBS[job_id] = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": status,
        "include_videos": include_videos,
        "message": message,
        "imported_dataset": imported_dataset,
        "local_path": local_path,
    }


def _validate_dataset_id_path(dataset_id: str) -> None:
    root = datasets_root()
    target = (root / dataset_id).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Invalid dataset_id: path traversal detected")


def _import_dataset_snapshot(
    dataset_id: str,
    *,
    include_videos: bool,
    force: bool,
) -> dict[str, Any]:
    _validate_dataset_id_path(dataset_id)
    root = datasets_root()
    target_dir = root / dataset_id
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if force and target_dir.exists():
        for child in target_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)

    snapshot_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=_import_allow_patterns(include_videos),
    )
    info_path = target_dir / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("source_dataset") != dataset_id:
                info["source_dataset"] = dataset_id
                info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to stamp source_dataset into {}", info_path)
    return {
        "dataset_id": dataset_id,
        "local_path": str(Path(snapshot_path)),
        "dataset_name": dataset_id,
    }


def _ensure_dataset_workspace(dataset_id: str) -> Path:
    return resolve_dataset_path(dataset_id)


async def _run_hf_import(job_id: str, body: HFDatasetImportRequest) -> None:
    _record_import_job(
        job_id,
        dataset_id=body.dataset_id,
        status="running",
        include_videos=body.include_videos,
        message="Downloading dataset snapshot from Hugging Face",
    )
    try:
        payload = await asyncio.to_thread(
            _import_dataset_snapshot,
            body.dataset_id,
            include_videos=body.include_videos,
            force=body.force,
        )
        _record_import_job(
            job_id,
            dataset_id=body.dataset_id,
            status="completed",
            include_videos=body.include_videos,
            message="Dataset imported",
            imported_dataset=payload["dataset_name"],
            local_path=payload["local_path"],
        )
    except Exception as exc:
        _record_import_job(
            job_id,
            dataset_id=body.dataset_id,
            status="error",
            include_videos=body.include_videos,
            message=str(exc),
        )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _episode_time_bounds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    timestamps = [
        timestamp
        for row in rows
        if (timestamp := resolve_timestamp(row)) is not None
    ]
    if not timestamps:
        return None, None
    return timestamps[0], timestamps[-1]


def _derive_task_value(data: dict[str, Any]) -> str:
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


def _serialize_quality_results(results: dict[str, Any] | None) -> dict[str, Any]:
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


def _serialize_prototype_results(results: dict[str, Any] | None) -> dict[str, Any]:
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
                "episode_index": _coerce_int(member.get("record_key")),
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


def _serialize_propagation_results(results: dict[str, Any] | None) -> dict[str, Any]:
    if not results:
        return {
            "source_episode_index": None,
            "target_count": 0,
            "propagated": [],
        }
    return results


def _build_workspace_payload(dataset: str, dataset_path: Path, episode_index: int) -> dict[str, Any]:
    data = load_episode_data(dataset_path, episode_index)
    info = data.get("info", {})
    rows = data.get("rows", [])
    start_timestamp, end_timestamp = _episode_time_bounds(rows)
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
    task_value = _derive_task_value(data)
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


def _update_annotation_stage(dataset_path: Path, episode_index: int) -> None:
    state = load_workflow_state(dataset_path)
    annotation_stage = state["stages"]["annotation"]
    annotated_episodes = {
        coerced
        for value in annotation_stage.get("annotated_episodes", [])
        if (coerced := _coerce_int(value)) is not None
    }
    annotated_episodes.add(episode_index)
    annotation_stage["annotated_episodes"] = sorted(annotated_episodes)
    annotation_stage["summary"] = {
        "annotated_count": len(annotation_stage["annotated_episodes"]),
        "last_saved_episode_index": episode_index,
    }
    save_workflow_state(dataset_path, state)


# ---------------------------------------------------------------------------
# Dataset listing (reuses embodied datasets module)
# ---------------------------------------------------------------------------


@router.get("/datasets")
async def workflow_datasets_list() -> list[dict]:
    """List available datasets."""
    return list_datasets(datasets_root())


@router.post("/datasets/import-hf")
async def workflow_import_hf_dataset(
    body: HFDatasetImportRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Download a Hugging Face dataset snapshot into the local datasets root."""
    job_id = uuid4().hex[:12]
    _record_import_job(
        job_id,
        dataset_id=body.dataset_id,
        status="queued",
        include_videos=body.include_videos,
        message="Queued for import",
    )
    background_tasks.add_task(_run_hf_import, job_id, body)
    return {"job_id": job_id, "status": "queued"}


@router.get("/datasets/import-status/{job_id}")
async def workflow_import_hf_status(job_id: str) -> dict[str, Any]:
    """Return background import status for a Hugging Face dataset."""
    payload = _IMPORT_JOBS.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Import job '{job_id}' not found")
    return payload


@router.get("/datasets/{name:path}")
async def workflow_dataset_detail(name: str) -> dict:
    """Get detailed info for a single dataset.

    Uses ``{name:path}`` so nested HF names like ``cadene/droid_1.0.1``
    are captured as a single parameter.  This route is registered after
    the fixed-prefix ``/datasets/import-*`` routes to avoid shadowing them.
    """
    info = get_dataset_info(datasets_root(), name)
    if info is not None:
        return info
    try:
        return build_remote_dataset_info(name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found: {exc}") from exc


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


@router.get("/state")
async def workflow_state(dataset: str) -> dict[str, Any]:
    """Get the current workflow state for a dataset."""
    dataset_path = _ensure_dataset_workspace(dataset)
    state = load_workflow_state(dataset_path)
    return _reconcile_stale_running_state(dataset_path, state)


@router.get("/quality-results")
async def workflow_quality_results(dataset: str) -> dict[str, Any]:
    """Get the latest detailed quality-validation results for a dataset."""
    dataset_path = _ensure_dataset_workspace(dataset)
    payload = _serialize_quality_results(load_quality_results(dataset_path))
    payload["working_parquet_path"] = str(workflow_quality_parquet_path(dataset_path))
    payload["published_parquet_path"] = str(dataset_quality_parquet_path(dataset_path))
    return payload


def _delete_quality_results(dataset: str) -> dict[str, Any]:
    dataset_path = _ensure_dataset_workspace(dataset)
    state = load_workflow_state(dataset_path)
    quality_stage = state["stages"]["quality_validation"]
    if quality_stage.get("status") == "running":
        raise HTTPException(status_code=409, detail="Quality validation is still running")

    removed_paths: list[str] = []
    for path in (
        dataset_path / ".workflow" / "quality" / "latest.json",
        workflow_quality_parquet_path(dataset_path),
        dataset_quality_parquet_path(dataset_path),
    ):
        if not path.exists():
            continue
        path.unlink()
        removed_paths.append(str(path))

    quality_stage["status"] = "idle"
    quality_stage["selected_validators"] = []
    quality_stage["latest_run"] = None
    quality_stage["pause_requested"] = False
    quality_stage["summary"] = None
    save_workflow_state(dataset_path, state)
    logger.info("Deleted quality results for dataset '{}'", dataset)
    return {"status": "deleted", "removed_paths": removed_paths}


@router.delete("/quality-results")
async def workflow_delete_quality_results(dataset: str) -> dict[str, Any]:
    """Delete the persisted quality-validation results for a dataset."""
    return _delete_quality_results(dataset)





@router.get("/prototype-results")
async def workflow_prototype_results(dataset: str) -> dict[str, Any]:
    """Get the latest detailed prototype-discovery results for a dataset."""
    dataset_path = _ensure_dataset_workspace(dataset)
    return _serialize_prototype_results(load_prototype_results(dataset_path))


@router.get("/propagation-results")
async def workflow_propagation_results(dataset: str) -> dict[str, Any]:
    """Get the latest semantic-propagation results for a dataset."""
    dataset_path = _ensure_dataset_workspace(dataset)
    payload = _serialize_propagation_results(load_propagation_results(dataset_path))
    payload["published_parquet_path"] = str(dataset_text_annotations_parquet_path(dataset_path))
    return payload


# ---------------------------------------------------------------------------
# Stage 1: Quality validation
# ---------------------------------------------------------------------------


@router.post("/quality-run")
async def quality_run(body: QualityRunRequest) -> dict[str, str]:
    """Start batch quality validation as a background task."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    service = CurationService(dataset_path, body.dataset)

    async def _task() -> None:
        await asyncio.to_thread(
            service.run_quality_batch,
            body.selected_validators,
            body.episode_indices,
            body.threshold_overrides,
        )

    _register_workflow_task(dataset_path, "quality_validation", _task())
    logger.info("Quality run queued for dataset '{}'", body.dataset)
    return {"status": "started"}


@router.post("/quality-pause")
async def quality_pause(body: DatasetPublishRequest) -> dict[str, Any]:
    """Request that a running quality-validation task pause after the current episode."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    state = load_workflow_state(dataset_path)
    quality_stage = state["stages"]["quality_validation"]
    if quality_stage.get("status") != "running":
        raise HTTPException(status_code=409, detail="Quality validation is not running")
    set_stage_pause_requested(dataset_path, "quality_validation", True)
    logger.info("Quality pause requested for dataset '{}'", body.dataset)
    return {"status": "pause_requested"}


@router.post("/quality-resume")
async def quality_resume(body: QualityRunRequest) -> dict[str, str]:
    """Resume a paused quality-validation task from the latest partial results."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    state = load_workflow_state(dataset_path)
    quality_stage = state["stages"]["quality_validation"]
    if quality_stage.get("status") != "paused":
        raise HTTPException(status_code=409, detail="Quality validation is not paused")

    existing = load_quality_results(dataset_path)
    if not existing:
        raise HTTPException(status_code=409, detail="No paused quality results to resume")

    completed = {
        int(episode.get("episode_index"))
        for episode in existing.get("episodes", [])
        if episode.get("episode_index") is not None
    }
    total = int(existing.get("total", 0) or 0)
    if body.episode_indices:
        remaining = [index for index in body.episode_indices if index not in completed]
    else:
        remaining = [index for index in range(total) if index not in completed]

    service = CurationService(dataset_path, body.dataset)
    selected_validators = existing.get("selected_validators") or body.selected_validators
    threshold_overrides = existing.get("threshold_overrides") or body.threshold_overrides

    async def _task() -> None:
        await asyncio.to_thread(
            service.run_quality_batch,
            selected_validators,
            remaining,
            threshold_overrides,
            None,
            True,
        )

    _register_workflow_task(dataset_path, "quality_validation", _task())
    logger.info(
        "Quality resume queued for dataset '{}' with {} remaining episodes",
        body.dataset,
        len(remaining),
    )
    return {"status": "started"}


@router.get("/quality-results.csv")
async def workflow_quality_results_csv(
    dataset: str,
    failed_only: bool = False,
) -> PlainTextResponse:
    """Export the current quality-result table as CSV."""
    dataset_path = _ensure_dataset_workspace(dataset)
    csv_text = export_quality_csv(dataset, dataset_path, failed_only=failed_only)
    filename = f"{Path(dataset).name}-quality-results.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return PlainTextResponse(csv_text, media_type="text/csv", headers=headers)


@router.post("/quality-publish")
async def workflow_quality_publish(body: DatasetPublishRequest) -> dict[str, Any]:
    """Publish the current quality results into dataset metadata as parquet."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    return publish_quality_metadata_parquet(body.dataset, dataset_path)


# ---------------------------------------------------------------------------
# Stage 2: Prototype discovery
# ---------------------------------------------------------------------------


@router.post("/prototype-run")
async def prototype_run(body: PrototypeRunRequest) -> dict[str, str]:
    """Start prototype discovery as a background task."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    service = CurationService(dataset_path, body.dataset)

    async def _task() -> None:
        await asyncio.to_thread(
            service.run_prototype_discovery,
            body.cluster_count,
            body.candidate_limit,
        )

    _register_workflow_task(dataset_path, "prototype_discovery", _task())
    logger.info("Prototype run queued for dataset '{}'", body.dataset)
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Stage 3: Annotations
# ---------------------------------------------------------------------------


@router.get("/annotations")
async def get_annotations(dataset: str, episode_index: int) -> dict[str, Any]:
    """Get annotations for a specific episode."""
    dataset_path = _ensure_dataset_workspace(dataset)
    result = load_annotations(dataset_path, episode_index)
    if result is None:
        return {
            "episode_index": episode_index,
            "annotations": [],
            "task_context": {},
            "version_number": 0,
        }
    return result


@router.get("/annotation-workspace")
async def get_annotation_workspace(dataset: str, episode_index: int) -> dict[str, Any]:
    """Load the annotation workspace payload for a specific episode."""
    dataset_path = _ensure_dataset_workspace(dataset)
    return _build_workspace_payload(dataset, dataset_path, episode_index)


@router.post("/annotations")
async def post_annotations(body: AnnotationSaveRequest) -> dict[str, Any]:
    """Save annotations for a specific episode."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    data: dict[str, Any] = {
        "episode_index": body.episode_index,
        "task_context": body.task_context,
        "annotations": body.annotations,
    }
    save_annotations(dataset_path, body.episode_index, data)
    _update_annotation_stage(dataset_path, body.episode_index)
    logger.info("Annotations saved for episode {} in '{}'", body.episode_index, body.dataset)
    saved = load_annotations(dataset_path, body.episode_index)
    if saved is None:
        raise HTTPException(status_code=500, detail="Annotation save did not persist")
    return saved


# ---------------------------------------------------------------------------
# Stage 3: Semantic propagation
# ---------------------------------------------------------------------------


@router.post("/propagation-run")
async def propagation_run(body: PropagationRunRequest) -> dict[str, str]:
    """Start semantic propagation as a background task."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    service = CurationService(dataset_path, body.dataset)

    async def _task() -> None:
        await asyncio.to_thread(
            service.run_semantic_propagation,
            body.source_episode_index,
        )

    _register_workflow_task(dataset_path, "annotation", _task())
    logger.info(
        "Propagation run queued for dataset '{}' from episode {}",
        body.dataset, body.source_episode_index,
    )
    return {"status": "started"}


@router.post("/text-annotations-publish")
async def workflow_text_annotations_publish(body: DatasetPublishRequest) -> dict[str, Any]:
    """Publish current annotation state into dataset metadata as parquet."""
    dataset_path = _ensure_dataset_workspace(body.dataset)
    return publish_text_annotations_metadata_parquet(body.dataset, dataset_path)


# ---------------------------------------------------------------------------
# Video serving
# ---------------------------------------------------------------------------


@router.get("/video/{path:path}")
async def serve_video(path: str, dataset: str) -> FileResponse:
    """Serve a video file from a dataset directory.

    The *path* is validated to stay within the dataset directory to prevent
    path traversal attacks.  The dataset name is passed as a query parameter
    to support nested names containing slashes (e.g. ``cadene/droid_1.0.1``).
    """
    dataset_path = _ensure_dataset_workspace(dataset)
    video_path = (dataset_path / path).resolve()

    # Prevent path traversal
    if not str(video_path).startswith(str(dataset_path.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )
