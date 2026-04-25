"""FastAPI routes for the curation quality/prototype/annotation pipeline.

Thin HTTP translation layer. Business logic lives in
``roboclaw.data.curation.service.CurationService``, serialisation helpers in
``roboclaw.data.curation.serializers``, and HuggingFace import logic in
``roboclaw.data.datasets.DatasetCatalog``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from loguru import logger
from pydantic import BaseModel

from roboclaw.data import dataset_sessions
from roboclaw.data.datasets import (
    DatasetCatalog,
    get_dataset_info,
    list_datasets,
)
from roboclaw.data.curation.exports import (
    export_quality_csv,
    publish_quality_metadata_parquet,
    publish_text_annotations_metadata_parquet,
)
from roboclaw.data.curation.paths import datasets_root
from roboclaw.data.curation.service import CurationService
from roboclaw.data.curation.state import (
    load_annotations,
    load_workflow_state,
    save_workflow_state,
    set_stage_pause_requested,
)

# Module-level service singleton
_service = CurationService()
_catalog = DatasetCatalog(root_resolver=lambda: datasets_root())


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
    episode_indices: list[int] | None = None
    quality_filter_mode: str = "passed"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_curation_dataset_summaries() -> list[dict[str, Any]]:
    """Return curation-ready dataset summaries (workspace + session datasets).

    The default implementation is intentionally patchable in tests, and uses
    ``datasets_root()`` so monkeypatching ``curation_routes.datasets_root`` is
    sufficient to control the workspace listing.
    """
    root = datasets_root()
    workspace_items = list_datasets(root)
    for item in workspace_items:
        item.setdefault("name", item.get("id"))
        item.setdefault("display_name", item.get("label"))
        item.setdefault("source_kind", "workspace")
    session_items = dataset_sessions.list_session_dataset_summaries(
        include_remote=True,
        include_local_directory=True,
    )
    return workspace_items + session_items


def resolve_dataset_path(name: str) -> Path:
    """Resolve a dataset identifier into a local workspace path.

    Supports:
    - Workspace datasets under ``datasets_root()`` (including nested HF names).
    - Prepared dataset sessions (e.g. ``session:remote:...``).
    """
    if dataset_sessions.is_session_handle(name):
        return dataset_sessions.resolve_session_dataset_path(name)

    root = datasets_root()
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Datasets root '{root}' does not exist")
    resolved_root = root.resolve()

    def _is_safe(path: Path) -> bool:
        resolved = path.resolve()
        return resolved.is_dir() and str(resolved).startswith(str(resolved_root) + "/")

    direct = root / name
    if _is_safe(direct):
        return direct.resolve()

    for parent in root.iterdir():
        if not parent.is_dir():
            continue
        candidate = parent / name
        if _is_safe(candidate):
            return candidate.resolve()

    raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")


def _dataset_ref_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Coerce a summary payload into a DatasetRef-ish dict used by the web UI."""
    if (
        isinstance(summary.get("id"), str)
        and isinstance(summary.get("label"), str)
        and isinstance(summary.get("kind"), str)
        and isinstance(summary.get("stats"), dict)
        and isinstance(summary.get("capabilities"), dict)
        and "runtime" in summary
    ):
        payload = dict(summary)
    else:
        dataset_id = str(summary.get("name") or "")
        label = str(summary.get("display_name") or dataset_id)

        stats: dict[str, Any]
        if isinstance(summary.get("stats"), dict):
            stats = dict(summary["stats"])
        else:
            stats = {
                "total_episodes": int(summary.get("total_episodes", 0) or 0),
                "total_frames": int(summary.get("total_frames", 0) or 0),
                "fps": int(summary.get("fps", 0) or 0),
                "robot_type": str(summary.get("robot_type") or ""),
                "features": list(summary.get("features") or []),
                "episode_lengths": list(summary.get("episode_lengths") or []),
            }

        capabilities: dict[str, Any]
        if isinstance(summary.get("capabilities"), dict):
            capabilities = dict(summary["capabilities"])
        else:
            capabilities = {
                "can_replay": False,
                "can_train": False,
                "can_delete": False,
                "can_push": False,
                "can_pull": False,
                "can_curate": True,
            }

        slug = summary.get("slug")
        if not isinstance(slug, str) or not slug:
            slug = dataset_id.rsplit("/", 1)[-1] if dataset_id else ""
            if ":" in dataset_id:
                slug = dataset_id.split(":")[-1]

        runtime = summary.get("runtime")
        if runtime is not None and not isinstance(runtime, dict):
            runtime = None

        payload = {
            "id": dataset_id,
            "kind": str(summary.get("kind") or "local"),
            "label": label,
            "slug": slug,
            "source_dataset": str(summary.get("source_dataset") or dataset_id),
            "stats": stats,
            "capabilities": capabilities,
            "runtime": runtime,
        }

    # Back-compat fields used by some tests/legacy UI code.
    payload.setdefault("name", payload.get("id"))
    payload.setdefault("display_name", payload.get("label"))
    payload.setdefault("source_kind", summary.get("source_kind") or "workspace")
    return payload


def _ensure_dataset_workspace(dataset_id: str) -> Path:
    try:
        resolved = resolve_dataset_path(dataset_id)
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' has no local workspace")
    return resolved


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_curation_routes(app: FastAPI) -> None:
    """Register all curation API routes on *app*."""

    # -----------------------------------------------------------------------
    # Dataset listing (reuses embodied datasets module)
    # -----------------------------------------------------------------------

    @app.get("/api/curation/datasets")
    async def workflow_datasets_list() -> list[dict]:
        """List available datasets."""
        summaries = list_curation_dataset_summaries()
        return [_dataset_ref_from_summary(item) for item in summaries]

    @app.post("/api/curation/datasets/import-hf")
    async def workflow_import_hf_dataset(
        body: HFDatasetImportRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        """Download a Hugging Face dataset snapshot into the local datasets root."""
        job_id = uuid4().hex[:12]
        queued = _catalog.queue_import_job(
            job_id,
            dataset_id=body.dataset_id,
            include_videos=body.include_videos,
        )
        background_tasks.add_task(
            _catalog.run_import_job,
            job_id,
            body.dataset_id,
            include_videos=body.include_videos,
            force=body.force,
        )
        return queued.to_dict()

    @app.get("/api/curation/datasets/import-status/{job_id}")
    async def workflow_import_hf_status(job_id: str) -> dict[str, Any]:
        """Return background import status for a Hugging Face dataset."""
        payload = _catalog.get_import_job(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Import job '{job_id}' not found")
        return payload.to_dict()

    @app.get("/api/curation/datasets/{dataset_id:path}")
    async def workflow_dataset_detail(dataset_id: str) -> dict:
        """Get detailed info for a single dataset.

        Uses ``{dataset_id:path}`` so nested HF names like ``cadene/droid_1.0.1``
        are captured as a single parameter.  This route is registered after
        the fixed-prefix ``/datasets/import-*`` routes to avoid shadowing them.
        """
        root = datasets_root()
        payload = get_dataset_info(root, dataset_id)
        if payload is not None:
            return payload
        try:
            dataset = await asyncio.to_thread(_catalog.resolve_remote_dataset, dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found: {exc}") from exc
        return dataset.to_dict()

    # -----------------------------------------------------------------------
    # Workflow state
    # -----------------------------------------------------------------------

    @app.get("/api/curation/state")
    async def workflow_state(dataset: str) -> dict[str, Any]:
        """Get the current workflow state for a dataset."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_workflow_state(dataset_path)

    @app.get("/api/curation/quality-results")
    async def workflow_quality_results(dataset: str) -> dict[str, Any]:
        """Get the latest detailed quality-validation results for a dataset."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_quality_results(dataset_path)

    @app.delete("/api/curation/quality-results")
    async def workflow_delete_quality_results(dataset: str) -> dict[str, Any]:
        """Delete the persisted quality-validation results for a dataset."""
        dataset_path = _ensure_dataset_workspace(dataset)
        try:
            return _service.delete_quality_results(dataset, dataset_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/curation/prototype-results")
    async def workflow_prototype_results(dataset: str) -> dict[str, Any]:
        """Get the latest detailed prototype-discovery results for a dataset."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_prototype_results(dataset_path)

    @app.get("/api/curation/alignment-overview")
    async def workflow_alignment_overview(dataset: str) -> dict[str, Any]:
        """Get the final overview payload combining quality and text-alignment state."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_alignment_overview(dataset_path)

    @app.get("/api/curation/propagation-results")
    async def workflow_propagation_results(dataset: str) -> dict[str, Any]:
        """Get the latest semantic-propagation results for a dataset."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_propagation_results(dataset_path)

    # -----------------------------------------------------------------------
    # Stage 1: Quality validation
    # -----------------------------------------------------------------------

    @app.post("/api/curation/quality-run")
    async def quality_run(body: QualityRunRequest) -> dict[str, str]:
        """Start batch quality validation as a background task."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        return await _service.start_quality_run(
            dataset_path,
            body.dataset,
            body.selected_validators,
            body.episode_indices,
            body.threshold_overrides,
        )

    @app.post("/api/curation/quality-pause")
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

    @app.post("/api/curation/quality-resume")
    async def quality_resume(body: QualityRunRequest) -> dict[str, str]:
        """Resume a paused quality-validation task from the latest partial results."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        try:
            return await _service.start_quality_resume(
                dataset_path,
                body.dataset,
                body.selected_validators,
                body.episode_indices,
                body.threshold_overrides,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/curation/quality-results.csv")
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

    @app.post("/api/curation/quality-publish")
    async def workflow_quality_publish(body: DatasetPublishRequest) -> dict[str, Any]:
        """Publish the current quality results into dataset metadata as parquet."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        return publish_quality_metadata_parquet(body.dataset, dataset_path)

    # -----------------------------------------------------------------------
    # Stage 2: Prototype discovery
    # -----------------------------------------------------------------------

    @app.post("/api/curation/prototype-run")
    async def prototype_run(body: PrototypeRunRequest) -> dict[str, str]:
        """Start prototype discovery as a background task."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        return await _service.start_prototype_run(
            dataset_path,
            body.dataset,
            body.cluster_count,
            body.candidate_limit,
            body.episode_indices,
            body.quality_filter_mode,
        )

    # -----------------------------------------------------------------------
    # Stage 3: Annotations
    # -----------------------------------------------------------------------

    @app.get("/api/curation/annotations")
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

    @app.get("/api/curation/annotation-workspace")
    async def get_annotation_workspace(dataset: str, episode_index: int) -> dict[str, Any]:
        """Load the annotation workspace payload for a specific episode."""
        dataset_path = _ensure_dataset_workspace(dataset)
        return _service.get_workspace_payload(dataset, dataset_path, episode_index)

    @app.post("/api/curation/annotations")
    async def post_annotations(body: AnnotationSaveRequest) -> dict[str, Any]:
        """Save annotations for a specific episode."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        data: dict[str, Any] = {
            "episode_index": body.episode_index,
            "task_context": body.task_context,
            "annotations": body.annotations,
        }
        try:
            saved = _service.save_episode_annotations(dataset_path, body.episode_index, data)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.info("Annotations saved for episode {} in '{}'", body.episode_index, body.dataset)
        return saved

    # -----------------------------------------------------------------------
    # Stage 3: Semantic propagation
    # -----------------------------------------------------------------------

    @app.post("/api/curation/propagation-run")
    async def propagation_run(body: PropagationRunRequest) -> dict[str, str]:
        """Start semantic propagation as a background task."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        return await _service.start_propagation_run(
            dataset_path,
            body.dataset,
            body.source_episode_index,
        )

    @app.post("/api/curation/text-annotations-publish")
    async def workflow_text_annotations_publish(body: DatasetPublishRequest) -> dict[str, Any]:
        """Publish current annotation state into dataset metadata as parquet."""
        dataset_path = _ensure_dataset_workspace(body.dataset)
        return publish_text_annotations_metadata_parquet(body.dataset, dataset_path)

    # -----------------------------------------------------------------------
    # Video serving
    # -----------------------------------------------------------------------

    @app.get("/api/curation/video/{path:path}")
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
