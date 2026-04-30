"""Curation service — orchestrates the 3-stage quality/prototype/annotation pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from loguru import logger

from . import propagation_history
from .alignment_overview import build_alignment_overview
from .exports import (
    dataset_quality_parquet_path,
    dataset_text_annotations_parquet_path,
    save_working_quality_parquet,
    workflow_quality_parquet_path,
)
from .features import resolve_timestamp
from .propagation import propagate_annotation_spans
from .prototypes import discover_grouped_prototypes
from .quality_defaults import build_quality_defaults
from .quality_results import aggregate_quality_results, run_base_quality_validators
from .reference_tube import TRAJECTORY_DTW_VALIDATOR
from .serializers import (
    build_workspace_payload,
    coerce_int,
    serialize_propagation_results,
    serialize_prototype_results,
    serialize_quality_results,
)
from .state import (
    is_stage_pause_requested,
    load_annotations,
    load_dataset_info,
    load_propagation_results,
    load_prototype_results,
    load_quality_results,
    load_workflow_state,
    save_annotations,
    save_propagation_results,
    save_prototype_results,
    save_quality_results,
    save_workflow_state,
)
from .trajectory_entries import (
    build_propagation_entry,
    build_prototype_entry,
    propagation_dtw_config,
)
from .trajectory_quality import append_trajectory_dtw_results
from .validators import load_episode_data, run_quality_validators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_load_info = load_dataset_info


def _episode_range(info: dict[str, Any]) -> list[int]:
    total = info.get("total_episodes", 0)
    return list(range(total))


def _is_remote_session_dataset(dataset_path: Path) -> bool:
    from roboclaw.data import dataset_sessions

    parsed = dataset_sessions.parse_session_handle(dataset_path.name)
    if parsed is not None:
        return parsed[0] == "remote"

    resolved = dataset_path.resolve()
    remote_root = (dataset_sessions._session_root() / "remote").resolve()
    try:
        relative = resolved.relative_to(remote_root)
    except ValueError:
        return False
    return len(relative.parts) == 2 and relative.parts[1] == "dataset"


def _safe_rmtree(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or not str(resolved).startswith(str(resolved_root) + "/"):
        return
    shutil.rmtree(resolved)


def _safe_unlink(path: Path, root: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not str(resolved).startswith(str(resolved_root) + "/"):
        return False
    resolved.unlink()
    return True


def _prune_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    resolved_stop = stop_at.resolve()
    while current.exists() and current.resolve() != resolved_stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remote_cache_download_root(dataset_path: Path) -> Path:
    return dataset_path / ".cache" / "huggingface" / "download"


def _cleanup_remote_quality_cache(dataset_path: Path) -> dict[str, Any]:
    if not _is_remote_session_dataset(dataset_path):
        return {"removed_paths": [], "removed_count": 0}

    removed_paths: list[str] = []
    for path in (dataset_path / "videos", dataset_path / ".remote-cache", _remote_cache_download_root(dataset_path)):
        if not path.exists():
            continue
        _safe_rmtree(path, dataset_path)
        removed_paths.append(str(path))
    return {"removed_paths": removed_paths, "removed_count": len(removed_paths)}


def _video_file_references(
    dataset_path: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    episode_index: int,
) -> list[Path]:
    template = info.get("video_path")
    features = info.get("features", {})
    if not isinstance(template, str) or not isinstance(features, dict):
        return []

    video_paths: list[Path] = []
    chunk_size = int(info.get("chunks_size", 1000) or 1000)
    for video_key, config in features.items():
        if not isinstance(config, dict) or config.get("dtype") != "video":
            continue
        prefix = f"videos/{video_key}/"
        chunk_index = coerce_int(episode_meta.get(f"{prefix}chunk_index"))
        if chunk_index is None:
            chunk_index = coerce_int(episode_meta.get("video_chunk_index"))
        if chunk_index is None:
            chunk_index = episode_index // max(chunk_size, 1)

        file_index = coerce_int(episode_meta.get(f"{prefix}file_index"))
        if file_index is None:
            file_index = coerce_int(episode_meta.get("video_file_index"))
        rendered = template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index if file_index is not None else 0,
            episode_index=episode_index,
            episode_chunk=chunk_index,
        )
        video_paths.append(dataset_path / rendered)
    return video_paths


def _load_episode_meta_map(dataset_path: Path) -> dict[int, dict[str, Any]]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        rows: dict[int, dict[str, Any]] = {}
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            index = coerce_int(payload.get("episode_index"))
            if index is not None:
                rows[index] = payload
        return rows

    episodes_root = dataset_path / "meta" / "episodes"
    if not episodes_root.exists():
        return {}
    rows = {}
    from .bridge import read_parquet_rows

    for parquet_path in sorted(episodes_root.rglob("*.parquet")):
        for payload in read_parquet_rows(parquet_path):
            index = coerce_int(payload.get("episode_index"))
            if index is not None:
                rows[index] = payload
    return rows


def _cleanup_completed_remote_episode_assets(
    dataset_path: Path,
    info: dict[str, Any],
    completed_episode_index: int,
    remaining_episode_indices: set[int],
) -> dict[str, Any]:
    if not _is_remote_session_dataset(dataset_path):
        return {"removed_paths": [], "removed_count": 0}

    episode_meta_map = _load_episode_meta_map(dataset_path)
    completed_meta = episode_meta_map.get(completed_episode_index, {})
    candidate_paths = set(
        _video_file_references(dataset_path, info, completed_meta, completed_episode_index)
    )
    if not candidate_paths:
        return {"removed_paths": [], "removed_count": 0}

    future_paths: set[Path] = set()
    for episode_index in remaining_episode_indices:
        future_paths.update(
            _video_file_references(
                dataset_path,
                info,
                episode_meta_map.get(episode_index, {}),
                episode_index,
            )
        )

    removed_paths: list[str] = []
    for path in sorted(candidate_paths - future_paths):
        if _safe_unlink(path, dataset_path):
            removed_paths.append(str(path))
            _prune_empty_parents(path, dataset_path / "videos")
    return {"removed_paths": removed_paths, "removed_count": len(removed_paths)}


def _cleanup_existing_remote_quality_assets(
    dataset_path: Path,
    info: dict[str, Any],
    completed_episode_indices: set[int],
    remaining_episode_indices: set[int],
) -> dict[str, Any]:
    if not _is_remote_session_dataset(dataset_path):
        return {"removed_paths": [], "removed_count": 0}

    episode_meta_map = _load_episode_meta_map(dataset_path)
    future_paths: set[Path] = set()
    for episode_index in remaining_episode_indices:
        future_paths.update(
            _video_file_references(
                dataset_path,
                info,
                episode_meta_map.get(episode_index, {}),
                episode_index,
            )
        )

    candidate_paths: set[Path] = set()
    for episode_index in completed_episode_indices:
        candidate_paths.update(
            _video_file_references(
                dataset_path,
                info,
                episode_meta_map.get(episode_index, {}),
                episode_index,
            )
        )

    removed_paths: list[str] = []
    for path in sorted(candidate_paths - future_paths):
        if _safe_unlink(path, dataset_path):
            removed_paths.append(str(path))
            _prune_empty_parents(path, dataset_path / "videos")
    return {"removed_paths": removed_paths, "removed_count": len(removed_paths)}


def _set_stage_status(
    dataset_path: Path,
    stage_key: str,
    status: str,
) -> dict[str, Any]:
    state = load_workflow_state(dataset_path)
    state["stages"][stage_key]["status"] = status
    save_workflow_state(dataset_path, state)
    return state


def _set_prototype_stage_context(
    dataset_path: Path,
    *,
    quality_filter_mode: str,
    selected_episode_indices: list[int],
    summary: dict[str, Any] | None = None,
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["prototype_discovery"]
    stage["quality_filter_mode"] = quality_filter_mode
    stage["selected_episode_indices"] = list(selected_episode_indices)
    if summary is not None:
        stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _update_prototype_running_summary(
    dataset_path: Path,
    summary_update: dict[str, Any],
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["prototype_discovery"]
    summary = stage.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    summary.update(summary_update)
    stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _update_annotation_running_summary(
    dataset_path: Path,
    summary_update: dict[str, Any],
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["annotation"]
    summary = stage.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    summary.update(summary_update)
    stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _update_quality_running_summary(
    dataset_path: Path,
    summary_update: dict[str, Any],
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    summary = stage.get("summary")
    if not isinstance(summary, dict):
        summary = _quality_summary_from_results(dataset_path, state)
    summary.update(summary_update)
    stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _update_stage_summary(
    dataset_path: Path,
    stage_key: str,
    summary: dict[str, Any],
    *,
    status: str = "completed",
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"][stage_key]
    stage["status"] = status
    stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _configure_quality_stage(
    dataset_path: Path,
    *,
    status: str,
    selected_validators: list[str],
    active_run_id: str | None = None,
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    stage["status"] = status
    stage["selected_validators"] = list(selected_validators)
    stage["active_run_id"] = active_run_id
    stage["pause_requested"] = False
    if status == "running":
        stage["summary"] = None
    save_workflow_state(dataset_path, state)


def _quality_run_is_current(dataset_path: Path, run_id: str | None) -> bool:
    if run_id is None:
        return True
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    return stage.get("active_run_id") == run_id


def _quality_summary_from_results(
    dataset_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    results = load_quality_results(dataset_path) or {}
    episodes = results.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []

    total = coerce_int(results.get("total"))
    if total is None:
        existing_summary = state["stages"]["quality_validation"].get("summary")
        if isinstance(existing_summary, dict):
            total = coerce_int(existing_summary.get("total"))
    if total is None:
        total = coerce_int(_load_info(dataset_path).get("total_episodes")) or len(episodes)

    passed = coerce_int(results.get("passed"))
    if passed is None:
        passed = sum(1 for episode in episodes if episode.get("passed"))

    failed = coerce_int(results.get("failed"))
    if failed is None:
        failed = max(len(episodes) - passed, 0)

    overall_score = results.get("overall_score", 0.0)
    completed = len(episodes)
    return {
        "total": total,
        "completed": completed,
        "remaining": max(total - completed, 0),
        "passed": passed,
        "failed": failed,
        "overall_score": overall_score,
        "progress_percent": round((completed / max(total, 1)) * 100, 1),
        "quality_parquet_path": None,
    }


def _mark_quality_stage_paused(
    dataset_path: Path,
    *,
    pause_requested: bool,
) -> dict[str, Any]:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    stage["status"] = "paused"
    stage["summary"] = _quality_summary_from_results(dataset_path, state)
    stage["pause_requested"] = pause_requested
    stage["active_run_id"] = None
    save_workflow_state(dataset_path, state)
    return state


def _finish_quality_stage(
    dataset_path: Path,
    *,
    status: str,
    summary: dict[str, Any],
    run_id: str | None,
) -> bool:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    if run_id is not None and stage.get("active_run_id") != run_id:
        return False
    stage["status"] = status
    stage["summary"] = summary
    stage["pause_requested"] = False
    stage["active_run_id"] = None
    save_workflow_state(dataset_path, state)
    return True


def _load_episode_duration(dataset_path: Path, episode_index: int) -> float:
    """Return episode duration in seconds from parquet timestamps."""
    data = load_episode_data(dataset_path, episode_index, include_videos=False)
    rows = data["rows"]
    if len(rows) < 2:
        return 0.0
    timestamps = [resolve_timestamp(r) for r in rows]
    valid = [t for t in timestamps if t is not None]
    if len(valid) < 2:
        return 0.0
    return max(valid[-1] - valid[0], 0.0)


# ---------------------------------------------------------------------------
# CurationService
# ---------------------------------------------------------------------------


class CurationService:
    """Orchestrates the 3-stage curation pipeline for a single dataset.

    A single instance is created at application startup.  Dataset-specific
    parameters are passed to each method rather than stored on ``__init__``.
    """

    def __init__(self) -> None:
        self._active_tasks: dict[tuple[str, str], asyncio.Task[Any]] = {}

    # ------------------------------------------------------------------
    # Legacy constructor shim — accepts (dataset_path, dataset_name) so
    # existing call-sites (e.g. ``CurationService(dp, dn)``) keep working
    # until they are migrated.
    # ------------------------------------------------------------------

    @classmethod
    def _legacy(cls, dataset_path: Path, dataset_name: str | None = None) -> _LegacyCurationService:
        return _LegacyCurationService(dataset_path, dataset_name)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _task_key(self, dataset_path: Path, stage_key: str) -> tuple[str, str]:
        return (str(dataset_path.resolve()), stage_key)

    def _active_stage_task(
        self,
        dataset_path: Path,
        stage_key: str,
    ) -> asyncio.Task[Any] | None:
        return self._active_tasks.get(self._task_key(dataset_path, stage_key))

    def _stage_task_is_running(self, dataset_path: Path, stage_key: str) -> bool:
        task = self._active_stage_task(dataset_path, stage_key)
        return task is not None and not task.done()

    async def _run_in_background(
        self,
        coro: Any,
        dataset_path: Path,
        stage_key: str,
    ) -> None:
        """Wrapper that logs errors and updates state on failure."""
        task_key = self._task_key(dataset_path, stage_key)
        current_task = asyncio.current_task()
        try:
            await coro
        except asyncio.CancelledError:
            if self._active_tasks.get(task_key) is current_task:
                state = load_workflow_state(dataset_path)
                stage = state["stages"][stage_key]
                if stage.get("status") != "paused" and not stage.get("pause_requested"):
                    stage["status"] = "error"
                    save_workflow_state(dataset_path, state)
            raise
        except Exception:
            logger.exception("Background workflow task failed")
            if self._active_tasks.get(task_key) is current_task:
                state = load_workflow_state(dataset_path)
                state["stages"][stage_key]["status"] = "error"
                save_workflow_state(dataset_path, state)
        finally:
            if self._active_tasks.get(task_key) is current_task:
                self._active_tasks.pop(task_key, None)

    def _register_workflow_task(
        self,
        dataset_path: Path,
        stage_key: str,
        coro: Any,
    ) -> None:
        """Schedule *coro* as the active background task for a stage."""
        task_key = self._task_key(dataset_path, stage_key)
        existing = self._active_stage_task(dataset_path, stage_key)
        if existing is not None and not existing.done():
            existing.cancel()
        task = asyncio.create_task(
            self._run_in_background(coro, dataset_path, stage_key),
        )
        self._active_tasks[task_key] = task

    def reconcile_stale_state(
        self,
        dataset_path: Path,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Mark ``running`` stages whose task has vanished as ``error``."""
        resolved_dataset = str(dataset_path.resolve())
        changed = False
        for stage_key, stage in state.get("stages", {}).items():
            if stage.get("status") != "running":
                continue
            active_task = self._active_tasks.get((resolved_dataset, stage_key))
            if active_task is not None and not active_task.done():
                continue
            stage["status"] = "error"
            if stage_key == "quality_validation":
                stage["active_run_id"] = None
                stage["pause_requested"] = False
            summary = stage.get("summary")
            if not isinstance(summary, dict):
                summary = {}
            summary["warning"] = "Previous run was interrupted before completion."
            stage["summary"] = summary
            changed = True
        if changed:
            save_workflow_state(dataset_path, state)
        return state

    # ------------------------------------------------------------------
    # High-level orchestration (called from thin route layer)
    # ------------------------------------------------------------------

    async def start_quality_run(
        self,
        dataset_path: Path,
        dataset_name: str,
        selected_validators: list[str],
        episode_indices: list[int] | None,
        threshold_overrides: dict[str, float] | None,
    ) -> dict[str, str]:
        svc = _LegacyCurationService(dataset_path, dataset_name)
        run_id = uuid4().hex

        async def _task() -> None:
            await asyncio.to_thread(
                svc.run_quality_batch,
                selected_validators,
                episode_indices,
                threshold_overrides,
                None,
                False,
                run_id,
            )

        self._register_workflow_task(dataset_path, "quality_validation", _task())
        logger.info("Quality run queued for dataset '{}'", dataset_name)
        return {"status": "started"}

    def pause_quality_run(
        self,
        dataset_path: Path,
        dataset_name: str,
    ) -> dict[str, Any]:
        state = load_workflow_state(dataset_path)
        quality_stage = state["stages"]["quality_validation"]
        if quality_stage.get("status") != "running":
            raise ValueError("Quality validation is not running")

        active_task = self._active_stage_task(dataset_path, "quality_validation")
        pause_requested = (
            active_task is not None
            and not active_task.done()
            and quality_stage.get("active_run_id") is None
        )
        _mark_quality_stage_paused(dataset_path, pause_requested=pause_requested)
        if active_task is not None and not active_task.done():
            active_task.cancel()
        logger.info("Quality pause applied for dataset '{}'", dataset_name)
        return {"status": "paused", "pause_requested": pause_requested}

    async def start_quality_resume(
        self,
        dataset_path: Path,
        dataset_name: str,
        selected_validators: list[str],
        episode_indices: list[int] | None,
        threshold_overrides: dict[str, float] | None,
    ) -> dict[str, str]:
        state = load_workflow_state(dataset_path)
        quality_stage = state["stages"]["quality_validation"]
        if quality_stage.get("status") != "paused":
            raise ValueError("Quality validation is not paused")

        existing = load_quality_results(dataset_path)
        if not existing:
            raise ValueError("No paused quality results to resume")

        completed = {
            int(episode.get("episode_index"))
            for episode in existing.get("episodes", [])
            if episode.get("episode_index") is not None
        }
        total = int(existing.get("total", 0) or 0)
        if episode_indices:
            remaining = [index for index in episode_indices if index not in completed]
        else:
            remaining = [index for index in range(total) if index not in completed]

        svc = _LegacyCurationService(dataset_path, dataset_name)
        resolved_validators = existing.get("selected_validators") or selected_validators
        resolved_overrides = existing.get("threshold_overrides") or threshold_overrides
        run_id = uuid4().hex
        last_progress_phase: str | None = None
        last_progress_bucket = -1

        def _progress(payload: dict[str, Any]) -> None:
            nonlocal last_progress_phase, last_progress_bucket
            phase = str(payload.get("phase", "quality_validation"))
            progress_percent = float(payload.get("progress_percent", 0.0) or 0.0)
            progress_bucket = int(progress_percent)
            is_complete = payload.get("completed") == payload.get("total")
            if (
                phase == last_progress_phase
                and progress_bucket == last_progress_bucket
                and not is_complete
            ):
                return
            last_progress_phase = phase
            last_progress_bucket = progress_bucket
            summary = {
                "phase": phase,
                "progress_percent": progress_percent,
            }
            if "total" in payload:
                summary["total"] = payload["total"]
            if "completed" in payload:
                summary["completed"] = payload["completed"]
            if "episode_index" in payload:
                summary["episode_index"] = payload["episode_index"]
            _update_quality_running_summary(dataset_path, summary)

        async def _task() -> None:
            await asyncio.to_thread(
                svc.run_quality_batch,
                resolved_validators,
                remaining,
                resolved_overrides,
                _progress,
                True,
                run_id,
            )

        self._register_workflow_task(dataset_path, "quality_validation", _task())
        logger.info(
            "Quality resume queued for dataset '{}' with {} remaining episodes",
            dataset_name,
            len(remaining),
        )
        return {"status": "started"}

    async def start_prototype_run(
        self,
        dataset_path: Path,
        dataset_name: str,
        cluster_count: int | None,
        candidate_limit: int | None,
        episode_indices: list[int] | None = None,
        quality_filter_mode: str = "passed",
    ) -> dict[str, str]:
        svc = _LegacyCurationService(dataset_path, dataset_name)
        selected_episode_indices = list(episode_indices or [])
        _set_prototype_stage_context(
            dataset_path,
            quality_filter_mode=quality_filter_mode,
            selected_episode_indices=selected_episode_indices,
            summary={
                "candidate_count": len(selected_episode_indices),
                "entry_count": 0,
                "cluster_count": 0,
                "group_count": 0,
                "quality_filter_mode": quality_filter_mode,
                "phase": "queued",
                "progress_percent": 0,
            },
        )

        last_progress_phase: str | None = None
        last_progress_bucket = -1

        def _progress(payload: dict[str, Any]) -> None:
            nonlocal last_progress_phase, last_progress_bucket
            phase = str(payload.get("phase", "running"))
            progress_percent = float(payload.get("progress_percent", 0) or 0)
            progress_bucket = int(progress_percent)
            is_complete = (
                payload.get("completed") == payload.get("total")
                or payload.get("pairs_completed") == payload.get("pairs_total")
            )
            if (
                phase == last_progress_phase
                and progress_bucket == last_progress_bucket
                and not is_complete
            ):
                return
            last_progress_phase = phase
            last_progress_bucket = progress_bucket
            summary_update = {
                "quality_filter_mode": quality_filter_mode,
                "phase": phase,
                "progress_percent": progress_percent,
            }
            if "total" in payload:
                summary_update["candidate_count"] = payload["total"]
            if "completed" in payload:
                summary_update["entry_count"] = payload["completed"]
            if "pairs_total" in payload:
                summary_update["distance_pair_count"] = payload["pairs_total"]
            if "pairs_completed" in payload:
                summary_update["distance_pairs_completed"] = payload["pairs_completed"]
            _update_prototype_running_summary(dataset_path, summary_update)

        async def _task() -> None:
            await asyncio.to_thread(
                svc.run_prototype_discovery,
                cluster_count,
                candidate_limit,
                _progress,
                selected_episode_indices or None,
                quality_filter_mode,
            )

        self._register_workflow_task(dataset_path, "prototype_discovery", _task())
        logger.info("Prototype run queued for dataset '{}'", dataset_name)
        return {"status": "started"}

    async def start_propagation_run(
        self,
        dataset_path: Path,
        dataset_name: str,
        source_episode_index: int,
    ) -> dict[str, str]:
        if self._stage_task_is_running(dataset_path, "annotation"):
            logger.info(
                "Propagation run already active for dataset '{}'; ignoring duplicate request",
                dataset_name,
            )
            return {"status": "already_running"}

        _set_stage_status(dataset_path, "annotation", "running")
        _update_annotation_running_summary(
            dataset_path,
            {
                "source_episode_index": source_episode_index,
                "phase": "queued",
                "completed": 0,
                "total": 0,
                "progress_percent": 0,
            },
        )
        svc = _LegacyCurationService(dataset_path, dataset_name)
        last_progress_bucket = -1

        def _progress(payload: dict[str, Any]) -> None:
            nonlocal last_progress_bucket
            progress_percent = float(payload.get("progress_percent", 0) or 0)
            progress_bucket = int(progress_percent)
            is_complete = payload.get("completed") == payload.get("total")
            if progress_bucket == last_progress_bucket and not is_complete:
                return
            last_progress_bucket = progress_bucket
            summary_update = {
                "source_episode_index": source_episode_index,
                "phase": str(payload.get("phase", "semantic_propagation")),
                "progress_percent": progress_percent,
            }
            if "completed" in payload:
                summary_update["completed"] = payload["completed"]
            if "total" in payload:
                summary_update["total"] = payload["total"]
                summary_update["target_count"] = payload["total"]
            _update_annotation_running_summary(dataset_path, summary_update)

        async def _task() -> None:
            def _run(_source_episode_index: int) -> dict[str, Any]:
                return svc.run_semantic_propagation(
                    source_episode_index,
                    _progress,
                )

            await asyncio.to_thread(
                _run,
                source_episode_index,
            )

        self._register_workflow_task(dataset_path, "annotation", _task())
        logger.info(
            "Propagation run queued for dataset '{}' from episode {}",
            dataset_name,
            source_episode_index,
        )
        return {"status": "started"}

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_quality_results(self, dataset_path: Path) -> dict[str, Any]:
        payload = serialize_quality_results(load_quality_results(dataset_path))
        payload["working_parquet_path"] = str(workflow_quality_parquet_path(dataset_path))
        payload["published_parquet_path"] = str(dataset_quality_parquet_path(dataset_path))
        return payload

    def get_quality_defaults(self, dataset_path: Path, dataset_name: str | None = None) -> dict[str, Any]:
        return build_quality_defaults(dataset_path, dataset_name)

    def get_prototype_results(self, dataset_path: Path) -> dict[str, Any]:
        return serialize_prototype_results(load_prototype_results(dataset_path))

    def get_propagation_results(self, dataset_path: Path) -> dict[str, Any]:
        payload = serialize_propagation_results(load_propagation_results(dataset_path))
        payload["published_parquet_path"] = str(dataset_text_annotations_parquet_path(dataset_path))
        return payload

    def get_workflow_state(self, dataset_path: Path) -> dict[str, Any]:
        state = load_workflow_state(dataset_path)
        propagation_results = load_propagation_results(dataset_path)
        changed = propagation_history.reconcile_propagated_source_episodes(
            dataset_path,
            state,
            propagation_results,
        )
        state = self.reconcile_stale_state(dataset_path, state)
        if changed:
            save_workflow_state(dataset_path, state)
        return state

    def delete_quality_results(
        self,
        dataset: str,
        dataset_path: Path,
    ) -> dict[str, Any]:
        state = load_workflow_state(dataset_path)
        quality_stage = state["stages"]["quality_validation"]
        if quality_stage.get("status") == "running":
            raise ValueError("Quality validation is still running")

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
        quality_stage["active_run_id"] = None
        quality_stage["pause_requested"] = False
        quality_stage["summary"] = None
        save_workflow_state(dataset_path, state)
        logger.info("Deleted quality results for dataset '{}'", dataset)
        return {"status": "deleted", "removed_paths": removed_paths}

    def save_episode_annotations(
        self,
        dataset_path: Path,
        episode_index: int,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        save_annotations(dataset_path, episode_index, data)
        self._update_annotation_stage(dataset_path, episode_index)
        saved = load_annotations(dataset_path, episode_index)
        if saved is None:
            raise RuntimeError("Annotation save did not persist")
        return saved

    def get_workspace_payload(
        self,
        dataset: str,
        dataset_path: Path,
        episode_index: int,
    ) -> dict[str, Any]:
        return build_workspace_payload(dataset, dataset_path, episode_index)

    def get_alignment_overview(self, dataset_path: Path) -> dict[str, Any]:
        return build_alignment_overview(dataset_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_annotation_stage(dataset_path: Path, episode_index: int) -> None:
        state = load_workflow_state(dataset_path)
        annotation_stage = state["stages"]["annotation"]
        annotated_episodes = {
            coerced
            for value in annotation_stage.get("annotated_episodes", [])
            if (coerced := coerce_int(value)) is not None
        }
        annotated_episodes.add(episode_index)
        annotation_stage["annotated_episodes"] = sorted(annotated_episodes)
        annotation_stage["summary"] = {
            "annotated_count": len(annotation_stage["annotated_episodes"]),
            "last_saved_episode_index": episode_index,
        }
        save_workflow_state(dataset_path, state)



# ---------------------------------------------------------------------------
# _LegacyCurationService — holds dataset_path/name for pipeline methods.
# Used internally by CurationService orchestration methods and by existing
# test code that constructs ``CurationService(dataset_path, name)``.
# ---------------------------------------------------------------------------


class _LegacyCurationService:
    """Bound pipeline executor for a single dataset."""

    def __init__(self, dataset_path: Path, dataset_name: str | None = None):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name or dataset_path.name

    # ------------------------------------------------------------------
    # Stage 1: Quality validation
    # ------------------------------------------------------------------

    def run_quality_batch(
        self,
        selected_validators: list[str],
        episode_indices: list[int] | None = None,
        threshold_overrides: dict[str, float] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        resume_existing: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Run quality validation across episodes.

        Updates workflow state to running/completed and persists results.
        """
        _configure_quality_stage(
            self.dataset_path,
            status="running",
            selected_validators=selected_validators,
            active_run_id=run_id,
        )
        logger.info("Quality batch started for {}", self.dataset_path.name)

        info = _load_info(self.dataset_path)
        indices = list(episode_indices) if episode_indices is not None else _episode_range(info)
        include_trajectory_dtw = TRAJECTORY_DTW_VALIDATOR in selected_validators
        base_validators = [
            name for name in selected_validators
            if name != TRAJECTORY_DTW_VALIDATOR
        ]
        per_episode: list[dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        total = len(indices)

        initial_completed = 0

        if resume_existing:
            existing = load_quality_results(self.dataset_path) or {}
            existing_episodes = existing.get("episodes", [])
            if isinstance(existing_episodes, list):
                per_episode = list(existing_episodes)
            initial_completed = len(per_episode)
            passed_count = sum(1 for episode in per_episode if episode.get("passed"))
            failed_count = max(len(per_episode) - passed_count, 0)
            existing_total = existing.get("total")
            try:
                total = int(existing_total)
            except (TypeError, ValueError):
                total = len(per_episode) + len(indices)

        remaining_indices = set(indices)
        if per_episode:
            completed_indices = {
                index
                for episode in per_episode
                if (index := coerce_int(episode.get("episode_index"))) is not None
            }
            cleanup = _cleanup_existing_remote_quality_assets(
                self.dataset_path,
                info,
                completed_indices,
                remaining_indices,
            )
            if cleanup["removed_count"]:
                logger.info(
                    "Removed {} remote quality cache files from completed episodes",
                    cleanup["removed_count"],
                )

        def current_or_saved_results() -> dict[str, Any]:
            return load_quality_results(self.dataset_path) or aggregate_quality_results(
                per_episode,
                selected_validators,
                passed_count,
                failed_count,
                total,
                threshold_overrides,
            )

        def finalize_quality_run(stage_status: str) -> dict[str, Any]:
            if not _quality_run_is_current(self.dataset_path, run_id):
                return current_or_saved_results()
            aggregated = aggregate_quality_results(
                per_episode,
                selected_validators,
                passed_count,
                failed_count,
                total,
                threshold_overrides,
            )
            save_quality_results(self.dataset_path, aggregated)

            parquet_path = None
            try:
                parquet_info = save_working_quality_parquet(self.dataset_name, self.dataset_path)
                parquet_path = parquet_info["path"]
            except Exception:
                logger.exception(
                    "Failed to write working quality parquet for {}",
                    self.dataset_path.name,
                )

            summary = {
                "total": total,
                "completed": len(per_episode),
                "remaining": max(total - len(per_episode), 0),
                "passed": passed_count,
                "failed": failed_count,
                "overall_score": aggregated["overall_score"],
                "progress_percent": round((len(per_episode) / max(total, 1)) * 100, 1),
                "quality_parquet_path": parquet_path,
            }
            finished = _finish_quality_stage(
                self.dataset_path,
                status=stage_status,
                summary=summary,
                run_id=run_id,
            )
            if not finished:
                return load_quality_results(self.dataset_path) or aggregated
            if stage_status == "paused":
                logger.info(
                    "Quality batch paused after {}/{} episodes",
                    len(per_episode),
                    total,
                )
            else:
                logger.info(
                    "Quality batch completed: {}/{} passed (mean score {:.1f})",
                    passed_count,
                    total,
                    aggregated["overall_score"],
                )
            return aggregated

        for position, ep_idx in enumerate(indices):
            if not _quality_run_is_current(self.dataset_path, run_id):
                return current_or_saved_results()
            if is_stage_pause_requested(self.dataset_path, "quality_validation"):
                return finalize_quality_run("paused")
            logger.info("Validating episode {}/{}", initial_completed + position + 1, total)
            result = run_base_quality_validators(
                self.dataset_path,
                ep_idx,
                selected_validators=base_validators,
                threshold_overrides=threshold_overrides,
                runner=run_quality_validators,
            )
            if not _quality_run_is_current(self.dataset_path, run_id):
                return current_or_saved_results()
            entry = {
                "episode_index": ep_idx,
                "passed": result["passed"],
                "score": result["score"],
                "validators": result["validators"],
                "issues": result["issues"],
            }
            per_episode.append(entry)
            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1
            remaining_indices.discard(ep_idx)

            save_quality_results(
                self.dataset_path,
                aggregate_quality_results(
                    per_episode,
                    selected_validators,
                    passed_count,
                    failed_count,
                    total,
                    threshold_overrides,
                ),
            )

            if is_stage_pause_requested(self.dataset_path, "quality_validation"):
                return finalize_quality_run("paused")

            cleanup = _cleanup_completed_remote_episode_assets(
                self.dataset_path,
                info,
                ep_idx,
                remaining_indices,
            )
            if cleanup["removed_count"]:
                logger.info(
                    "Removed {} remote quality cache files after episode {}",
                    cleanup["removed_count"],
                    ep_idx,
                )

            if progress_callback is not None:
                progress_callback({
                    "phase": "quality_validation",
                    "episode_index": ep_idx,
                    "completed": initial_completed + position + 1,
                    "total": total,
                    "progress_percent": round(
                        ((initial_completed + position + 1) / max(total, 1)) * 100,
                        1,
                    ),
                })

        if include_trajectory_dtw:
            if not _quality_run_is_current(self.dataset_path, run_id):
                return current_or_saved_results()
            append_trajectory_dtw_results(
                self.dataset_path,
                per_episode,
                threshold_overrides=threshold_overrides,
                progress_callback=progress_callback,
            )
            if not _quality_run_is_current(self.dataset_path, run_id):
                return current_or_saved_results()
            passed_count = sum(1 for episode in per_episode if episode.get("passed"))
            failed_count = max(len(per_episode) - passed_count, 0)

        completed = finalize_quality_run("completed")
        cleanup = _cleanup_remote_quality_cache(self.dataset_path)
        if cleanup["removed_count"]:
            logger.info(
                "Removed {} remote quality cache directories after completion",
                cleanup["removed_count"],
            )
        return completed

    # ------------------------------------------------------------------
    # Stage 2: Prototype discovery
    # ------------------------------------------------------------------

    def run_prototype_discovery(
        self,
        cluster_count: int | None = None,
        candidate_limit: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        episode_indices: list[int] | None = None,
        quality_filter_mode: str = "passed",
    ) -> dict[str, Any]:
        """Run DTW + k-medoids prototype discovery on the selected episode subset."""
        _set_stage_status(self.dataset_path, "prototype_discovery", "running")
        logger.info("Prototype discovery started for {}", self.dataset_path.name)

        if episode_indices is not None:
            selected_episodes = list(episode_indices)
        elif quality_filter_mode == "raw":
            selected_episodes = _episode_range(_load_info(self.dataset_path))
        else:
            selected_episodes = _collect_passed_episodes(self.dataset_path)
        if not selected_episodes:
            return _finish_prototype_empty(self.dataset_path)

        candidates = (
            selected_episodes
            if candidate_limit is None
            else selected_episodes[:candidate_limit]
        )
        _set_prototype_stage_context(
            self.dataset_path,
            quality_filter_mode=quality_filter_mode,
            selected_episode_indices=candidates,
            summary={
                "candidate_count": len(candidates),
                "entry_count": 0,
                "cluster_count": 0,
                "group_count": 0,
                "quality_filter_mode": quality_filter_mode,
                "phase": "building_canonical",
                "progress_percent": 0,
            },
        )
        entries = _build_canonical_entries(self.dataset_path, candidates, progress_callback)
        if not entries:
            return _finish_prototype_empty(self.dataset_path)

        _update_prototype_running_summary(
            self.dataset_path,
            {
                "candidate_count": len(candidates),
                "entry_count": len(entries),
                "phase": "building_dtw_graph",
                "progress_percent": 0,
            },
        )
        prototypes = discover_grouped_prototypes(
            entries,
            cluster_count=cluster_count,
            progress_callback=progress_callback,
        )
        clustering = prototypes["clustering"]
        refined = prototypes["refinement"]

        results = {
            "clustering": clustering,
            "refinement": refined,
            "candidate_count": len(candidates),
            "entry_count": len(entries),
            "cluster_count": refined.get("cluster_count", clustering.get("cluster_count", 0)),
            "group_count": prototypes["group_count"],
            "quality_filter_mode": quality_filter_mode,
            "selected_episode_indices": candidates,
        }
        save_prototype_results(self.dataset_path, results)
        _update_stage_summary(
            self.dataset_path,
            "prototype_discovery",
            {
                "candidate_count": len(candidates),
                "entry_count": len(entries),
                "cluster_count": results["cluster_count"],
                "group_count": results["group_count"],
                "quality_filter_mode": quality_filter_mode,
                "selection_mode": clustering.get("selection_mode"),
                "distance_pair_count": clustering.get("distance_pair_count", 0),
                "distance_backend": clustering.get("distance_backend", "cpu"),
            },
        )
        logger.info(
            "Prototype discovery completed: {} entries, {} clusters",
            len(entries), results["cluster_count"],
        )
        return results

    # ------------------------------------------------------------------
    # Stage 3: Semantic propagation
    # ------------------------------------------------------------------

    def run_semantic_propagation(
        self,
        source_episode_index: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Propagate annotations from source episode to cluster members."""
        _set_stage_status(self.dataset_path, "annotation", "running")
        logger.info(
            "Semantic propagation started from episode {} for {}",
            source_episode_index, self.dataset_path.name,
        )

        source_annotations = load_annotations(self.dataset_path, source_episode_index)
        if source_annotations is None:
            return _finish_propagation_empty(self.dataset_path, source_episode_index)

        spans = source_annotations.get("annotations", [])
        if not spans:
            return _finish_propagation_empty(self.dataset_path, source_episode_index)

        source_duration = _load_episode_duration(self.dataset_path, source_episode_index)
        source_entry = build_propagation_entry(self.dataset_path, source_episode_index)
        prototype_results = load_prototype_results(self.dataset_path)
        targets = propagation_history.collect_propagation_targets(
            prototype_results, source_episode_index,
        )

        propagated: list[dict[str, Any]] = []
        persisted_annotation_targets: set[int] = {source_episode_index}
        total = len(targets)
        for position, target in enumerate(targets):
            result, persisted = _propagate_single_target(
                self.dataset_path,
                target,
                spans,
                source_duration,
                source_entry,
                source_annotations,
                source_episode_index,
            )
            propagated.append(result)
            if persisted:
                persisted_annotation_targets.add(target["episode_index"])
            if progress_callback is not None:
                progress_callback({
                    "phase": "semantic_propagation",
                    "completed": position + 1,
                    "total": total,
                    "progress_percent": round(((position + 1) / max(total, 1)) * 100, 1),
                })

        previous_results = load_propagation_results(self.dataset_path)
        state = load_workflow_state(self.dataset_path)
        annotation_stage = state["stages"]["annotation"]
        propagated_source_episodes = propagation_history.collect_propagated_source_episodes(
            annotation_stage,
            previous_results,
            source_episode_index,
        )
        results = {
            "source_episode_index": source_episode_index,
            "source_episode_indices": propagated_source_episodes,
            "target_count": len(propagated),
            "propagated": propagated,
        }
        save_propagation_results(self.dataset_path, results)
        existing_targets = {
            int(value)
            for value in annotation_stage.get("annotated_episodes", [])
            if isinstance(value, int) or str(value).isdigit()
        }
        annotation_stage["annotated_episodes"] = sorted(existing_targets | persisted_annotation_targets)
        annotation_stage["propagated_source_episodes"] = propagated_source_episodes
        save_workflow_state(self.dataset_path, state)
        _update_stage_summary(
            self.dataset_path,
            "annotation",
            {
                "source_episode_index": source_episode_index,
                "propagated_source_episodes": propagated_source_episodes,
                "target_count": len(propagated),
                "annotated_count": len(annotation_stage["annotated_episodes"]),
                "completed": len(propagated),
                "total": len(propagated),
                "phase": "semantic_propagation",
                "progress_percent": 100,
            },
        )
        logger.info(
            "Semantic propagation completed: {} targets from episode {}",
            len(propagated), source_episode_index,
        )
        return results


# ---------------------------------------------------------------------------
# Prototype helpers
# ---------------------------------------------------------------------------


def _collect_passed_episodes(dataset_path: Path) -> list[int]:
    quality = load_quality_results(dataset_path)
    if quality is None:
        return []
    return [
        ep["episode_index"]
        for ep in quality.get("episodes", [])
        if ep.get("passed")
    ]


def _build_canonical_entries(
    dataset_path: Path,
    episode_indices: list[int],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total = len(episode_indices)
    for position, ep_idx in enumerate(episode_indices):
        logger.info("Building canonical trajectory for episode {}/{}", position + 1, total)
        data = load_episode_data(dataset_path, ep_idx, include_videos=False)
        rows = data["rows"]
        if not rows:
            continue
        entry = build_prototype_entry(
            dataset_path,
            ep_idx,
            quality=_episode_quality_summary(dataset_path, ep_idx),
            data=data,
        )
        if not entry.get("sequence"):
            continue
        entries.append(entry)

        if progress_callback is not None:
            progress_callback({
                "phase": "building_canonical",
                "completed": position + 1,
                "total": total,
                "progress_percent": round(((position + 1) / max(total, 1)) * 100, 1),
            })

    return entries


def _episode_quality_summary(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    quality = load_quality_results(dataset_path)
    if quality is None:
        return {}
    for ep in quality.get("episodes", []):
        if ep.get("episode_index") == episode_index:
            return {"score": ep.get("score", 0), "passed": ep.get("passed", False)}
    return {}


def _finish_prototype_empty(dataset_path: Path) -> dict[str, Any]:
    results: dict[str, Any] = {
        "clustering": {},
        "refinement": {},
        "candidate_count": 0,
        "entry_count": 0,
        "cluster_count": 0,
    }
    save_prototype_results(dataset_path, results)
    _update_stage_summary(
        dataset_path,
        "prototype_discovery",
        {"candidate_count": 0, "entry_count": 0, "cluster_count": 0},
    )
    logger.warning("Prototype discovery: no passed episodes found")
    return results


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------


def _propagate_single_target(
    dataset_path: Path,
    target: dict[str, Any],
    spans: list[dict[str, Any]],
    source_duration: float,
    source_entry: dict[str, Any],
    source_annotations: dict[str, Any],
    source_episode_index: int,
) -> tuple[dict[str, Any], bool]:
    target_idx = target["episode_index"]
    target_duration = _load_episode_duration(dataset_path, target_idx)
    target_entry = build_propagation_entry(dataset_path, target_idx)
    target_spans = propagate_annotation_spans(
        spans,
        source_duration=source_duration,
        target_duration=target_duration,
        target_record_key=str(target_idx),
        prototype_score=target.get("prototype_score", 0.0),
        source_sequence=source_entry.get("sequence"),
        target_sequence=target_entry.get("sequence"),
        source_time_axis=source_entry.get("time_axis"),
        target_time_axis=target_entry.get("time_axis"),
        dtw_config=propagation_dtw_config(source_entry, target_entry),
    )
    result = {
        "episode_index": target_idx,
        "spans": target_spans,
        "prototype_score": target.get("prototype_score", 0.0),
        "alignment_method": "dtw" if any(span.get("source") == "dtw_propagated" for span in target_spans) else "scale",
    }
    existing = load_annotations(dataset_path, target_idx) or {}
    existing_annotations = existing.get("annotations", []) or []
    has_manual = any(
        isinstance(span, dict) and span.get("source") == "user"
        for span in existing_annotations
    )
    if has_manual:
        return result, False
    save_annotations(
        dataset_path,
        target_idx,
        {
            "episode_index": target_idx,
            "task_context": {
                **(source_annotations.get("task_context", {}) or {}),
                "source_episode_index": source_episode_index,
                "source": "propagation",
            },
            "annotations": target_spans,
        },
    )
    return result, True


def _finish_propagation_empty(
    dataset_path: Path,
    source_episode_index: int,
) -> dict[str, Any]:
    previous_results = load_propagation_results(dataset_path)
    state = load_workflow_state(dataset_path)
    annotation_stage = state["stages"]["annotation"]
    propagated_source_episodes = propagation_history.collect_propagated_source_episodes(
        annotation_stage,
        previous_results,
        source_episode_index,
    )
    results: dict[str, Any] = {
        "source_episode_index": source_episode_index,
        "source_episode_indices": propagated_source_episodes,
        "target_count": 0,
        "propagated": [],
    }
    save_propagation_results(dataset_path, results)
    annotation_stage["propagated_source_episodes"] = propagated_source_episodes
    save_workflow_state(dataset_path, state)
    _update_stage_summary(
        dataset_path,
        "annotation",
        {
            "source_episode_index": source_episode_index,
            "propagated_source_episodes": propagated_source_episodes,
            "target_count": 0,
            "completed": 0,
            "total": 0,
            "phase": "semantic_propagation",
            "progress_percent": 100,
        },
    )
    logger.warning("Semantic propagation: no annotations found for episode {}", source_episode_index)
    return results
