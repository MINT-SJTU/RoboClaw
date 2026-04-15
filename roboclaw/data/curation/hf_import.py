"""HuggingFace dataset import — download remote datasets into the local workspace."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from loguru import logger

from roboclaw.data.datasets import datasets_root


# ---------------------------------------------------------------------------
# In-memory job tracking
# ---------------------------------------------------------------------------

IMPORT_JOBS: dict[str, dict[str, Any]] = {}


def record_import_job(
    job_id: str,
    *,
    dataset_id: str,
    status: str,
    include_videos: bool,
    message: str = "",
    imported_dataset: str | None = None,
    local_path: str | None = None,
) -> None:
    """Upsert an import-job status entry."""
    IMPORT_JOBS[job_id] = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": status,
        "include_videos": include_videos,
        "message": message,
        "imported_dataset": imported_dataset,
        "local_path": local_path,
    }


def get_import_job(job_id: str) -> dict[str, Any] | None:
    """Return the status payload for *job_id*, or ``None``."""
    return IMPORT_JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset_id_path(dataset_id: str) -> None:
    """Raise ``ValueError`` if *dataset_id* escapes the datasets root."""
    root = datasets_root()
    target = (root / dataset_id).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError(f"Invalid dataset_id: path traversal detected — {dataset_id!r}")


# ---------------------------------------------------------------------------
# Snapshot download
# ---------------------------------------------------------------------------


def import_allow_patterns(include_videos: bool) -> list[str]:
    """Build the ``allow_patterns`` list for ``snapshot_download``."""
    patterns = [
        "meta/**",
        "README*",
    ]
    if include_videos:
        patterns.append("videos/**")
    return patterns


def import_dataset_snapshot(
    dataset_id: str,
    *,
    include_videos: bool,
    force: bool,
) -> dict[str, Any]:
    """Download a HuggingFace dataset snapshot into the local datasets root.

    Returns a dict with ``dataset_id``, ``local_path``, and ``dataset_name``.
    """
    validate_dataset_id_path(dataset_id)
    root = datasets_root()
    target_dir = root / dataset_id
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if force and target_dir.exists():
        import shutil

        for child in target_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    snapshot_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=import_allow_patterns(include_videos),
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


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------


async def run_hf_import(
    job_id: str,
    dataset_id: str,
    *,
    include_videos: bool,
    force: bool,
) -> None:
    """Background coroutine that drives the snapshot download."""
    record_import_job(
        job_id,
        dataset_id=dataset_id,
        status="running",
        include_videos=include_videos,
        message="Downloading dataset snapshot from Hugging Face",
    )
    try:
        payload = await asyncio.to_thread(
            import_dataset_snapshot,
            dataset_id,
            include_videos=include_videos,
            force=force,
        )
        record_import_job(
            job_id,
            dataset_id=dataset_id,
            status="completed",
            include_videos=include_videos,
            message="Dataset imported",
            imported_dataset=payload["dataset_name"],
            local_path=payload["local_path"],
        )
    except Exception as exc:
        record_import_job(
            job_id,
            dataset_id=dataset_id,
            status="error",
            include_videos=include_videos,
            message=str(exc),
        )
