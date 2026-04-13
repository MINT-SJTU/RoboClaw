"""FastAPI routes for the remote-first dataset explorer page."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from loguru import logger

from roboclaw.http.remote_explorer import (
    build_remote_dataset_info,
    build_remote_explorer_payload,
    load_remote_episode_detail,
)

router = APIRouter(prefix="/api/explorer")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/datasets")
async def explorer_datasets() -> list[dict]:
    """Dataset explorer is remote-first; no local catalog is maintained."""
    return []


@router.get("/dashboard")
async def explorer_dashboard(dataset: str) -> dict[str, Any]:
    """Return full explorer payload for a dataset."""
    payload = await asyncio.to_thread(build_remote_explorer_payload, dataset)
    logger.info("Explorer dashboard loaded for '{}'", dataset)
    return payload


@router.get("/episode")
async def explorer_episode(dataset: str, episode_index: int) -> dict[str, Any]:
    """Return episode detail: sample rows, joint trajectory, video paths."""
    payload = await asyncio.to_thread(load_remote_episode_detail, dataset, episode_index)
    logger.info("Explorer episode loaded for '{}' #{}", dataset, episode_index)
    return payload


@router.get("/dataset-info")
async def explorer_dataset_info(dataset: str) -> dict[str, Any]:
    """Return a small dataset summary for direct HF dataset selection."""
    payload = await asyncio.to_thread(build_remote_dataset_info, dataset)
    logger.info("Explorer dataset info loaded for '{}'", dataset)
    return payload
