"""FastAPI routes for the remote-first dataset explorer page."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from loguru import logger

from roboclaw.data.explorer.remote import (
    build_remote_dataset_info,
    build_remote_episode_page,
    build_remote_explorer_details,
    build_remote_explorer_payload,
    build_remote_explorer_summary,
    load_remote_episode_detail,
    search_remote_datasets,
)


def register_explorer_routes(app: FastAPI) -> None:
    """Register all explorer API routes on *app*."""

    @app.get("/api/explorer/datasets")
    async def explorer_datasets(query: str = "", limit: int = 8) -> list[dict]:
        """Return remote dataset suggestions for the explorer search box."""
        return await asyncio.to_thread(search_remote_datasets, query, limit)

    @app.get("/api/explorer/dashboard")
    async def explorer_dashboard(dataset: str) -> dict[str, Any]:
        """Return full explorer payload for a dataset."""
        payload = await asyncio.to_thread(build_remote_explorer_payload, dataset)
        logger.info("Explorer dashboard loaded for '{}'", dataset)
        return payload

    @app.get("/api/explorer/summary")
    async def explorer_summary(dataset: str) -> dict[str, Any]:
        """Return lightweight explorer summary counts for a dataset."""
        payload = await asyncio.to_thread(build_remote_explorer_summary, dataset)
        logger.info("Explorer summary loaded for '{}'", dataset)
        return payload

    @app.get("/api/explorer/details")
    async def explorer_details(dataset: str) -> dict[str, Any]:
        """Return explorer details without embedding the full episode index."""
        payload = await asyncio.to_thread(build_remote_explorer_details, dataset)
        logger.info("Explorer details loaded for '{}'", dataset)
        return payload

    @app.get("/api/explorer/episodes")
    async def explorer_episodes(
        dataset: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Return a paginated explorer episode index."""
        safe_page_size = max(1, min(page_size, 200))
        payload = await asyncio.to_thread(build_remote_episode_page, dataset, page, safe_page_size)
        logger.info(
            "Explorer episode page loaded for '{}' page {} size {}",
            dataset,
            payload.get("page"),
            payload.get("page_size"),
        )
        return payload

    @app.get("/api/explorer/episode")
    async def explorer_episode(
        dataset: str,
        episode_index: int,
        preview_only: bool = False,
    ) -> dict[str, Any]:
        """Return episode detail: sample rows, joint trajectory, video paths."""
        payload = await asyncio.to_thread(
            load_remote_episode_detail,
            dataset,
            episode_index,
            preview_only=preview_only,
        )
        logger.info("Explorer episode loaded for '{}' #{}", dataset, episode_index)
        return payload

    @app.get("/api/explorer/dataset-info")
    async def explorer_dataset_info(dataset: str) -> dict[str, Any]:
        """Return a small dataset summary for direct HF dataset selection."""
        payload = await asyncio.to_thread(build_remote_dataset_info, dataset)
        logger.info("Explorer dataset info loaded for '{}'", dataset)
        return payload
