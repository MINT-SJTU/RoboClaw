"""Dataset list / detail / delete routes."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException

from roboclaw.embodied.service import EmbodiedService


def register_dataset_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.get("/api/datasets")
    async def datasets_list_route() -> list[dict]:
        refs = await asyncio.to_thread(service.datasets.list_local_datasets)
        return [ref.to_dict() for ref in refs]

    @app.get("/api/datasets/{dataset_id:path}")
    async def dataset_detail(dataset_id: str) -> dict:
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ref.to_dict()

    @app.delete("/api/datasets/{dataset_id:path}")
    async def dataset_delete(dataset_id: str) -> dict[str, str]:
        try:
            await asyncio.to_thread(service.datasets.delete_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "id": dataset_id}
