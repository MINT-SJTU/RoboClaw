"""Dataset list / detail / delete routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException

from roboclaw.embodied.service import EmbodiedService
from roboclaw.data.datasets import delete_dataset, get_dataset_info, list_datasets


def _datasets_root(service: EmbodiedService) -> Path:
    from roboclaw.embodied.command.helpers import dataset_path
    from pathlib import Path
    root = service.manifest.snapshot.get("datasets", {}).get("root", "")
    if root:
        return Path(root).expanduser()
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"


def register_dataset_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.get("/api/datasets")
    async def datasets_list_route() -> list[dict]:
        return await asyncio.to_thread(list_datasets, _datasets_root(service))

    @app.get("/api/datasets/{name}")
    async def dataset_detail(name: str) -> dict:
        info = await asyncio.to_thread(get_dataset_info, _datasets_root(service), name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        return info

    @app.delete("/api/datasets/{name}")
    async def dataset_delete(name: str) -> dict[str, str]:
        await asyncio.to_thread(delete_dataset, _datasets_root(service), name)
        return {"status": "deleted", "name": name}
