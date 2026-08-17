"""Device management REST API — manifest CRUD + catalog."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class RenameRequest(BaseModel):
    new_alias: str


def _map_service_errors(app: FastAPI) -> None:
    from fastapi.requests import Request
    from fastapi.responses import JSONResponse
    from roboclaw.embodied.service import EmbodimentBusyError

    @app.exception_handler(EmbodimentBusyError)
    async def _busy_error(request: Request, exc: EmbodimentBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})


def register_device_routes(app: FastAPI, service: Any) -> None:
    """Register /api/devices/* routes on the given app."""
    _map_service_errors(app)

    # -- Catalog ---------------------------------------------------------------

    @app.get("/api/devices/catalog")
    async def devices_catalog() -> dict[str, Any]:
        from roboclaw.embodied.embodiment.catalog import (
            EmbodimentCategory,
            is_supported,
            models_for,
        )

        categories = [
            {"id": cat.value, "supported": is_supported(cat)}
            for cat in EmbodimentCategory
        ]
        models: dict[str, list[dict[str, Any]]] = {}
        for cat in EmbodimentCategory:
            specs = models_for(cat)
            if specs:
                models[cat.value] = [
                    {"name": s.name, "roles": list(s.roles)} for s in specs
                ]
        return {"categories": categories, "models": models}

    # -- Manifest snapshot -----------------------------------------------------

    @app.get("/api/devices")
    async def devices_list() -> dict[str, Any]:
        return service.manifest.snapshot

    # -- CRUD for arms, cameras, hands -----------------------------------------

    _DEVICE_TYPES = [
        ("arms", "arm"),
        ("cameras", "camera"),
        ("hands", "hand"),
    ]

    for _resource, _kind in _DEVICE_TYPES:
        def _make_rename(kind: str):
            rename_fn = getattr(service, f"rename_{kind}")

            async def handler(alias: str, body: RenameRequest) -> dict[str, str]:
                try:
                    await asyncio.to_thread(rename_fn, alias, body.new_alias)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                return {"status": "renamed", "old": alias, "new": body.new_alias}

            handler.__name__ = f"devices_rename_{kind}"
            return handler

        def _make_remove(kind: str):
            unbind_fn = getattr(service, f"unbind_{kind}")

            async def handler(alias: str) -> dict[str, str]:
                try:
                    await asyncio.to_thread(unbind_fn, alias)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                return {"status": "removed", "alias": alias}

            handler.__name__ = f"devices_remove_{kind}"
            return handler

        app.patch(f"/api/devices/{_resource}/{{alias}}")(_make_rename(_kind))
        app.delete(f"/api/devices/{_resource}/{{alias}}")(_make_remove(_kind))
