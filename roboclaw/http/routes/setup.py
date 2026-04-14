"""Setup wizard REST API routes — session-based discovery workflow.

Device CRUD has moved to devices.py (/api/devices/*).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ScanRequest(BaseModel):
    model: str = ""


class AssignRequest(BaseModel):
    interface_stable_id: str
    alias: str
    spec_name: str
    side: str = ""


def _map_service_errors(app: FastAPI) -> None:
    """Map EmbodimentBusyError to 409 Conflict."""
    from fastapi.requests import Request
    from fastapi.responses import JSONResponse

    from roboclaw.embodied.service import EmbodimentBusyError

    @app.exception_handler(EmbodimentBusyError)
    async def _busy_error(request: Request, exc: EmbodimentBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})


def register_setup_routes(app: FastAPI, service: Any) -> None:
    """Register /api/setup/* routes on the given app."""
    _map_service_errors(app)

    @app.post("/api/setup/scan")
    async def setup_scan(body: ScanRequest | None = None) -> dict[str, Any]:
        model = body.model if body else ""
        try:
            result = await asyncio.to_thread(service.setup.run_full_scan, model)
            return {
                "ports": [{"stable_id": p.stable_id, **p.to_dict()} for p in result["ports"]],
                "cameras": [{"stable_id": c.stable_id, **c.to_dict()} for c in result["cameras"]],
            }
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/api/setup/previews")
    async def setup_camera_previews() -> list[dict]:
        from roboclaw.embodied.service import EmbodimentBusyError

        output_dir = str(Path("/tmp/roboclaw-camera-previews"))
        try:
            return await asyncio.to_thread(
                service.setup.capture_previews, output_dir,
            )
        except EmbodimentBusyError:
            raise
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/setup/previews/{index}")
    async def setup_camera_preview_image(index: int):
        from fastapi.responses import FileResponse

        preview_dir = Path("/tmp/roboclaw-camera-previews")
        for f in preview_dir.glob(f"{index:02d}_*.jpg"):
            return FileResponse(str(f), media_type="image/jpeg")
        raise HTTPException(404, f"Preview not found for camera index {index}")

    @app.post("/api/setup/motion/start")
    async def motion_start() -> dict[str, Any]:
        from roboclaw.embodied.service import EmbodimentBusyError

        try:
            port_count = await asyncio.to_thread(
                service.setup.start_motion_detection,
            )
        except EmbodimentBusyError:
            raise
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "watching", "port_count": port_count}

    @app.get("/api/setup/motion/poll")
    async def motion_poll() -> dict[str, Any]:
        try:
            results = await asyncio.to_thread(service.setup.poll_motion)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ports": results}

    @app.post("/api/setup/motion/stop")
    async def motion_stop() -> dict[str, str]:
        await asyncio.to_thread(service.setup.stop_motion_detection)
        return {"status": "stopped"}

    # -- SetupSession assign/commit ------------------------------------------

    @app.get("/api/setup/session")
    async def setup_session_status() -> dict[str, Any]:
        return service.setup.to_dict()

    @app.post("/api/setup/session/assign")
    async def setup_assign(body: AssignRequest) -> dict[str, Any]:
        try:
            assignment = service.setup.assign(
                body.interface_stable_id, body.alias, body.spec_name,
                side=body.side,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "status": "assigned",
            "alias": assignment.alias,
            "spec_name": assignment.spec_name,
        }

    @app.delete("/api/setup/session/assign/{alias}")
    async def setup_unassign(alias: str) -> dict[str, str]:
        try:
            service.setup.unassign(alias)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "unassigned", "alias": alias}

    @app.post("/api/setup/session/commit")
    async def setup_commit() -> dict[str, Any]:
        try:
            count = await asyncio.to_thread(service.setup.commit)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "committed", "bindings_created": count}

    @app.post("/api/setup/session/reset")
    async def setup_reset() -> dict[str, str]:
        service.setup.reset()
        return {"status": "reset"}
