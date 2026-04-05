"""Setup wizard REST API routes — thin HTTP shell over EmbodiedService."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from roboclaw.embodied.manifest.helpers import _resolve_serial_interface


class ScanRequest(BaseModel):
    model: str = ""


class AddArmRequest(BaseModel):
    alias: str
    arm_type: str
    port_id: str


class AddCameraRequest(BaseModel):
    alias: str
    camera_index: int


class RenameRequest(BaseModel):
    new_alias: str


class AssignRequest(BaseModel):
    interface_stable_id: str
    alias: str
    spec_name: str


def _map_service_errors(app: FastAPI) -> None:
    """Map EmbodimentBusyError to 409 Conflict."""
    from fastapi.requests import Request
    from fastapi.responses import JSONResponse

    from roboclaw.embodied.service import EmbodimentBusyError

    @app.exception_handler(EmbodimentBusyError)
    async def _busy_error(request: Request, exc: EmbodimentBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})


def register_setup_routes(app: FastAPI, service: Any) -> None:
    """Register /api/dashboard/setup/* routes on the given app."""
    _map_service_errors(app)

    @app.post("/api/dashboard/setup/scan")
    async def setup_scan(body: ScanRequest | None = None) -> dict[str, Any]:
        model = body.model if body else ""
        try:
            result = await asyncio.to_thread(service.setup.run_full_scan, model)
            return {
                "ports": [p.to_dict() for p in result["ports"]],
                "cameras": [c.to_dict() for c in result["cameras"]],
            }
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/api/dashboard/setup/camera-previews")
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

    @app.get("/api/dashboard/setup/camera-preview/{index}")
    async def setup_camera_preview_image(index: int):
        from fastapi.responses import FileResponse

        preview_dir = Path("/tmp/roboclaw-camera-previews")
        for f in preview_dir.glob(f"{index:02d}_*.jpg"):
            return FileResponse(str(f), media_type="image/jpeg")
        raise HTTPException(404, f"Preview not found for camera index {index}")

    @app.post("/api/dashboard/setup/motion/start")
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

    @app.get("/api/dashboard/setup/motion/poll")
    async def motion_poll() -> dict[str, Any]:
        try:
            results = await asyncio.to_thread(service.setup.poll_motion)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ports": results}

    @app.post("/api/dashboard/setup/motion/stop")
    async def motion_stop() -> dict[str, str]:
        await asyncio.to_thread(service.setup.stop_motion_detection)
        return {"status": "stopped"}

    # -- SetupSession assign/commit ------------------------------------------

    @app.get("/api/dashboard/setup/session")
    async def setup_session_status() -> dict[str, Any]:
        return service.setup.to_dict()

    @app.post("/api/dashboard/setup/session/assign")
    async def setup_assign(body: AssignRequest) -> dict[str, Any]:
        try:
            assignment = service.setup.assign(
                body.interface_stable_id, body.alias, body.spec_name,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "status": "assigned",
            "alias": assignment.alias,
            "spec_name": assignment.spec_name,
        }

    @app.delete("/api/dashboard/setup/session/assign/{alias}")
    async def setup_unassign(alias: str) -> dict[str, str]:
        try:
            service.setup.unassign(alias)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "unassigned", "alias": alias}

    @app.post("/api/dashboard/setup/session/commit")
    async def setup_commit() -> dict[str, Any]:
        try:
            count = await asyncio.to_thread(service.setup.commit)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "committed", "bindings_created": count}

    # -- Direct arm/camera CRUD (legacy, still useful) -----------------------

    @app.post("/api/dashboard/setup/arm")
    async def setup_add_arm(body: AddArmRequest) -> dict[str, Any]:
        try:
            interface = await asyncio.to_thread(_resolve_serial_interface, body.port_id)
            await asyncio.to_thread(service.bind_arm, body.alias, body.arm_type, interface)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "added", "alias": body.alias}

    @app.delete("/api/dashboard/setup/arm/{alias}")
    async def setup_remove_arm(alias: str) -> dict[str, str]:
        try:
            await asyncio.to_thread(service.unbind_arm, alias)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "removed", "alias": alias}

    @app.patch("/api/dashboard/setup/arm/{alias}/rename")
    async def setup_rename_arm(alias: str, body: RenameRequest) -> dict[str, str]:
        try:
            await asyncio.to_thread(service.rename_arm, alias, body.new_alias)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "renamed", "old": alias, "new": body.new_alias}

    @app.post("/api/dashboard/setup/camera")
    async def setup_add_camera(body: AddCameraRequest) -> dict[str, Any]:
        from roboclaw.embodied.hardware.scan import scan_cameras

        try:
            scanned = await asyncio.to_thread(scan_cameras)
            if body.camera_index < 0 or body.camera_index >= len(scanned):
                raise HTTPException(
                    400,
                    f"camera_index {body.camera_index} out of range. Found {len(scanned)} camera(s).",
                )
            interface = scanned[body.camera_index]
            if not interface.address:
                raise HTTPException(
                    400,
                    f"Scanned camera at index {body.camera_index} has no usable path.",
                )
            await asyncio.to_thread(service.bind_camera, body.alias, interface)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "added", "alias": body.alias}

    @app.delete("/api/dashboard/setup/camera/{alias}")
    async def setup_remove_camera(alias: str) -> dict[str, str]:
        try:
            await asyncio.to_thread(service.unbind_camera, alias)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "removed", "alias": alias}

    @app.get("/api/dashboard/setup/current")
    async def setup_current() -> dict[str, Any]:
        return service.manifest.snapshot
