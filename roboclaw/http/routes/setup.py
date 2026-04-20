"""Setup wizard REST API routes — session-based discovery workflow.

Device CRUD has moved to devices.py (/api/devices/*).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.http.routes._previews import serve_preview_image

SETUP_PREVIEW_DIR = Path("/tmp/roboclaw-camera-previews/setup")


class ScanRequest(BaseModel):
    model: str = ""


class AssignRequest(BaseModel):
    interface_stable_id: str
    alias: str
    spec_name: str
    side: str = ""


class DismissRequest(BaseModel):
    interface_stable_id: str


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

    # -- Permissions ------------------------------------------------------------

    @app.get("/api/setup/permissions")
    async def setup_permissions() -> dict[str, Any]:
        from roboclaw.embodied.embodiment.hardware.scan import check_device_permissions
        return await asyncio.to_thread(check_device_permissions)

    @app.post("/api/setup/permissions/fix")
    async def setup_permissions_fix() -> dict[str, Any]:
        import os
        from roboclaw.embodied.embodiment.hardware.scan import (
            check_device_permissions, fix_serial_permissions,
        )
        fixed = await asyncio.to_thread(fix_serial_permissions)
        status = await asyncio.to_thread(check_device_permissions)
        status["fixed"] = fixed
        if not fixed:
            user = os.environ.get("USER", "$USER")
            status["hint"] = f"sudo usermod -aG dialout,video {user}"
        return status

    # -- Scan -------------------------------------------------------------------

    @app.post("/api/setup/scan")
    async def setup_scan(body: ScanRequest | None = None) -> dict[str, Any]:
        model = body.model if body else ""
        try:
            result = await asyncio.to_thread(service.setup.run_full_scan, model)
            return {
                "ports": [{"stable_id": p.stable_id, **p.to_dict()} for p in result["ports"]],
                "cameras": [{"stable_id": c.stable_id, **c.to_dict()} for c in result["cameras"]],
            }
        except PermissionError:
            import os
            raise HTTPException(403, {
                "code": "serialPermissionDenied",
                "user": os.environ.get("USER", "?"),
            })

    @app.post("/api/setup/previews")
    async def setup_camera_previews() -> list[dict]:
        from roboclaw.embodied.service import EmbodimentBusyError

        output_dir = str(SETUP_PREVIEW_DIR)
        try:
            previews = await asyncio.to_thread(
                service.setup.capture_previews, output_dir,
            )
            return [
                {
                    **preview,
                    "preview_url": f"/api/setup/previews/by-key/{preview['preview_key']}",
                }
                for preview in previews
            ]
        except EmbodimentBusyError:
            raise
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/setup/previews/by-key/{preview_key}")
    async def setup_camera_preview_image(preview_key: str):
        return serve_preview_image(SETUP_PREVIEW_DIR, preview_key)

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

    @app.post("/api/setup/session/dismiss")
    async def setup_dismiss(body: DismissRequest) -> dict[str, str]:
        try:
            service.setup.dismiss(body.interface_stable_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "dismissed", "interface_stable_id": body.interface_stable_id}

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
