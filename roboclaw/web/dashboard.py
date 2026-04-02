"""Data Collection Dashboard — consolidated REST API routes.

All dashboard endpoints live under ``/api/dashboard/``.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from roboclaw.embodied.hardware_monitor import HardwareMonitor
from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.setup import load_setup
from roboclaw.web.dashboard_datasets import delete_dataset, get_dataset_info, list_datasets
from roboclaw.web.troubleshooting import generate_fault_snapshot, get_troubleshoot_map_json


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RecordStartRequest(BaseModel):
    task: str
    num_episodes: int = 10
    fps: int = 30
    episode_time_s: int = 300
    reset_time_s: int = 10


class TeleopStartRequest(BaseModel):
    fps: int = 30


class RecheckRequest(BaseModel):
    fault_type: str
    device_alias: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        logger.warning("Failed to detect LAN IP, falling back to 127.0.0.1")
        return "127.0.0.1"


def _datasets_root() -> Path:
    from roboclaw.embodied.ops.helpers import dataset_root
    return dataset_root(load_setup())


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_dashboard_routes(
    app: FastAPI,
    web_channel: Any,
    service: EmbodiedService,
    get_config: Callable[[], tuple[str, int]],
) -> None:
    """Register all dashboard API endpoints on the FastAPI app."""

    # -- Hardware status ---------------------------------------------------

    @app.get("/api/dashboard/hardware-status")
    async def hardware_status() -> dict[str, Any]:
        return service.get_hardware_status()

    # -- Session lifecycle -------------------------------------------------

    @app.get("/api/dashboard/session/status")
    async def session_status() -> dict[str, Any]:
        return service.get_status()

    @app.post("/api/dashboard/session/teleop/start")
    async def teleop_start(body: TeleopStartRequest | None = None) -> dict[str, str]:
        fps = body.fps if body else 30
        await service.start_teleop(fps=fps)
        return {"status": "teleoperating"}

    @app.post("/api/dashboard/session/teleop/stop")
    async def teleop_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}

    @app.post("/api/dashboard/session/record/start")
    async def record_start(body: RecordStartRequest) -> dict[str, Any]:
        dataset_name = await service.start_recording(
            task=body.task,
            num_episodes=body.num_episodes,
            fps=body.fps,
            episode_time_s=body.episode_time_s,
            reset_time_s=body.reset_time_s,
        )
        return {"status": "recording", "dataset_name": dataset_name}

    @app.post("/api/dashboard/session/record/stop")
    async def record_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}

    # -- Episode control ---------------------------------------------------

    @app.post("/api/dashboard/session/episode/save")
    async def episode_save() -> dict[str, str]:
        await service.save_episode()
        return {"status": "episode_saved"}

    @app.post("/api/dashboard/session/episode/discard")
    async def episode_discard() -> dict[str, str]:
        await service.discard_episode()
        return {"status": "episode_discarded"}

    @app.post("/api/dashboard/session/episode/skip-reset")
    async def episode_skip_reset() -> dict[str, str]:
        await service.skip_reset()
        return {"status": "reset_skipped"}

    # -- Datasets ----------------------------------------------------------

    @app.get("/api/dashboard/datasets")
    async def datasets_list_route() -> list[dict]:
        return await asyncio.to_thread(list_datasets, _datasets_root())

    @app.get("/api/dashboard/datasets/{name}")
    async def dataset_detail(name: str) -> dict:
        info = await asyncio.to_thread(get_dataset_info, _datasets_root(), name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        return info

    @app.delete("/api/dashboard/datasets/{name}")
    async def dataset_delete(name: str) -> dict[str, str]:
        await asyncio.to_thread(delete_dataset, _datasets_root(), name)
        return {"status": "deleted", "name": name}

    # -- Servo positions ---------------------------------------------------

    @app.get("/api/dashboard/servo-positions")
    async def servo_positions() -> dict[str, Any]:
        if service.busy:
            return {"error": "busy", "arms": {}}
        from roboclaw.embodied.motors import read_servo_positions
        return await asyncio.to_thread(read_servo_positions)

    # -- Troubleshooting ---------------------------------------------------

    @app.get("/api/dashboard/troubleshoot-map")
    async def troubleshoot_map() -> dict[str, Any]:
        return get_troubleshoot_map_json()

    @app.post("/api/dashboard/troubleshoot/recheck")
    async def troubleshoot_recheck(body: RecheckRequest) -> dict[str, Any]:
        monitor: HardwareMonitor = app.state.hardware_monitor
        faults = monitor.check_hardware()
        return {"faults": [f.to_dict() for f in faults]}

    @app.post("/api/dashboard/troubleshoot/snapshot")
    async def troubleshoot_snapshot() -> dict[str, Any]:
        setup = load_setup()
        monitor: HardwareMonitor = app.state.hardware_monitor
        faults = monitor.active_faults
        return generate_fault_snapshot(setup, faults, "")

    # -- Network info ------------------------------------------------------

    @app.get("/api/dashboard/network-info")
    async def network_info() -> dict[str, Any]:
        host, port = get_config()
        return {"host": host, "port": port, "lan_ip": _get_lan_ip()}

    # -- Setup wizard routes -----------------------------------------------

    from roboclaw.web.dashboard_setup import register_setup_routes
    register_setup_routes(app)

    # -- Calibration routes ------------------------------------------------

    from roboclaw.web.dashboard_calibrate import register_calibrate_routes
    register_calibrate_routes(app)
