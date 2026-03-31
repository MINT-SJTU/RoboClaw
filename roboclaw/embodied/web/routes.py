"""FastAPI routes for the web-based data collection UI."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel

from roboclaw.embodied.setup import load_setup
from roboclaw.embodied.web.datasets import delete_dataset, get_dataset_info, list_datasets

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Lazy session accessor (avoid circular import at module level)
# ---------------------------------------------------------------------------

def _session():
    from roboclaw.embodied.web.app import get_session
    return get_session()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RecordStartRequest(BaseModel):
    dataset_name: str
    task: str
    fps: int = 30
    num_episodes: int = 10


class TeleopStartRequest(BaseModel):
    fps: int = 30


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def status() -> dict[str, Any]:
    """Return current session state, episode info, etc."""
    return _session().get_status()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@router.post("/connect")
async def connect() -> dict[str, str]:
    """Connect robot using setup.json config."""
    setup = load_setup()
    _session().connect(setup)
    return {"status": "connected"}


@router.post("/disconnect")
async def disconnect() -> dict[str, str]:
    """Disconnect robot."""
    _session().disconnect()
    return {"status": "disconnected"}


# ---------------------------------------------------------------------------
# Teleoperation
# ---------------------------------------------------------------------------

@router.post("/teleop/start")
async def teleop_start(body: TeleopStartRequest | None = None) -> dict[str, str]:
    """Start teleoperation."""
    fps = body.fps if body else 30
    _session().start_teleop(fps=fps)
    return {"status": "teleoperating"}


@router.post("/teleop/stop")
async def teleop_stop() -> dict[str, str]:
    """Stop teleoperation."""
    _session().stop_teleop()
    return {"status": "connected"}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

@router.post("/record/start")
async def record_start(body: RecordStartRequest) -> dict[str, str]:
    """Start recording."""
    _session().start_recording(
        dataset_name=body.dataset_name,
        task=body.task,
        fps=body.fps,
        num_episodes=body.num_episodes,
    )
    return {"status": "recording"}


@router.post("/record/stop")
async def record_stop() -> dict[str, str]:
    """Stop recording."""
    _session().stop_recording()
    return {"status": "connected"}


@router.post("/record/save")
async def record_save() -> dict[str, str]:
    """Save current episode."""
    _session().save_episode()
    return {"status": "episode_saved", "episode": _session().get_status()["episode_count"]}


@router.post("/record/discard")
async def record_discard() -> dict[str, str]:
    """Discard current episode."""
    _session().discard_episode()
    return {"status": "episode_discarded"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _datasets_root() -> Path:
    setup = load_setup()
    root = setup.get("datasets", {}).get("root", "")
    if not root:
        raise ValueError("No datasets root configured in setup.json")
    return Path(root)


@router.get("/datasets")
async def datasets_list() -> list[dict]:
    """List datasets from setup.json datasets root."""
    return list_datasets(_datasets_root())


@router.get("/datasets/{name}")
async def dataset_detail(name: str) -> dict:
    """Get detailed info for a single dataset."""
    info = get_dataset_info(_datasets_root(), name)
    if info is None:
        raise ValueError(f"Dataset '{name}' not found")
    return info


@router.delete("/datasets/{name}")
async def dataset_delete(name: str) -> dict[str, str]:
    """Delete a dataset."""
    delete_dataset(_datasets_root(), name)
    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

@router.get("/cameras")
async def cameras_discover() -> list[dict]:
    """Discover available cameras via hardware scan."""
    from roboclaw.embodied.scan import scan_cameras
    return scan_cameras()


# ---------------------------------------------------------------------------
# WebSocket: camera stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/camera")
async def ws_camera(websocket: WebSocket):
    """Stream JPEG camera frames as base64-encoded JSON messages.

    Client sends: {"camera": "<camera_name>", "fps": 10}
    Server streams: {"camera": "<name>", "frame": "<base64 jpeg>"}
    """
    await websocket.accept()
    try:
        init_msg = await websocket.receive_text()
        params = json.loads(init_msg)
        camera_name = params.get("camera", "")
        fps = min(params.get("fps", 10), 30)
        dt = 1.0 / fps

        while True:
            start = asyncio.get_event_loop().time()
            try:
                frame_bytes = _session().get_camera_frame(camera_name)
                b64 = base64.b64encode(frame_bytes).decode("ascii")
                await websocket.send_json({"camera": camera_name, "frame": b64})
            except (KeyError, RuntimeError):
                await websocket.send_json({"camera": camera_name, "error": "no_frame"})

            elapsed = asyncio.get_event_loop().time() - start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    except WebSocketDisconnect:
        logger.debug("Camera WebSocket disconnected")


# ---------------------------------------------------------------------------
# WebSocket: status stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Stream session status at ~5 Hz."""
    await websocket.accept()
    try:
        while True:
            status = _session().get_status()
            await websocket.send_json(status)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        logger.debug("Status WebSocket disconnected")
