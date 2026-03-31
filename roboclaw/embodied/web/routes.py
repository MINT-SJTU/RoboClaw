"""FastAPI routes for the web-based data collection UI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel

from roboclaw.embodied.setup import load_setup
from roboclaw.embodied.web.datasets import delete_dataset, get_dataset_info, list_datasets

router = APIRouter(prefix="/api/embodied")


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


@router.post("/record/save-episode")
async def record_save_episode() -> dict[str, str]:
    """Save current episode and start next one."""
    _session().save_episode()
    return {"status": "episode_saved"}


@router.post("/record/discard-episode")
async def record_discard_episode() -> dict[str, str]:
    """Discard current episode and rerecord."""
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
# Servo positions
# ---------------------------------------------------------------------------

@router.get("/servo-positions")
async def servo_positions() -> dict[str, Any]:
    """Read raw positions of all servo motors on all follower arms.

    Only works when NOT teleoperating/recording (serial port would be busy).
    Acquires session lock to prevent race with start_teleop/start_recording.
    """
    session = _session()
    if session.cameras_locked:
        return {"error": "busy", "arms": {}}
    return await asyncio.to_thread(_read_servo_positions_locked, session)


def _read_servo_positions_locked(session: Any) -> dict[str, Any]:
    with session._lock:
        if session.cameras_locked:
            return {"error": "busy", "arms": {}}
        return _read_servo_positions()


def _read_servo_positions() -> dict[str, Any]:
    from roboclaw.embodied.setup import load_setup
    setup = load_setup()
    arms = setup.get("arms", [])
    result: dict[str, Any] = {"error": None, "arms": {}}
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    for arm in arms:
        if "follower" not in arm.get("type", ""):
            continue
        alias = arm.get("alias", "")
        port = arm.get("port", "")
        if not port:
            continue
        try:
            from lerobot.motors.feetech import FeetechMotorsBus
            from lerobot.motors.motors_bus import Motor, MotorNormMode

            motors = {
                name: Motor(id=i + 1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100)
                for i, name in enumerate(motor_names)
            }
            bus = FeetechMotorsBus(port=port, motors=motors)
            bus.connect()
            positions = {}
            for name in motor_names:
                try:
                    positions[name] = int(bus.read("Present_Position", name, normalize=False))
                except Exception:
                    positions[name] = None
            bus.disconnect()
            result["arms"][alias] = positions
        except Exception as e:
            result["arms"][alias] = {"error": str(e)}
    return result


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
