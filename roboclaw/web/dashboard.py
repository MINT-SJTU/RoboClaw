"""Data Collection Dashboard REST API routes."""

from __future__ import annotations

import asyncio
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel

from roboclaw.embodied.hardware_monitor import HardwareMonitor
from roboclaw.embodied.recording import RecordingSession, RecordingStatus
from roboclaw.embodied.setup import load_setup
from roboclaw.web.troubleshooting import generate_fault_snapshot, get_troubleshoot_map_json


class RecordingStartRequest(BaseModel):
    task: str
    num_episodes: int = 10
    episode_time_s: int = 60
    reset_time_s: int = 10


class RecheckRequest(BaseModel):
    fault_type: str
    device_alias: str


def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        logger.warning("Failed to detect LAN IP, falling back to 127.0.0.1")
        return "127.0.0.1"


def _check_arm_status(arm: dict[str, Any]) -> dict[str, Any]:
    alias = arm.get("alias", "unknown")
    port = arm.get("port", "")
    connected = bool(port and Path(port).exists())
    calibrated = bool(arm.get("calibrated", False))
    arm_type = arm.get("type", "")
    role = "follower" if "follower" in arm_type else "leader" if "leader" in arm_type else ""
    return {
        "alias": alias,
        "type": arm_type,
        "role": role,
        "connected": connected,
        "calibrated": calibrated,
    }


def _check_camera_status(cam: dict[str, Any]) -> dict[str, Any]:
    alias = cam.get("alias", "unknown")
    port = cam.get("port", "")
    connected = bool(port and Path(port).exists())
    return {
        "alias": alias,
        "connected": connected,
        "width": cam.get("width", 640),
        "height": cam.get("height", 480),
    }


def _compute_readiness(
    arms: list[dict[str, Any]],
    arm_statuses: list[dict[str, Any]],
    camera_statuses: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    from roboclaw.embodied.ops.helpers import _group_arms

    missing: list[str] = []
    grouped = _group_arms(arms)
    followers = grouped["followers"]
    leaders = grouped["leaders"]

    if not followers:
        missing.append("No follower arm configured")
    if not leaders:
        missing.append("No leader arm configured")

    for status in arm_statuses:
        if not status["connected"]:
            missing.append(f"Arm '{status['alias']}' is disconnected")
        elif not status["calibrated"]:
            missing.append(f"Arm '{status['alias']}' is not calibrated")

    for status in camera_statuses:
        if not status["connected"]:
            missing.append(f"Camera '{status['alias']}' is disconnected")

    if followers and leaders and len(followers) != len(leaders):
        missing.append(
            f"Follower/leader count mismatch: {len(followers)} vs {len(leaders)}"
        )

    return len(missing) == 0, missing


def _capture_preview_bytes(cam: dict[str, Any]) -> bytes:
    """Open camera, grab one JPEG frame. Runs in a thread — blocking I/O."""
    import cv2

    port = cam.get("port", "")
    cap = cv2.VideoCapture(port)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail=f"Cannot open camera at {port}")
    try:
        width = cam.get("width")
        if isinstance(width, int) and width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        height = cam.get("height")
        if isinstance(height, int) and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fourcc = cam.get("fourcc")
        if isinstance(fourcc, str) and len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=503, detail="Failed to capture frame")
        success, jpeg_buf = cv2.imencode(".jpg", frame)
        if not success:
            raise HTTPException(status_code=500, detail="JPEG encoding failed")
        return jpeg_buf.tobytes()
    finally:
        cap.release()


def _build_recording_argv(
    setup: dict[str, Any], params: RecordingStartRequest,
) -> tuple[list[str], str, str, list[str]]:
    """Build LeRobot record CLI argv from dashboard params.

    Returns (argv, dataset_name, dataset_root, temp_dirs_to_cleanup).
    """
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.ops.helpers import (
        _BIMANUAL_ID,
        _arm_id,
        _dataset_path,
        _group_arms,
        _stage_bimanual_arm_pair,
        _validate_pairing,
    )
    from roboclaw.embodied.sensor.camera import resolve_cameras

    arms = setup.get("arms", [])
    grouped = _group_arms(arms)
    followers = grouped["followers"]
    leaders = grouped["leaders"]

    error = _validate_pairing(followers, leaders)
    if error:
        raise HTTPException(status_code=400, detail=error)

    dataset_name = f"rec_{datetime.now():%Y%m%d_%H%M%S}"
    cameras = resolve_cameras(setup)

    record_kwargs: dict[str, Any] = {
        "cameras": cameras,
        "repo_id": f"local/{dataset_name}",
        "task": params.task,
        "dataset_root": str(_dataset_path(setup, dataset_name)),
        "push_to_hub": False,
        "fps": 30,
        "num_episodes": params.num_episodes,
        "episode_time_s": params.episode_time_s,
        "reset_time_s": params.reset_time_s,
    }
    dataset_root = record_kwargs["dataset_root"]
    controller = SO101Controller()

    if len(followers) == 1:
        argv = controller.record(
            robot_type=followers[0]["type"],
            robot_port=followers[0]["port"],
            robot_cal_dir=followers[0]["calibration_dir"],
            robot_id=_arm_id(followers[0]),
            teleop_type=leaders[0]["type"],
            teleop_port=leaders[0]["port"],
            teleop_cal_dir=leaders[0]["calibration_dir"],
            teleop_id=_arm_id(leaders[0]),
            **record_kwargs,
        )
        return argv, dataset_name, dataset_root, []

    # Bimanual: persistent temp dirs, cleaned up on recording completion
    import tempfile

    robot_dir = tempfile.mkdtemp(prefix="roboclaw-bimanual-robot-")
    teleop_dir = tempfile.mkdtemp(prefix="roboclaw-bimanual-teleop-")
    _stage_bimanual_arm_pair(followers[0], followers[1], robot_dir)
    _stage_bimanual_arm_pair(leaders[0], leaders[1], teleop_dir)

    argv = controller.record_bimanual(
        robot_id=_BIMANUAL_ID,
        robot_cal_dir=robot_dir,
        left_robot=followers[0],
        right_robot=followers[1],
        teleop_id=_BIMANUAL_ID,
        teleop_cal_dir=teleop_dir,
        left_teleop=leaders[0],
        right_teleop=leaders[1],
        **record_kwargs,
    )
    return argv, dataset_name, dataset_root, [robot_dir, teleop_dir]


def register_dashboard_routes(
    app: FastAPI,
    web_channel: Any,
    get_config: Callable[[], tuple[str, int]],
) -> None:
    """Register all dashboard API endpoints on the FastAPI app."""

    app.state.active_recording = None
    app.state.recording_temp_dirs: list[str] = []

    def _cleanup_recording() -> None:
        """Reset recording state and clean up temp dirs."""
        app.state.active_recording = None
        app.state.hardware_monitor.set_recording_active(False)
        for d in app.state.recording_temp_dirs:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        app.state.recording_temp_dirs = []

    async def _on_progress(status: RecordingStatus) -> None:
        await web_channel.broadcast_dashboard_event({
            "type": "dashboard.recording.progress",
            **status.to_dict(),
        })

    async def _on_recording_end(event_type: str, status: RecordingStatus) -> None:
        _cleanup_recording()
        await web_channel.broadcast_dashboard_event({
            "type": event_type,
            **status.to_dict(),
        })

    @app.get("/api/dashboard/hardware-status")
    async def hardware_status() -> dict[str, Any]:
        setup = load_setup()
        arms = setup.get("arms", [])
        cameras = setup.get("cameras", [])
        arm_statuses = [_check_arm_status(a) for a in arms]
        camera_statuses = [_check_camera_status(c) for c in cameras]
        ready, missing = _compute_readiness(arms, arm_statuses, camera_statuses)
        recording = app.state.active_recording
        return {
            "ready": ready,
            "missing": missing,
            "arms": arm_statuses,
            "cameras": camera_statuses,
            "recording_active": recording is not None and recording.active,
        }

    @app.get("/api/dashboard/camera-preview/{alias}")
    async def camera_preview(alias: str) -> Response:
        recording = app.state.active_recording
        if recording is not None and recording.active:
            raise HTTPException(
                status_code=409,
                detail="Cannot preview camera while recording is active",
            )
        setup = load_setup()
        cam = next((c for c in setup.get("cameras", []) if c.get("alias") == alias), None)
        if cam is None:
            raise HTTPException(status_code=404, detail=f"Camera '{alias}' not found")
        port = cam.get("port", "")
        if not port:
            raise HTTPException(status_code=400, detail=f"Camera '{alias}' has no port")
        jpeg_bytes = await asyncio.to_thread(_capture_preview_bytes, cam)
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    @app.post("/api/dashboard/recording/start")
    async def recording_start(body: RecordingStartRequest) -> dict[str, Any]:
        recording = app.state.active_recording
        if recording is not None and recording.active:
            raise HTTPException(
                status_code=409, detail="A recording session is already active",
            )
        setup = load_setup()
        argv, dataset_name, dataset_root, temp_dirs = _build_recording_argv(setup, body)
        app.state.recording_temp_dirs = temp_dirs
        session = RecordingSession(
            argv=argv,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            task=body.task,
            total_episodes=body.num_episodes,
            on_progress=_on_progress,
            on_completed=lambda s: _on_recording_end("dashboard.recording.completed", s),
            on_error=lambda s: _on_recording_end("dashboard.recording.error", s),
        )
        await session.start()
        app.state.active_recording = session
        app.state.hardware_monitor.set_recording_active(True)
        logger.info(
            "Dashboard recording started: session={}, dataset={}",
            session.session_id, dataset_name,
        )
        return {"session_id": session.session_id, "dataset_name": dataset_name}

    @app.post("/api/dashboard/recording/stop")
    async def recording_stop() -> dict[str, str]:
        recording = app.state.active_recording
        if recording is None or not recording.active:
            raise HTTPException(status_code=404, detail="No active recording session")
        recording.stop()
        return {"status": "stopping"}

    @app.get("/api/dashboard/recording/status")
    async def recording_status() -> dict[str, Any]:
        recording = app.state.active_recording
        if recording is None:
            return {"active": False}
        return {"active": recording.active, **recording.status.to_dict()}

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
        recording = app.state.active_recording
        stderr_tail = recording.stderr_tail if recording is not None else ""
        return generate_fault_snapshot(setup, faults, stderr_tail)

    @app.get("/api/dashboard/network-info")
    async def network_info() -> dict[str, Any]:
        host, port = get_config()
        return {"host": host, "port": port, "lan_ip": _get_lan_ip()}
