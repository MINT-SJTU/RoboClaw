"""Setup wizard REST API routes for hardware discovery and configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.identify import (
    MOTION_THRESHOLD,
    _filter_feetech_ports,
    _resolve_port_by_id,
    _resolve_port_path,
    detect_motion,
    read_positions,
)
from roboclaw.embodied.scan import capture_camera_frames, scan_cameras, scan_serial_ports
from roboclaw.embodied.setup import (
    load_setup,
    remove_arm,
    remove_camera,
    rename_arm,
    set_arm,
    set_camera,
)


class SetupWizardState:
    """Transient state for the setup wizard session."""

    def __init__(self) -> None:
        self.scanned_ports: list[dict] = []
        self.scanned_cameras: list[dict] = []
        self.baselines: dict[str, dict[int, int]] = {}
        self.active = False  # True when motion detection is running


# Request models


class AddArmRequest(BaseModel):
    alias: str
    arm_type: str
    port_id: str


class AddCameraRequest(BaseModel):
    alias: str
    camera_index: int


class RenameRequest(BaseModel):
    new_alias: str


def register_setup_routes(app: FastAPI) -> None:
    """Register setup wizard API endpoints."""

    wizard = SetupWizardState()
    app.state.setup_wizard = wizard

    def _require_not_busy() -> None:
        session = getattr(app.state, "dashboard_session", None)
        if session is not None and session.busy:
            raise HTTPException(409, "Session busy — stop teleop/recording first")

    @app.post("/api/dashboard/setup/scan")
    async def setup_scan() -> dict[str, Any]:
        _require_not_busy()
        ports = await asyncio.to_thread(_scan_and_probe)
        cameras = await asyncio.to_thread(scan_cameras)
        wizard.scanned_ports = ports
        wizard.scanned_cameras = cameras
        wizard.active = False
        wizard.baselines = {}
        return {"ports": ports, "cameras": cameras}

    @app.post("/api/dashboard/setup/camera-previews")
    async def setup_camera_previews() -> list[dict]:
        _require_not_busy()
        if not wizard.scanned_cameras:
            raise HTTPException(400, "No cameras scanned. Run scan first.")
        output_dir = Path("/tmp/roboclaw-camera-previews")
        previews = await asyncio.to_thread(
            capture_camera_frames, wizard.scanned_cameras, output_dir,
        )
        return previews

    @app.get("/api/dashboard/setup/camera-preview/{index}")
    async def setup_camera_preview_image(index: int):
        from fastapi.responses import FileResponse

        preview_dir = Path("/tmp/roboclaw-camera-previews")
        for f in preview_dir.glob(f"{index:02d}_*.jpg"):
            return FileResponse(str(f), media_type="image/jpeg")
        raise HTTPException(404, f"Preview not found for camera index {index}")

    @app.post("/api/dashboard/setup/motion/start")
    async def motion_start() -> dict[str, Any]:
        _require_not_busy()
        if not wizard.scanned_ports:
            raise HTTPException(400, "No scanned ports. Run scan first.")
        baselines = await asyncio.to_thread(
            _read_all_baselines, wizard.scanned_ports,
        )
        wizard.baselines = baselines
        wizard.active = True
        return {"status": "watching", "port_count": len(baselines)}

    @app.get("/api/dashboard/setup/motion/poll")
    async def motion_poll() -> dict[str, Any]:
        if not wizard.active or not wizard.baselines:
            raise HTTPException(400, "Motion detection not started")
        _require_not_busy()
        results = await asyncio.to_thread(
            _poll_motion, wizard.scanned_ports, wizard.baselines,
        )
        return {"ports": results}

    @app.post("/api/dashboard/setup/motion/stop")
    async def motion_stop() -> dict[str, str]:
        wizard.active = False
        wizard.baselines = {}
        return {"status": "stopped"}

    @app.post("/api/dashboard/setup/arm")
    async def setup_add_arm(body: AddArmRequest) -> dict[str, Any]:
        await asyncio.to_thread(set_arm, body.alias, body.arm_type, body.port_id)
        return {"status": "added", "alias": body.alias}

    @app.delete("/api/dashboard/setup/arm/{alias}")
    async def setup_remove_arm(alias: str) -> dict[str, str]:
        await asyncio.to_thread(remove_arm, alias)
        return {"status": "removed", "alias": alias}

    @app.patch("/api/dashboard/setup/arm/{alias}/rename")
    async def setup_rename_arm(alias: str, body: RenameRequest) -> dict[str, str]:
        await asyncio.to_thread(rename_arm, alias, body.new_alias)
        return {"status": "renamed", "old": alias, "new": body.new_alias}

    @app.post("/api/dashboard/setup/camera")
    async def setup_add_camera(body: AddCameraRequest) -> dict[str, Any]:
        await asyncio.to_thread(set_camera, body.alias, body.camera_index)
        return {"status": "added", "alias": body.alias}

    @app.delete("/api/dashboard/setup/camera/{alias}")
    async def setup_remove_camera(alias: str) -> dict[str, str]:
        await asyncio.to_thread(remove_camera, alias)
        return {"status": "removed", "alias": alias}

    @app.get("/api/dashboard/setup/current")
    async def setup_current() -> dict[str, Any]:
        setup = load_setup()
        return {
            "arms": setup.get("arms", []),
            "cameras": setup.get("cameras", []),
            "hands": setup.get("hands", []),
        }


def _scan_and_probe() -> list[dict]:
    ports = scan_serial_ports()
    try:
        return _filter_feetech_ports(ports)
    except Exception as exc:
        if "Permission denied" not in str(exc) and "Errno 13" not in str(exc):
            raise
        # Auto-fix: install udev rules and retry
        if _try_fix_serial_permissions():
            return _filter_feetech_ports(ports)
        raise HTTPException(
            status_code=403,
            detail="Serial port permission denied. Run: bash scripts/setup-udev.sh",
        ) from exc


def _try_fix_serial_permissions() -> bool:
    """Attempt to install udev rules for serial device access. Returns True on success."""
    import subprocess

    from loguru import logger

    # Try passwordless sudo first
    udev_rule = (
        'KERNEL=="ttyACM[0-9]*", MODE="0666"\n'
        'KERNEL=="ttyUSB[0-9]*", MODE="0666"\n'
        'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666"\n'
        'SUBSYSTEM=="video4linux", MODE="0666"\n'
    )
    try:
        # Write rule and reload
        result = subprocess.run(
            ["sudo", "-n", "tee", "/etc/udev/rules.d/99-roboclaw.rules"],
            input=udev_rule.encode(), capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("Passwordless sudo not available for udev rules")
            return _try_chmod_devices()
        subprocess.run(["sudo", "-n", "udevadm", "control", "--reload-rules"],
                       capture_output=True, timeout=5)
        subprocess.run(["sudo", "-n", "udevadm", "trigger"],
                       capture_output=True, timeout=5)
        logger.info("Installed udev rules for serial device access")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _try_chmod_devices()


def _try_chmod_devices() -> bool:
    """Fallback: chmod individual device files."""
    import os
    import subprocess

    from loguru import logger

    from roboclaw.embodied.scan import list_serial_device_paths
    devices = list_serial_device_paths()
    if not devices:
        return False
    for dev in devices:
        try:
            os.chmod(dev, 0o666)
        except PermissionError:
            # Try sudo chmod
            result = subprocess.run(
                ["sudo", "-n", "chmod", "666", dev],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                logger.warning("Cannot chmod {}: no passwordless sudo", dev)
                return False
    logger.info("Fixed serial device permissions via chmod")
    return True


def _read_all_baselines(ports: list[dict]) -> dict[str, dict[int, int]]:
    baselines: dict[str, dict[int, int]] = {}
    for port in ports:
        path = _resolve_port_path(port)
        baselines[path] = read_positions(path, port["motor_ids"])
    return baselines


def _poll_motion(
    ports: list[dict], baselines: dict[str, dict[int, int]],
) -> list[dict]:
    results = []
    for port in ports:
        path = _resolve_port_path(port)
        current = read_positions(path, port["motor_ids"])
        baseline = baselines.get(path, {})
        delta = detect_motion(baseline, current)
        results.append({
            "port_id": _resolve_port_by_id(port),
            "dev": port.get("dev", ""),
            "by_id": port.get("by_id", ""),
            "motor_ids": port.get("motor_ids", []),
            "delta": delta,
            "moved": delta > MOTION_THRESHOLD,
        })
    return results
