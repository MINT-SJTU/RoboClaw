"""Recovery routes for active faults and dashboard self-restart."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
from roboclaw.http.recovery import get_recovery_guides_json


class DeviceCheckRequest(BaseModel):
    kind: str
    alias: str


class ArmMotorCheckRequest(BaseModel):
    alias: str


def _pretty_motor_name(name: str) -> str:
    return name.replace("_", " ")


def schedule_dashboard_restart(app: FastAPI, delay_s: float = 0.5) -> None:
    """Restart the dashboard process in-place after *delay_s* seconds.

    The task reference is retained on ``app.state`` so the event loop
    cannot garbage-collect it before execv fires.
    """

    async def _restart() -> None:
        await asyncio.sleep(delay_s)
        logger.info("Restarting dashboard process")
        os.execv(sys.executable, [sys.executable, "-m", "roboclaw", *sys.argv[1:]])

    app.state.restart_task = asyncio.create_task(_restart())


def register_recovery_routes(app: FastAPI) -> None:

    @app.get("/api/recovery/guides")
    async def recovery_guides() -> dict[str, Any]:
        return get_recovery_guides_json()

    @app.get("/api/recovery/faults")
    async def recovery_faults() -> dict[str, Any]:
        monitor: HardwareMonitor = app.state.hardware_monitor
        return {"faults": [fault.to_dict() for fault in monitor.active_faults]}

    @app.post("/api/recovery/check-device")
    async def recovery_check_device(body: DeviceCheckRequest) -> dict[str, Any]:
        service = app.state.embodied_service
        if body.kind == "arm":
            from roboclaw.embodied.embodiment.hardware.monitor import check_arm_status

            arm = service.manifest.find_arm(body.alias)
            if arm is None:
                return {"ok": False, "kind": body.kind, "alias": body.alias}
            status = await asyncio.to_thread(check_arm_status, arm)
            return {"ok": status.connected, "kind": body.kind, "alias": body.alias}

        if body.kind == "camera":
            from roboclaw.embodied.embodiment.hardware.monitor import check_camera_status

            camera = service.manifest.find_camera(body.alias)
            if camera is None:
                return {"ok": False, "kind": body.kind, "alias": body.alias}
            status = await asyncio.to_thread(check_camera_status, camera)
            return {"ok": status.connected, "kind": body.kind, "alias": body.alias}

        return {"ok": False, "kind": body.kind, "alias": body.alias}

    @app.post("/api/recovery/check-arm-motors")
    async def recovery_check_arm_motors(body: ArmMotorCheckRequest) -> dict[str, Any]:
        from roboclaw.embodied.embodiment.arm.registry import (
            get_model,
            get_probe_config,
            get_runtime_spec,
        )
        from roboclaw.embodied.embodiment.hardware.probers import get_prober
        from roboclaw.embodied.embodiment.hardware.motors import _motor_config_from_arm

        service = app.state.embodied_service
        arm = service.manifest.find_arm(body.alias)
        if arm is None or get_model(arm.arm_type) != "so101" or not arm.connected:
            return {"ok": False, "alias": body.alias, "missing_motors": []}

        runtime_spec = get_runtime_spec(arm.arm_type)
        motor_config = _motor_config_from_arm(arm, runtime_spec)
        probe_cfg = get_probe_config(arm.arm_type)
        prober = get_prober(probe_cfg.protocol)
        found_ids = await asyncio.to_thread(
            prober.probe,
            arm.port,
            probe_cfg.baudrate,
            list(probe_cfg.motor_ids),
        )
        found_id_set = set(found_ids)
        missing_motors = [
            _pretty_motor_name(name)
            for name, (motor_id, _) in motor_config.items()
            if motor_id not in found_id_set
        ]
        return {
            "ok": not missing_motors,
            "alias": body.alias,
            "missing_motors": missing_motors,
        }

    @app.post("/api/recovery/restart-dashboard")
    async def recovery_restart_dashboard() -> dict[str, str]:
        schedule_dashboard_restart(app)
        return {"status": "restarting"}
