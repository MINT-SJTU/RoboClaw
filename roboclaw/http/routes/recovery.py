"""Recovery routes for active faults and dashboard self-restart."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from fastapi import FastAPI
from loguru import logger

from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor


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
    @app.post("/api/recovery/check-hardware")
    async def recovery_check_hardware() -> dict[str, Any]:
        monitor: HardwareMonitor = app.state.hardware_monitor
        await monitor.run_check_once()
        return {"faults": [fault.to_dict() for fault in monitor.active_faults]}

    @app.post("/api/recovery/restart-dashboard")
    async def recovery_restart_dashboard() -> dict[str, str]:
        schedule_dashboard_restart(app)
        return {"status": "restarting"}
