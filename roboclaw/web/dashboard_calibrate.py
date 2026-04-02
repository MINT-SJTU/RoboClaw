"""Dashboard calibration API routes.

Exposes the shared CalibrationSession via HTTP endpoints so the
dashboard frontend can drive calibration step-by-step.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.calibration import CalibrationSession
from roboclaw.embodied.port_lock import port_locks
from roboclaw.embodied.setup import load_setup, mark_arm_calibrated


class StartCalibrationRequest(BaseModel):
    arm_alias: str


class _CalibrationState:
    """Holds the active calibration session for the dashboard."""

    def __init__(self) -> None:
        self.session: CalibrationSession | None = None
        self.arm_alias: str = ""
        self._lock_handle: Any = None

    @property
    def active(self) -> bool:
        return self.session is not None


_state = _CalibrationState()


def register_calibrate_routes(app: FastAPI) -> None:
    """Register /api/dashboard/calibrate/* routes on the given app."""

    API = "/api/dashboard/calibrate"

    @app.post(f"{API}/start")
    async def calibrate_start(body: StartCalibrationRequest) -> dict:
        if _state.active:
            raise HTTPException(409, "Calibration already in progress.")

        # Check session not busy
        dashboard_session = getattr(app.state, "dashboard_session", None)
        if dashboard_session and dashboard_session.busy:
            raise HTTPException(409, "Cannot calibrate while teleop/recording is active.")

        setup = load_setup()
        arm = _find_arm(setup, body.arm_alias)

        # Acquire port lock
        port = arm.get("port", "")
        if port:
            _state._lock_handle = await port_locks.acquire(port).__aenter__()

        session = CalibrationSession(arm)
        try:
            await asyncio.to_thread(session.connect)
        except Exception:
            if _state._lock_handle:
                await port_locks.acquire(port).__aexit__(None, None, None)
                _state._lock_handle = None
            raise

        _state.session = session
        _state.arm_alias = body.arm_alias
        return {"state": session.state, "arm_alias": body.arm_alias}

    @app.get(f"{API}/status")
    async def calibrate_status() -> dict:
        if not _state.active:
            return {"state": "idle", "arm_alias": ""}
        return {"state": _state.session.state, "arm_alias": _state.arm_alias}

    @app.post(f"{API}/set-homing")
    async def calibrate_set_homing() -> dict:
        _require_session()
        offsets = await asyncio.to_thread(_state.session.set_homing)
        return {"state": _state.session.state, "homing_offsets": offsets}

    @app.get(f"{API}/positions")
    async def calibrate_positions() -> dict:
        _require_session()
        if _state.session.state != "recording":
            raise HTTPException(400, f"Not recording (state={_state.session.state}).")
        snapshot = await asyncio.to_thread(_state.session.read_range_positions)
        return {
            "positions": snapshot.positions,
            "mins": snapshot.mins,
            "maxes": snapshot.maxes,
        }

    @app.post(f"{API}/finish")
    async def calibrate_finish() -> dict:
        _require_session()
        calibration = await asyncio.to_thread(_state.session.finish)
        mark_arm_calibrated(_state.arm_alias)
        await _cleanup()
        return {"state": "done", "calibration": calibration}

    @app.post(f"{API}/cancel")
    async def calibrate_cancel() -> dict:
        if _state.active:
            await asyncio.to_thread(_state.session.cancel)
            await _cleanup()
        return {"state": "idle"}


def _find_arm(setup: dict, alias: str) -> dict:
    """Find arm config by alias."""
    for arm in setup.get("arms", []):
        if arm.get("alias") == alias:
            return arm
    raise HTTPException(404, f"Arm '{alias}' not found in setup.")


def _require_session() -> None:
    if not _state.active:
        raise HTTPException(400, "No calibration session active.")


async def _cleanup() -> None:
    """Release port lock and clear session."""
    if _state._lock_handle and _state.session:
        port = _state.session._arm.get("port", "")
        if port:
            await port_locks.acquire(port).__aexit__(None, None, None)
    _state._lock_handle = None
    if _state.session:
        await asyncio.to_thread(_state.session.disconnect)
    _state.session = None
    _state.arm_alias = ""
