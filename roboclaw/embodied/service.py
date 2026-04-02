"""Unified service layer for all embodied operations.

Every interface (Dashboard, CLI, Agent Tool) calls EmbodiedService
instead of managing subprocess lifecycle directly. This ensures
consistent busy checks, port locking, and hardware monitor state.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

from roboclaw.embodied.hardware_monitor import (
    ArmStatus,
    CameraStatus,
    HardwareMonitor,
    check_arm_status,
    check_camera_status,
)
from roboclaw.embodied.operation_session import OperationSession, StatusCallback
from roboclaw.embodied.ops.helpers import group_arms
from roboclaw.embodied.setup import load_setup


def _compute_readiness(
    arms: list[dict[str, Any]],
    arm_statuses: list[ArmStatus],
    camera_statuses: list[CameraStatus],
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    grouped = group_arms(arms)
    if not grouped["followers"]:
        missing.append("No follower arm configured")
    if not grouped["leaders"]:
        missing.append("No leader arm configured")
    for s in arm_statuses:
        if not s.connected:
            missing.append(f"Arm '{s.alias}' is disconnected")
        elif not s.calibrated:
            missing.append(f"Arm '{s.alias}' is not calibrated")
    for s in camera_statuses:
        if not s.connected:
            missing.append(f"Camera '{s.alias}' is disconnected")
    f, l = grouped["followers"], grouped["leaders"]
    if f and l and len(f) != len(l):
        missing.append(f"Follower/leader count mismatch: {len(f)} vs {len(l)}")
    return len(missing) == 0, missing


class EmbodiedService:
    """Single point of control for all teleop/record operations.

    Responsibilities:
    1. Ensure only one operation runs at a time (teleop or record)
    2. Coordinate with HardwareMonitor (recording_active lifecycle)
    3. Provide operation status queries (busy, get_status)
    4. Hardware readiness checks
    """

    def __init__(
        self,
        hardware_monitor: HardwareMonitor | None = None,
        on_state_change: StatusCallback | None = None,
    ) -> None:
        self._monitor = hardware_monitor
        self._session = OperationSession(on_state_change=self._on_session_state_change)
        self._external_callback = on_state_change
        self._recording_started = False

    # -- Properties -----------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._session.busy

    def get_status(self) -> dict[str, Any]:
        return self._session.get_status()

    # -- Teleop ---------------------------------------------------------------

    async def start_teleop(self, *, fps: int = 30) -> None:
        await self._session.start_teleop(fps=fps)

    # -- Recording ------------------------------------------------------------

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
    ) -> str:
        """Start recording. Returns dataset_name."""
        dataset_name = await self._session.start_recording(
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
        )
        self._recording_started = True
        if self._monitor is not None:
            self._monitor.set_recording_active(True)
        return dataset_name

    async def stop(self) -> None:
        await self._session.stop()
        # recording_active is reset by _on_session_state_change callback

    # -- Episode control ------------------------------------------------------

    async def save_episode(self) -> None:
        await self._session.save_episode()

    async def discard_episode(self) -> None:
        await self._session.discard_episode()

    async def skip_reset(self) -> None:
        await self._session.skip_reset()

    # -- Hardware status ------------------------------------------------------

    def get_hardware_status(self) -> dict[str, Any]:
        setup = load_setup()
        arms = setup.get("arms", [])
        cameras = setup.get("cameras", [])
        arm_statuses = [check_arm_status(a) for a in arms]
        camera_statuses = [check_camera_status(c) for c in cameras]
        ready, missing = _compute_readiness(arms, arm_statuses, camera_statuses)
        return {
            "ready": ready,
            "missing": missing,
            "arms": [s.to_dict() for s in arm_statuses],
            "cameras": [s.to_dict() for s in camera_statuses],
            "session_busy": self._session.busy,
        }

    # -- Shutdown -------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._session.busy:
            await self._session.stop()
        if self._monitor is not None:
            self._monitor.set_recording_active(False)

    # -- Internal: state change routing ---------------------------------------

    async def _on_session_state_change(self, status: dict[str, Any]) -> None:
        """Called by OperationSession on every state transition.

        Only resets recording_active when transitioning from an active
        recording (not from teleop or other states).
        """
        new_state = status.get("state", "idle")
        if new_state == "idle" and self._recording_started:
            self._recording_started = False
            if self._monitor is not None:
                self._monitor.set_recording_active(False)

        if self._external_callback is not None:
            result = self._external_callback(status)
            if inspect.isawaitable(result):
                await result
