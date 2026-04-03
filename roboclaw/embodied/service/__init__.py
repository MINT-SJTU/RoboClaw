"""Unified service layer for all embodied operations.

Every interface (Dashboard, CLI, Agent Tool) calls EmbodiedService.
This is the single coordination point for embodiment locking, hardware
monitor integration, and operation lifecycle.
"""

from __future__ import annotations

import threading
from typing import Any

from roboclaw.embodied.engine import StatusCallback
from roboclaw.embodied.hardware_monitor import HardwareMonitor
from roboclaw.embodied.service.calibration import CalibrationService
from roboclaw.embodied.service.config import ConfigService
from roboclaw.embodied.service.queries import QueryService
from roboclaw.embodied.service.scanning import ScanningService
from roboclaw.embodied.service.session import SessionService


class EmbodimentBusyError(RuntimeError):
    """Raised when the embodiment lock cannot be acquired."""


class EmbodiedService:
    """Single point of control for ALL embodied operations.

    Two-layer mutex:
    - Layer 1: Embodiment lock — physical robot is one resource.
      Owner is a string like "teleop", "recording", "calibrating", "scanning".
    - Layer 2: Monitor coordination — recording_active lifecycle.

    Sub-services:
    - session: teleop/recording via OperationEngine
    - calibration: arm calibration via CalibrationSession
    - scanning: port/camera scanning via HardwareScanner
    """

    def __init__(
        self,
        hardware_monitor: HardwareMonitor | None = None,
        on_state_change: StatusCallback | None = None,
    ) -> None:
        self._monitor = hardware_monitor
        self._lock = threading.Lock()
        self._embodiment_owner: str = ""

        # Sub-services
        self.session = SessionService(self, external_callback=on_state_change)
        self.calibration = CalibrationService(self)
        self.scanning = ScanningService(self)
        self.config = ConfigService(self)
        self.queries = QueryService(self)

    # -- Embodiment lock ------------------------------------------------------

    @property
    def embodiment_busy(self) -> bool:
        with self._lock:
            return self._embodiment_owner != ""

    @property
    def busy(self) -> bool:
        with self._lock:
            return self.session.busy or self._embodiment_owner != ""

    @property
    def busy_reason(self) -> str:
        with self._lock:
            if self.session.busy:
                return self.session.state
            return self._embodiment_owner

    def acquire_embodiment(self, owner: str) -> None:
        with self._lock:
            if self.session.busy or self._embodiment_owner:
                reason = self.session.state if self.session.busy else self._embodiment_owner
                raise EmbodimentBusyError(f"Embodiment busy: {reason}")
            self._embodiment_owner = owner

    def release_embodiment(self, owner: str = "") -> None:
        """Release the embodiment lock. If owner is specified, only release if it matches."""
        with self._lock:
            if owner and self._embodiment_owner != owner:
                return
            self._embodiment_owner = ""

    # -- Delegated: session ---------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return self.session.get_status()

    async def start_teleop(self, *, fps: int = 30) -> None:
        await self.session.start_teleop(fps=fps)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
    ) -> str:
        return await self.session.start_recording(
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
        )

    async def stop(self) -> None:
        await self.session.stop()

    async def save_episode(self) -> None:
        await self.session.save_episode()

    async def discard_episode(self) -> None:
        await self.session.discard_episode()

    async def skip_reset(self) -> None:
        await self.session.skip_reset()

    # -- Delegated: calibration -----------------------------------------------

    async def start_calibration(self, arm_alias: str) -> dict[str, Any]:
        return await self.calibration.start(arm_alias)

    def get_calibration_status(self) -> dict[str, Any]:
        return self.calibration.get_status()

    async def set_calibration_homing(self) -> dict[str, Any]:
        return await self.calibration.set_homing()

    async def read_calibration_positions(self) -> dict[str, Any]:
        return await self.calibration.read_positions()

    async def finish_calibration(self) -> dict[str, Any]:
        return await self.calibration.finish()

    async def cancel_calibration(self) -> None:
        await self.calibration.cancel()

    # -- Scanning is accessed via service.scanning directly --------------------
    # No top-level delegation needed — callers use service.scanning.*

    # -- Delegated: queries ----------------------------------------------------

    def get_hardware_status(self) -> dict[str, Any]:
        return self.queries.get_hardware_status()

    def read_servo_positions(self) -> dict[str, Any]:
        return self.queries.read_servo_positions()

    # -- Async operations (wrap execute.py with embodiment lock) ---------------

    async def run_calibrate(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_calibrate
        return await _do_calibrate(setup, kwargs, tty_handoff)

    async def run_identify(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_identify
        return await _do_identify(setup, kwargs, tty_handoff)

    async def run_replay(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        self.acquire_embodiment("replaying")
        try:
            from roboclaw.embodied.ops.execute import _do_replay
            return await _do_replay(setup, kwargs, tty_handoff)
        finally:
            self.release_embodiment()

    async def run_doctor(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_doctor
        return await _do_doctor(setup, kwargs, tty_handoff)

    async def start_training(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_train
        return await _do_train(setup, kwargs, tty_handoff)

    async def get_job_status(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_job_status
        return await _do_job_status(setup, kwargs, tty_handoff)

    async def run_policy(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.ops.execute import _do_run_policy
        return await _do_run_policy(setup, kwargs, tty_handoff)

    async def hand_open(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.hand_actions import _do_hand_open
        return await _do_hand_open(setup, kwargs, tty_handoff)

    async def hand_close(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.hand_actions import _do_hand_close
        return await _do_hand_close(setup, kwargs, tty_handoff)

    async def hand_pose(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.hand_actions import _do_hand_pose
        return await _do_hand_pose(setup, kwargs, tty_handoff)

    async def hand_status(self, setup: dict, kwargs: dict, tty_handoff: Any) -> str:
        from roboclaw.embodied.hand_actions import _do_hand_status
        return await _do_hand_status(setup, kwargs, tty_handoff)

    # -- Shutdown -------------------------------------------------------------

    async def shutdown(self) -> None:
        if self.session.busy:
            await self.session.stop()
        if self.calibration.active:
            await self.calibration.cancel()
        if self.scanning.motion_active:
            self.scanning.stop_motion_detection()
        if self._monitor is not None:
            self._monitor.set_recording_active(False)
