"""Unified service layer for all embodied operations.

Every interface (Dashboard, CLI, Agent Tool) calls EmbodiedService.
This is the single coordination point for embodiment locking, hardware
monitor integration, and operation lifecycle.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from roboclaw.embodied.engine import OperationEngine
from roboclaw.embodied.engine.helpers import group_arms
from roboclaw.embodied.events import EventBus
from roboclaw.embodied.hardware.monitor import HardwareMonitor
from roboclaw.embodied.hardware.monitor import (
    ArmStatus,
    CameraStatus,
    check_arm_status,
    check_camera_status,
)
from roboclaw.embodied.manifest import Manifest
from roboclaw.embodied.manifest.binding import Binding
from roboclaw.embodied.manifest.binding import Binding
from roboclaw.embodied.service.calibration import CalibrationService
from roboclaw.embodied.service.calibration_session import CalibrationSession as CalibrationCLI
from roboclaw.embodied.service.doctor_service import DoctorService
from roboclaw.embodied.service.hand_session import HandSession
from roboclaw.embodied.service.infer_session import InferSession
from roboclaw.embodied.service.record_session import RecordSession
from roboclaw.embodied.service.replay_session import ReplaySession
from roboclaw.embodied.service.setup_session import SetupSession
from roboclaw.embodied.service.teleop_session import TeleopSession
from roboclaw.embodied.service.train_session import TrainSession


class EmbodimentBusyError(RuntimeError):
    """Raised when the embodiment lock cannot be acquired."""


def _compute_readiness(
    arms: list[Binding],
    arm_statuses: list[ArmStatus],
    camera_statuses: list[CameraStatus],
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    grouped = group_arms(arms)
    if not grouped["followers"]:
        missing.append("No follower arm configured")
    if not grouped["leaders"]:
        missing.append("No leader arm configured")
    for status in arm_statuses:
        if not status.connected:
            missing.append(f"Arm '{status.alias}' is disconnected")
        elif not status.calibrated:
            missing.append(f"Arm '{status.alias}' is not calibrated")
    for status in camera_statuses:
        if not status.connected:
            missing.append(f"Camera '{status.alias}' is disconnected")
    followers = grouped["followers"]
    leaders = grouped["leaders"]
    if followers and leaders and len(followers) != len(leaders):
        missing.append(f"Follower/leader count mismatch: {len(followers)} vs {len(leaders)}")
    return len(missing) == 0, missing


class EmbodiedService:
    """Single point of control for ALL embodied operations.

    Two-layer mutex:
    - Layer 1: Embodiment lock — physical robot is one resource.
      Owner is a string like "teleop", "recording", "calibrating", "scanning".
    - Layer 2: Monitor coordination — recording_active lifecycle.

    Sub-services:
    - session: teleop/recording via OperationEngine
    - calibration: arm calibration via CalibrationSession
    - setup: hardware setup workflow via SetupSession
    """

    def __init__(
        self,
        hardware_monitor: HardwareMonitor | None = None,
        event_bus: EventBus | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._monitor = hardware_monitor
        self._bus = event_bus or EventBus()
        self.manifest = manifest or Manifest(event_bus=self._bus)
        self._lock = threading.Lock()
        self._embodiment_owner: str = ""
        self._engine = OperationEngine(on_state_change=self._on_engine_state_change)
        self._recording_started = False

        # Sub-services
        self.calibration = CalibrationService(self, event_bus=self._bus)
        self.setup = SetupSession(self)
        self.teleop = TeleopSession(self)
        self.record = RecordSession(self)
        self.replay = ReplaySession(self)
        self.train = TrainSession(self)
        self.infer = InferSession(self)
        self.hand = HandSession(self)
        self.doctor = DoctorService(self)
        self.calibration_session = CalibrationCLI(self)

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    # -- Embodiment lock ------------------------------------------------------

    @property
    def embodiment_busy(self) -> bool:
        with self._lock:
            return self._embodiment_owner != ""

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._engine.busy or self._embodiment_owner != ""

    @property
    def busy_reason(self) -> str:
        with self._lock:
            if self._engine.busy:
                return self._engine.state
            return self._embodiment_owner

    def acquire_embodiment(self, owner: str) -> None:
        with self._lock:
            if self._engine.busy or self._embodiment_owner:
                reason = self._engine.state if self._engine.busy else self._embodiment_owner
                raise EmbodimentBusyError(f"Embodiment busy: {reason}")
            self._embodiment_owner = owner

    def release_embodiment(self, owner: str = "") -> None:
        """Release the embodiment lock. If owner is specified, only release if it matches."""
        with self._lock:
            if owner and self._embodiment_owner != owner:
                return
            self._embodiment_owner = ""

    def get_status(self) -> dict[str, Any]:
        return self._engine.get_status()

    async def start_teleop(self, *, fps: int = 30) -> None:
        await self._engine.start_teleop(fps=fps, setup=self.manifest)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
    ) -> str:
        dataset_name = await self._engine.start_recording(
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            setup=self.manifest,
        )
        self._recording_started = True
        if self._monitor is not None:
            self._monitor.set_recording_active(True)
        return dataset_name

    async def stop(self) -> None:
        await self._engine.stop()

    async def save_episode(self) -> None:
        await self._engine.save_episode()

    async def discard_episode(self) -> None:
        await self._engine.discard_episode()

    async def skip_reset(self) -> None:
        await self._engine.skip_reset()

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

    # -- Manifest mutations (atomic, busy-checked) ----------------------------

    def _require_not_busy(self) -> None:
        """Raise if embodiment is busy. Must be called inside _lock or from sync context."""
        if self._engine.busy or self._embodiment_owner:
            reason = self._engine.state if self._engine.busy else self._embodiment_owner
            raise EmbodimentBusyError(f"Cannot modify config while busy: {reason}")

    def bind_arm(self, alias: str, arm_type: str, interface: Any) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.set_arm(alias, arm_type, interface)

    def unbind_arm(self, alias: str) -> None:
        with self._lock:
            self._require_not_busy()
            self.manifest.remove_arm(alias)

    def rename_arm(self, alias: str, new_alias: str) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.rename_arm(alias, new_alias)

    def bind_camera(self, alias: str, interface: Any) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.set_camera(alias, interface)

    def unbind_camera(self, alias: str) -> None:
        with self._lock:
            self._require_not_busy()
            self.manifest.remove_camera(alias)

    def rename_camera(self, old_alias: str, new_alias: str) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.rename_camera(old_alias, new_alias)

    def bind_hand(self, alias: str, hand_type: str, interface: Any, slave_id: int) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.set_hand(alias, hand_type, interface, slave_id)

    def unbind_hand(self, alias: str) -> None:
        with self._lock:
            self._require_not_busy()
            self.manifest.remove_hand(alias)

    def rename_hand(self, old_alias: str, new_alias: str) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.rename_hand(old_alias, new_alias)

    # -- Queries ---------------------------------------------------------------

    def get_manifest_summary(self) -> str:
        snapshot = self.manifest.snapshot
        snapshot["status"] = self.get_hardware_status(self.manifest)
        return json.dumps(snapshot, indent=2, ensure_ascii=False)

    def get_hardware_status(self, manifest: Manifest | None = None) -> dict[str, Any]:
        if manifest is None:
            manifest = self.manifest
        arms = manifest.arms
        cameras = manifest.cameras
        arm_statuses = [check_arm_status(arm) for arm in arms]
        camera_statuses = [check_camera_status(camera) for camera in cameras]
        ready, missing = _compute_readiness(arms, arm_statuses, camera_statuses)
        return {
            "ready": ready,
            "missing": missing,
            "arms": [status.to_dict() for status in arm_statuses],
            "cameras": [status.to_dict() for status in camera_statuses],
            "session_busy": self._engine.busy,
        }

    def read_servo_positions(self) -> dict[str, Any]:
        if self.busy:
            return {"error": "busy", "arms": {}}
        from roboclaw.embodied.hardware.motors import read_servo_positions

        return read_servo_positions(self.manifest.arms)

    async def _on_engine_state_change(self, status: dict[str, Any]) -> None:
        new_state = status.get("state", "idle")
        if new_state == "idle" and self._recording_started:
            self._recording_started = False
            if self._monitor is not None:
                self._monitor.set_recording_active(False)
        from roboclaw.embodied.events import SessionStateChangedEvent

        await self._bus.emit(SessionStateChangedEvent(**status))

    # -- Shutdown -------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._engine.busy:
            await self.stop()
        if self.calibration.active:
            await self.calibration.cancel()
        if self.setup.motion_active:
            self.setup.stop_motion_detection()
        if self._monitor is not None:
            self._monitor.set_recording_active(False)
