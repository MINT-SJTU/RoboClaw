"""Unified service layer for all embodied operations."""

from __future__ import annotations

import json
import threading
from typing import Any

from roboclaw.embodied.board import Board, Command, SessionState
from roboclaw.embodied.board.board import IDLE_STATE
from roboclaw.embodied.command import CommandBuilder, group_arms
from roboclaw.embodied.embodiment.hardware.monitor import (
    ArmStatus, CameraStatus, HardwareMonitor,
    check_arm_status, check_camera_status,
)
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.service.session import (
    InferSession, RecordSession, ReplaySession, Session,
    TeleopSession, TrainSession,
)
from roboclaw.embodied.service.session.calibrate import (
    CalibrationService, CalibrationSession as CalibrationCLI,
)
from roboclaw.embodied.embodiment.doctor import DoctorService
from roboclaw.embodied.service.session.setup import SetupSession


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
    - Layer 2: Active session — only one operation at a time.
    """

    def __init__(
        self,
        hardware_monitor: HardwareMonitor | None = None,
        board: Board | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._monitor = hardware_monitor
        self.board = board or Board()
        self.manifest = manifest or Manifest(board=self.board)
        self.manifest.ensure()
        self._lock = threading.Lock()
        self._embodiment_owner: str = ""
        self._active_session: Session | None = None
        self._recording_started = False

        # Sub-services
        self.calibration = CalibrationService(self, board=self.board)
        self.setup = SetupSession(self)
        self.teleop = TeleopSession(self)
        self.record = RecordSession(self)
        self.replay = ReplaySession(self)
        self.train = TrainSession(self)
        self.infer = InferSession(self)
        self.doctor = DoctorService(self)
        self.calibration_session = CalibrationCLI(self)

    # -- Embodiment lock --

    @property
    def embodiment_busy(self) -> bool:
        with self._lock:
            return self._embodiment_owner != ""

    @property
    def busy(self) -> bool:
        with self._lock:
            active = self._active_session is not None and self._active_session.busy
            return active or self._embodiment_owner != ""

    @property
    def busy_reason(self) -> str:
        with self._lock:
            if self._active_session and self._active_session.busy:
                return self.board.state.get("state", "unknown")
            return self._embodiment_owner

    def acquire_embodiment(self, owner: str) -> None:
        with self._lock:
            active = self._active_session is not None and self._active_session.busy
            if active or self._embodiment_owner:
                reason = self.board.state.get("state", "") if active else self._embodiment_owner
                raise EmbodimentBusyError(f"Embodiment busy: {reason}")
            self._embodiment_owner = owner

    def release_embodiment(self, owner: str = "") -> None:
        with self._lock:
            if owner and self._embodiment_owner != owner:
                return
            self._embodiment_owner = ""

    # -- Status --

    def get_status(self) -> dict[str, Any]:
        return self.board.state

    # -- Operations (Web entry points) --

    async def start_teleop(self, *, fps: int = 30) -> None:
        argv = CommandBuilder.teleop(self.manifest, fps=fps)
        self._active_session = self.teleop
        await self.teleop.start(argv)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
    ) -> str:
        argv, dataset_name = CommandBuilder.record(
            self.manifest,
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
        )
        await self.board.update(target_episodes=num_episodes, dataset=dataset_name)
        self._active_session = self.record
        await self.record.start(argv)
        self._recording_started = True
        if self._monitor is not None:
            self._monitor.set_recording_active(True)
        return dataset_name

    async def start_replay(
        self,
        *,
        dataset_name: str = "default",
        episode: int = 0,
        fps: int = 30,
    ) -> None:
        argv = CommandBuilder.replay(
            self.manifest, dataset_name=dataset_name, episode=episode, fps=fps,
        )
        self._active_session = self.replay
        await self.replay.start(argv, initial_state=SessionState.REPLAYING)

    async def start_inference(
        self,
        *,
        checkpoint_path: str = "",
        source_dataset: str = "",
        dataset_name: str = "",
        task: str = "eval",
        num_episodes: int = 1,
    ) -> None:
        argv = CommandBuilder.infer(
            self.manifest,
            checkpoint_path=checkpoint_path,
            source_dataset=source_dataset,
            dataset_name=dataset_name,
            task=task,
            num_episodes=num_episodes,
        )
        self._active_session = self.infer
        await self.infer.start(argv, initial_state=SessionState.INFERRING)

    async def stop(self) -> None:
        if self._active_session:
            await self._active_session.stop()
            if self._recording_started:
                self._recording_started = False
                if self._monitor is not None:
                    self._monitor.set_recording_active(False)

    async def save_episode(self) -> None:
        if self._active_session:
            self.board.post_command(Command.SAVE_EPISODE)

    async def discard_episode(self) -> None:
        if self._active_session:
            self.board.post_command(Command.DISCARD_EPISODE)

    async def skip_reset(self) -> None:
        if self._active_session:
            self.board.post_command(Command.SKIP_RESET)

    # -- Calibration (delegated) --

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

    # -- Manifest mutations (kept identical) --

    def _require_not_busy(self) -> None:
        active = self._active_session is not None and self._active_session.busy
        if active or self._embodiment_owner:
            reason = self.board.state.get("state", "") if active else self._embodiment_owner
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

    # -- Queries --

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
        active = self._active_session is not None and self._active_session.busy
        return {
            "ready": ready,
            "missing": missing,
            "arms": [status.to_dict() for status in arm_statuses],
            "cameras": [status.to_dict() for status in camera_statuses],
            "session_busy": active,
        }

    def read_servo_positions(self) -> dict[str, Any]:
        if self.busy:
            return {"error": "busy", "arms": {}}
        from roboclaw.embodied.embodiment.hardware.motors import read_servo_positions
        return read_servo_positions(self.manifest.arms)

    # -- Shutdown --

    async def shutdown(self) -> None:
        if self._active_session and self._active_session.busy:
            await self.stop()
        if self.calibration.active:
            await self.calibration.cancel()
        if self.setup.motion_active:
            self.setup.stop_motion_detection()
        if self._monitor is not None:
            self._monitor.set_recording_active(False)
