"""Unified service layer for all embodied operations."""

from __future__ import annotations

import json
import threading
from typing import Any

from roboclaw.data.datasets import DatasetCatalog, datasets_root_from_manifest
from roboclaw.embodied.board import Board
from roboclaw.embodied.board.board import IDLE_STATE
from roboclaw.embodied.calibration import AutoCalibrationBatch
from roboclaw.embodied.command import ActionError, CommandBuilder
from roboclaw.embodied.embodiment.doctor import DoctorService
from roboclaw.embodied.embodiment.hardware.monitor import (
    HardwareMonitor,
    check_arm_status,
    check_camera_status,
)
from roboclaw.embodied.embodiment.lock import EmbodimentBusyError, EmbodimentFileLock
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.service.capabilities import build_hardware_snapshot
from roboclaw.embodied.service.hub import HubService
from roboclaw.embodied.service.session import (
    InferSession,
    RecordSession,
    ReplaySession,
    Session,
    TeleopSession,
    TrainSession,
)
from roboclaw.embodied.service.session.calibrate import CalibrationSession
from roboclaw.embodied.service.session.setup import SetupSession
from roboclaw.embodied.service.verification import (
    PreflightVerifier,
    VerificationRequest,
    Verifier,
)


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
        preflight_verifier: Verifier | None = None,
    ) -> None:
        self._monitor = hardware_monitor
        self.board = board or Board()
        self.manifest = manifest or Manifest(board=self.board)
        self.manifest.ensure()
        self.datasets = DatasetCatalog(root_resolver=lambda: datasets_root_from_manifest(self.manifest))
        self._lock = threading.Lock()
        self._file_lock = EmbodimentFileLock()
        self._embodiment_owner: str = ""
        self._active_operation: Any | None = None
        self._active_session: Session | None = None
        self._recording_started = False
        self._preflight_verifier = preflight_verifier or PreflightVerifier()

        # Sub-services
        self.calibration = CalibrationSession(self)
        self.auto_calibration = AutoCalibrationBatch(board=self.board, manifest=self.manifest)
        self.setup = SetupSession(self)
        self.teleop = TeleopSession(self)
        self.record = RecordSession(self)
        self.replay = ReplaySession(self)
        self.train = TrainSession(self)
        self.infer = InferSession(self)
        self.hub = HubService(self)
        self.doctor = DoctorService(self)

        for operation in (
            self.calibration,
            self.auto_calibration,
            self.teleop,
            self.record,
            self.replay,
            self.infer,
        ):
            operation._exit_callback = self._on_operation_exit

    # -- Embodiment lock --

    @property
    def embodiment_busy(self) -> bool:
        with self._lock:
            return self._embodiment_owner != ""

    @property
    def busy(self) -> bool:
        with self._lock:
            busy, _ = self._busy_state_unlocked()
            return busy

    @property
    def busy_reason(self) -> str:
        with self._lock:
            _, reason = self._busy_state_unlocked(default_reason="unknown")
            return reason

    def _busy_state_unlocked(self, default_reason: str = "") -> tuple[bool, str]:
        """Return (busy, reason) using the active operation's state or the owner string."""
        if self._active_operation is not None and self._active_operation.busy:
            return True, self.board.state.get("state", default_reason)
        if self._embodiment_owner:
            return True, self._embodiment_owner
        return False, ""

    def acquire_embodiment(self, owner: str) -> None:
        with self._lock:
            busy, reason = self._busy_state_unlocked()
            if busy:
                raise EmbodimentBusyError(f"Embodiment busy: {reason}")
            self._file_lock.acquire_exclusive(owner)  # cross-process
            self._embodiment_owner = owner
            self.board.set_field("embodiment_owner", owner)

    def release_embodiment(self, owner: str = "") -> None:
        with self._lock:
            if owner and self._embodiment_owner != owner:
                return
            self._file_lock.release_exclusive()  # cross-process
            self._embodiment_owner = ""
            self.board.set_field("embodiment_owner", "")

    # -- Status --

    def get_status(self) -> dict[str, Any]:
        status = self.board.state
        # Merge cross-process owner info for REST callers
        if not status.get("embodiment_owner"):
            status["embodiment_owner"] = self._embodiment_owner or self._file_lock.owner()
        return status

    def get_logs(self) -> list[str]:
        return self.board.all_logs()

    def clear_logs(self) -> None:
        self.board.clear_logs()

    def _clear_recording_tracking(self, operation: Any | None) -> None:
        if operation is self.record or (operation is None and self._recording_started):
            self._recording_started = False
            if self._monitor is not None:
                self._monitor.set_recording_active(False)

    def _finish_active_operation(self, operation: Any | None, owner: str = "") -> None:
        """Clear service operation bookkeeping and release embodiment ownership."""
        with self._lock:
            if operation is None or self._active_operation is operation:
                self._active_operation = None
            if operation is None or self._active_session is operation or operation is self.auto_calibration:
                self._active_session = None
        self._clear_recording_tracking(operation)
        self.release_embodiment(owner)

    def _on_operation_exit(self, operation: Any) -> None:
        """Called when an operation exits naturally (not via stop())."""
        self._finish_active_operation(operation)

    async def _start_managed_session(
        self,
        session: Session,
        *,
        owner: str,
        argv: list[str],
    ) -> None:
        self.acquire_embodiment(owner)
        self._active_operation = session
        self._active_session = session
        try:
            await session.start(argv)
        except Exception:
            self._finish_active_operation(session, owner)
            raise

    async def _run_managed_session(
        self,
        session: Session,
        *,
        owner: str,
        argv: list[str],
        tty_handoff: Any | None = None,
    ) -> str:
        await self._start_managed_session(session, owner=owner, argv=argv)
        if tty_handoff:
            from roboclaw.embodied.toolkit.tty import TtySession

            try:
                return await TtySession(tty_handoff).run(session)
            finally:
                self._finish_active_operation(session, owner)
        try:
            await session.wait()
            return session.result()
        finally:
            self._finish_active_operation(session, owner)

    def _verify_inference_preflight(
        self,
        *,
        argv: list[str],
        dataset: Any,
        num_episodes: int,
        episode_time_s: int,
        use_cameras: bool,
    ) -> None:
        result = self._preflight_verifier.verify(VerificationRequest(
            argv=argv,
            manifest=self.manifest,
            dataset=dataset,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            use_cameras=use_cameras,
        ))
        if not result.ok:
            raise ActionError(result.format_violations())

    # -- Operations (Web entry points) --

    async def start_teleop(self, *, fps: int = 30, arms: str = "") -> None:
        self._require_capability("teleop")
        argv = CommandBuilder.teleop(self.manifest, fps=fps, arms=arms)
        await self._start_managed_session(self.teleop, owner="teleop", argv=argv)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
        dataset_name: str = "",
        use_cameras: bool = True,
        arms: str = "",
    ) -> str:
        self._require_capability("record" if use_cameras else "record_without_cameras")
        dataset = self.datasets.prepare_recording_dataset(dataset_name, prefix="rec")
        argv = CommandBuilder.record(
            self.manifest,
            dataset=dataset.runtime,
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            use_cameras=use_cameras,
            arms=arms,
        )
        await self._start_managed_session(self.record, owner="recording", argv=argv)
        await self.board.update(target_episodes=num_episodes, dataset=dataset.runtime.name)
        self._recording_started = True
        if self._monitor is not None:
            self._monitor.set_recording_active(True)
        return dataset.runtime.name

    async def start_replay(
        self,
        *,
        dataset_name: str = "default",
        episode: int = 0,
        fps: int = 30,
        arms: str = "",
    ) -> None:
        self._require_capability("replay")
        dataset = self.datasets.resolve_runtime_dataset(dataset_name)
        argv = CommandBuilder.replay(
            self.manifest, dataset=dataset.runtime, episode=episode, fps=fps, arms=arms,
        )
        await self._start_managed_session(self.replay, owner="replaying", argv=argv)

    async def start_inference(
        self,
        *,
        checkpoint_path: str = "",
        source_dataset: str = "",
        dataset_name: str = "",
        task: str = "eval",
        num_episodes: int = 1,
        episode_time_s: int = 60,
        arms: str = "",
        use_cameras: bool = True,
    ) -> None:
        self._require_capability("infer" if use_cameras else "infer_without_cameras")
        output_dataset = self.datasets.prepare_recording_dataset(dataset_name, prefix="eval")
        source = self.datasets.resolve_runtime_dataset(source_dataset) if source_dataset else None
        argv = CommandBuilder.infer(
            self.manifest,
            dataset=output_dataset.runtime,
            checkpoint_path=checkpoint_path,
            source_dataset=source.runtime if source else None,
            task=task,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            arms=arms,
            use_cameras=use_cameras,
        )
        self._verify_inference_preflight(
            argv=argv,
            dataset=output_dataset.runtime,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            use_cameras=use_cameras,
        )
        await self._start_managed_session(self.infer, owner="inferring", argv=argv)

    async def run_replay(
        self,
        *,
        dataset_name: str = "default",
        episode: int = 0,
        fps: int = 30,
        arms: str = "",
        tty_handoff: Any | None = None,
    ) -> str:
        self._require_capability("replay")
        dataset = self.datasets.resolve_runtime_dataset(dataset_name)
        argv = CommandBuilder.replay(
            self.manifest, dataset=dataset.runtime, episode=episode, fps=fps, arms=arms,
        )
        return await self._run_managed_session(
            self.replay, owner="replaying", argv=argv, tty_handoff=tty_handoff,
        )

    async def run_inference(
        self,
        *,
        checkpoint_path: str = "",
        source_dataset: str = "",
        dataset_name: str = "",
        task: str = "eval",
        num_episodes: int = 1,
        episode_time_s: int = 60,
        arms: str = "",
        use_cameras: bool = True,
        tty_handoff: Any | None = None,
    ) -> str:
        self._require_capability("infer" if use_cameras else "infer_without_cameras")
        output_dataset = self.datasets.prepare_recording_dataset(dataset_name, prefix="eval")
        source = self.datasets.resolve_runtime_dataset(source_dataset) if source_dataset else None
        argv = CommandBuilder.infer(
            self.manifest,
            dataset=output_dataset.runtime,
            checkpoint_path=checkpoint_path,
            source_dataset=source.runtime if source else None,
            task=task,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            arms=arms,
            use_cameras=use_cameras,
        )
        self._verify_inference_preflight(
            argv=argv,
            dataset=output_dataset.runtime,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            use_cameras=use_cameras,
        )
        return await self._run_managed_session(
            self.infer, owner="inferring", argv=argv, tty_handoff=tty_handoff,
        )

    async def dismiss_error(self) -> None:
        """Clear error state and release embodiment lock so user can retry."""
        self._finish_active_operation(self._active_operation)
        await self.board.update(**IDLE_STATE)

    async def stop(self) -> None:
        operation = self._active_operation
        if operation:
            await operation.stop()
            self._finish_active_operation(operation)

    async def save_episode(self) -> None:
        if self._active_operation is self.record:
            await self.record.request_save_episode()

    async def discard_episode(self) -> None:
        if self._active_operation is self.record:
            await self.record.request_discard_episode()

    async def skip_reset(self) -> None:
        if self._active_operation is self.record:
            await self.record.request_skip_reset()

    # -- Calibration (web) --

    async def start_calibration(self, arm_alias: str) -> dict[str, Any]:
        """Start calibration subprocess for a single arm (web path)."""
        arm = self.manifest.find_arm(arm_alias)
        if arm is None:
            raise RuntimeError(f"Arm '{arm_alias}' not found in manifest.")
        self.acquire_embodiment("calibrating")
        self._active_operation = self.calibration
        self._active_session = self.calibration
        try:
            await self.calibration.start_calibration(arm, self.manifest)
        except Exception:
            self._finish_active_operation(self.calibration, "calibrating")
            raise
        return {"state": "calibrating", "arm_alias": arm_alias}

    async def stop_calibration(self) -> None:
        """Properly terminate calibration subprocess (ESC → SIGINT → kill)."""
        await self.stop()

    async def start_auto_calibration(self) -> dict[str, Any]:
        arms = self.manifest.arms
        if not arms:
            raise RuntimeError("No arms configured.")
        self.acquire_embodiment("calibrating")
        self._active_operation = self.auto_calibration
        self._active_session = None
        try:
            total = await self.auto_calibration.start(arms)
        except Exception:
            self._finish_active_operation(self.auto_calibration, "calibrating")
            raise
        return {"state": "calibrating", "mode": "auto", "scope": "batch", "total": total}

    async def stop_auto_calibration(self) -> None:
        await self.stop()

    def post_calibration_command(self, command: str) -> None:
        """Forward a calibration command to the Board."""
        self.board.post_command(command)

    # -- Manifest mutations (kept identical) --

    def _require_not_busy(self) -> None:
        busy, reason = self._busy_state_unlocked()
        if busy:
            raise EmbodimentBusyError(f"Cannot modify config while busy: {reason}")

    def bind_arm(self, alias: str, arm_type: str, interface: Any, side: str = "") -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.set_arm(alias, arm_type, interface, side=side)

    def unbind_arm(self, alias: str) -> None:
        with self._lock:
            self._require_not_busy()
            self.manifest.remove_arm(alias)

    def rename_arm(self, alias: str, new_alias: str) -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.rename_arm(alias, new_alias)

    def bind_camera(self, alias: str, interface: Any, side: str = "") -> Binding:
        with self._lock:
            self._require_not_busy()
            return self.manifest.set_camera(alias, interface, side)

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

    def _hardware_snapshot(self, manifest: Manifest | None = None) -> dict[str, Any]:
        if manifest is None:
            manifest = self.manifest
        arm_statuses = [check_arm_status(arm) for arm in manifest.arms]
        camera_statuses = [check_camera_status(camera) for camera in manifest.cameras]
        active = self._active_operation is not None and self._active_operation.busy
        return build_hardware_snapshot(
            manifest.arms,
            arm_statuses,
            camera_statuses,
            session_busy=active,
        ).to_dict()

    def _require_capability(self, capability_name: str) -> None:
        status = self._hardware_snapshot()
        capability = status["capabilities"][capability_name]
        if capability["ready"]:
            return
        raise RuntimeError(" · ".join(capability["missing"]))

    def get_hardware_status(self, manifest: Manifest | None = None) -> dict[str, Any]:
        return self._hardware_snapshot(manifest)

    def read_servo_positions(self) -> dict[str, Any]:
        # Hold the service lock through serial I/O. This keeps same-process
        # recording starts from releasing this service's shared file-lock fd
        # while a worker thread is still polling the servos.
        with self._lock:
            busy, _ = self._busy_state_unlocked()
            if busy:
                return {"error": "busy", "arms": {}}
            if not self._file_lock.try_shared():
                return {"error": "busy", "arms": {}}
            try:
                from roboclaw.embodied.embodiment.hardware.motors import read_servo_positions
                return read_servo_positions(self.manifest.arms)
            finally:
                self._file_lock.release_shared()

    # -- Shutdown --

    async def shutdown(self) -> None:
        if self._active_operation and self._active_operation.busy:
            await self.stop()
        if self.setup.motion_active:
            self.setup.stop_motion_detection()
        if self._monitor is not None:
            self._monitor.set_recording_active(False)
        self.release_embodiment()
