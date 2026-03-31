"""Web-based robot session — delegates to existing embodied infrastructure."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Any

from loguru import logger

from roboclaw.embodied.ops.helpers import (
    _arm_id,
    _dataset_path,
    _group_arms,
    _resolve_action_arms,
    _validate_pairing,
)
from roboclaw.embodied.sensor.camera import resolve_cameras
from roboclaw.embodied.setup import load_setup


class RobotSession:
    """Manages robot lifecycle via subprocess delegation to LeRobot CLI."""

    _STATES = ("disconnected", "connected", "teleoperating", "recording")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "disconnected"
        self._setup: dict[str, Any] = {}
        self._grouped: dict[str, list[dict]] = {}
        self._cameras: dict[str, dict] = {}
        self._process_pid: int | None = None
        self._process_stdin: Any = None
        self._recording_dataset: str = ""
        self._temp_dirs: list[str] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def cameras_locked(self) -> bool:
        """True when cameras/serial must not be opened (teleop/record owns the devices)."""
        return self._state in ("preparing", "teleoperating", "recording")

    def _require_state(self, *allowed: str) -> None:
        if self._state not in allowed:
            raise RuntimeError(f"Requires state {allowed}, current is '{self._state}'")

    def connect(self, setup: dict | None = None) -> None:
        with self._lock:
            self._require_state("disconnected")
            setup = setup or load_setup()
            self._setup = setup

            arms = _resolve_action_arms(setup, {})
            grouped = _group_arms(arms)
            error = _validate_pairing(grouped["followers"], grouped["leaders"])
            if error:
                raise RuntimeError(error)

            self._grouped = grouped
            self._cameras = resolve_cameras(setup)
            self._state = "connected"
            logger.info("Robot session connected: {} follower(s), {} leader(s), {} camera(s)",
                        len(grouped["followers"]), len(grouped["leaders"]), len(self._cameras))

    def disconnect(self) -> None:
        with self._lock:
            if self._state == "disconnected":
                return
            self._kill_subprocess()
            self._state = "disconnected"
            logger.info("Robot session disconnected")

    def start_teleop(self, fps: int = 30) -> None:
        with self._lock:
            self._require_state("connected")
            self._state = "preparing"
        import time; time.sleep(5)
        with self._lock:
            argv = self._build_teleop_argv()
            self._state = "teleoperating"
            self._launch_subprocess(argv)
            logger.info("Teleoperation started")

    def stop_teleop(self) -> None:
        with self._lock:
            self._require_state("teleoperating")
            self._kill_subprocess()
            self._state = "connected"
            logger.info("Teleoperation stopped")

    def start_recording(
        self,
        dataset_name: str,
        task: str,
        fps: int = 30,
        num_episodes: int = 10,
    ) -> None:
        from datetime import datetime
        dataset_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._lock:
            self._require_state("connected", "teleoperating")
            if self._state == "teleoperating":
                self._kill_subprocess()

            self._recording_dataset = dataset_name
            self._state = "preparing"
        import time; time.sleep(5)
        with self._lock:
            argv = self._build_record_argv(dataset_name, task, fps, num_episodes)
            self._state = "recording"
            self._launch_subprocess(argv)
            logger.info("Recording started: dataset={}, episodes={}", dataset_name, num_episodes)

    def stop_recording(self) -> None:
        with self._lock:
            self._require_state("recording")
            self._kill_subprocess()
            self._state = "connected"
            logger.info("Recording stopped")

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "episode_count": 0,
            "frame_count": 0,
            "target_episodes": 0,
            "recording_fps": 30,
            "teleop_fps": 30,
            "task": "",
            "dataset": self._recording_dataset if self._state == "recording" else None,
        }

    # -- Command building (reuses SO101Controller) ---------------------

    def _build_teleop_argv(self) -> list[str]:
        from roboclaw.embodied.embodiment.arm.so101 import SO101Controller

        controller = SO101Controller()
        followers = self._grouped["followers"]
        leaders = self._grouped["leaders"]

        if len(followers) == 1:
            return controller.teleoperate(
                robot_type=followers[0]["type"],
                robot_port=followers[0]["port"],
                robot_cal_dir=followers[0]["calibration_dir"],
                robot_id=_arm_id(followers[0]),
                teleop_type=leaders[0]["type"],
                teleop_port=leaders[0]["port"],
                teleop_cal_dir=leaders[0]["calibration_dir"],
                teleop_id=_arm_id(leaders[0]),
                cameras=self._cameras,
            )

        robot_dir, teleop_dir = self._create_bimanual_cal_dirs(followers, leaders)
        return controller.teleoperate_bimanual(
            robot_id="bimanual",
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            teleop_id="bimanual",
            teleop_cal_dir=teleop_dir,
            left_teleop=leaders[0],
            right_teleop=leaders[1],
            cameras=self._cameras,
        )

    def _build_record_argv(
        self, dataset_name: str, task: str, fps: int, num_episodes: int,
    ) -> list[str]:
        from roboclaw.embodied.embodiment.arm.so101 import SO101Controller

        controller = SO101Controller()
        followers = self._grouped["followers"]
        leaders = self._grouped["leaders"]
        ds_path = _dataset_path(self._setup, dataset_name)
        record_kwargs: dict[str, Any] = {
            "cameras": self._cameras,
            "repo_id": f"local/{dataset_name}",
            "dataset_root": str(ds_path),
            "task": task,
            "fps": fps,
            "num_episodes": num_episodes,
        }

        if len(followers) == 1:
            return controller.record(
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

        robot_dir, teleop_dir = self._create_bimanual_cal_dirs(followers, leaders)
        return controller.record_bimanual(
            robot_id="bimanual",
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            teleop_id="bimanual",
            teleop_cal_dir=teleop_dir,
            left_teleop=leaders[0],
            right_teleop=leaders[1],
            **record_kwargs,
        )

    def _create_bimanual_cal_dirs(
        self, followers: list[dict], leaders: list[dict],
    ) -> tuple[str, str]:
        """Create persistent temp dirs for bimanual calibration (cleaned on stop/disconnect)."""
        from pathlib import Path
        robot_dir = tempfile.mkdtemp(prefix="roboclaw-bimanual-robot-")
        teleop_dir = tempfile.mkdtemp(prefix="roboclaw-bimanual-teleop-")
        self._temp_dirs.extend([robot_dir, teleop_dir])
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = _arm_id(arm)
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            if src.exists():
                shutil.copy2(src, Path(robot_dir) / f"bimanual_{side}.json")
        for side, arm in [("left", leaders[0]), ("right", leaders[1])]:
            serial = _arm_id(arm)
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            if src.exists():
                shutil.copy2(src, Path(teleop_dir) / f"bimanual_{side}.json")
        return robot_dir, teleop_dir

    # -- Subprocess management -----------------------------------------

    def save_episode(self) -> None:
        """Send 'right arrow' to subprocess → save current episode, start next."""
        self._send_key(b"\x1b[C")
        logger.info("Sent save-episode signal (right arrow)")

    def discard_episode(self) -> None:
        """Send 'left arrow' to subprocess → discard and rerecord current episode."""
        self._send_key(b"\x1b[D")
        logger.info("Sent discard-episode signal (left arrow)")

    def _send_key(self, key: bytes) -> None:
        if self._process_stdin is None:
            raise RuntimeError("No subprocess stdin available")
        self._process_stdin.write(key)
        self._process_stdin.flush()

    def _launch_subprocess(self, argv: list[str]) -> None:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # Auto-confirm calibration prompts (Press ENTER...)
        if proc.stdin:
            proc.stdin.write(b"\n\n\n\n")
            proc.stdin.flush()
        self._process_stdin = proc.stdin
        self._process_pid = proc.pid
        logger.info("Launched subprocess pid={}: {}", proc.pid, " ".join(argv[:5]))

    def _kill_subprocess(self) -> None:
        if self._process_stdin is not None:
            try:
                self._process_stdin.close()
            except OSError:
                pass
            self._process_stdin = None
        pid = self._process_pid
        if pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGINT)
                logger.info("Sent SIGINT to process group pid={}", pid)
            except (ProcessLookupError, PermissionError):
                pass
            self._process_pid = None
        self._cleanup_temp_dirs()

    def _cleanup_temp_dirs(self) -> None:
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()
