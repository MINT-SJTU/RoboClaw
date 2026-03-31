"""In-process robot session for web-based data collection."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.embodied.setup import load_setup


# ---------------------------------------------------------------------------
# Arm config builders
# ---------------------------------------------------------------------------

def _build_follower_config(arm: dict, cameras: dict):
    """Build a follower robot config from a setup.json arm entry."""
    arm_type = arm["type"]
    port = arm["port"]
    cal_dir = arm.get("calibration_dir", "")

    if arm_type == "koch_follower":
        from lerobot.robots.koch_follower.config_koch_follower import KochFollowerConfig
        return KochFollowerConfig(port=port, calibration_dir=cal_dir, cameras=cameras)
    if arm_type == "so101_follower":
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        return SO101FollowerConfig(port=port, calibration_dir=cal_dir, cameras=cameras)
    raise ValueError(f"Unsupported follower arm type: {arm_type}")


def _build_leader_config(arm: dict):
    """Build a leader (teleop) config from a setup.json arm entry."""
    arm_type = arm["type"]
    port = arm["port"]
    cal_dir = arm.get("calibration_dir", "")

    if arm_type == "koch_leader":
        from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
        return KochLeaderConfig(port=port, calibration_dir=cal_dir)
    if arm_type == "so101_leader":
        from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
        return SO101LeaderConfig(port=port, calibration_dir=cal_dir)
    raise ValueError(f"Unsupported leader arm type: {arm_type}")


def _build_camera_configs(cameras_setup: list[dict]) -> dict:
    """Build camera config dict from setup.json camera entries."""
    from lerobot.cameras.opencv import OpenCVCameraConfig

    cameras = {}
    for cam in cameras_setup:
        cameras[cam["alias"]] = OpenCVCameraConfig(
            port=cam["port"],
            width=cam.get("width", 640),
            height=cam.get("height", 480),
            fps=cam.get("fps", 30),
        )
    return cameras


def _find_arm_by_type_suffix(arms: list[dict], suffix: str) -> dict | None:
    """Find the first arm whose type ends with *suffix* (e.g. '_follower')."""
    for arm in arms:
        if arm.get("type", "").endswith(suffix):
            return arm
    return None


# ---------------------------------------------------------------------------
# RobotSession
# ---------------------------------------------------------------------------

class RobotSession:
    """Manages robot lifecycle: connect -> teleop -> record -> disconnect."""

    _STATES = ("disconnected", "connected", "teleoperating", "recording")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "disconnected"

        # Robot / teleop objects (set on connect)
        self._robot: Any = None
        self._teleop: Any = None

        # Processors
        self._teleop_proc: Any = None
        self._robot_proc: Any = None
        self._obs_proc: Any = None

        # Teleop thread
        self._teleop_stop = threading.Event()
        self._teleop_thread: threading.Thread | None = None
        self._teleop_fps: int = 30

        # Recording state
        self._record_stop = threading.Event()
        self._record_thread: threading.Thread | None = None
        self._dataset: Any = None
        self._task: str = ""
        self._target_episodes: int = 0
        self._episode_count: int = 0
        self._frame_count: int = 0
        self._recording_fps: int = 30

        # Latest observation for camera streaming
        self._latest_obs: dict[str, Any] = {}
        self._setup: dict[str, Any] = {}

    # -- State helpers -------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def _require_state(self, *allowed: str) -> None:
        if self._state not in allowed:
            raise RuntimeError(
                f"Operation requires state {allowed}, but current state is '{self._state}'"
            )

    # -- Connect / Disconnect ------------------------------------------------

    def connect(self, setup: dict | None = None) -> None:
        """Connect to robot and teleop using setup.json configuration."""
        with self._lock:
            self._require_state("disconnected")
            setup = setup or load_setup()
            self._setup = setup

            arms = setup.get("arms", [])
            follower_arm = _find_arm_by_type_suffix(arms, "_follower")
            leader_arm = _find_arm_by_type_suffix(arms, "_leader")
            if not follower_arm:
                raise ValueError("No follower arm configured in setup.json")
            if not leader_arm:
                raise ValueError("No leader arm configured in setup.json")

            cameras_setup = setup.get("cameras", [])
            camera_configs = _build_camera_configs(cameras_setup)

            follower_cfg = _build_follower_config(follower_arm, camera_configs)
            leader_cfg = _build_leader_config(leader_arm)

            from lerobot.robots import make_robot_from_config
            from lerobot.teleoperators import make_teleoperator_from_config

            logger.info("Connecting follower: {}", follower_arm.get("alias"))
            self._robot = make_robot_from_config(follower_cfg)
            self._robot.connect()

            logger.info("Connecting leader: {}", leader_arm.get("alias"))
            self._teleop = make_teleoperator_from_config(leader_cfg)
            self._teleop.connect()

            from lerobot.processor import make_default_processors
            self._teleop_proc, self._robot_proc, self._obs_proc = make_default_processors()

            self._state = "connected"
            logger.info("Robot session connected")

    def disconnect(self) -> None:
        """Disconnect robot and teleop, stopping any running threads."""
        with self._lock:
            if self._state == "disconnected":
                return

            if self._state == "recording":
                self._stop_recording_locked()
            if self._state == "teleoperating":
                self._stop_teleop_locked()

            if self._robot is not None:
                self._robot.disconnect()
                self._robot = None
            if self._teleop is not None:
                self._teleop.disconnect()
                self._teleop = None

            self._teleop_proc = None
            self._robot_proc = None
            self._obs_proc = None
            self._state = "disconnected"
            logger.info("Robot session disconnected")

    # -- Teleoperation -------------------------------------------------------

    def start_teleop(self, fps: int = 30) -> None:
        """Start control thread that reads teleop and sends to robot."""
        with self._lock:
            self._require_state("connected")
            self._teleop_fps = fps
            self._teleop_stop.clear()
            self._teleop_thread = threading.Thread(
                target=self._teleop_loop, daemon=True, name="teleop",
            )
            self._state = "teleoperating"
            self._teleop_thread.start()
            logger.info("Teleoperation started at {} fps", fps)

    def stop_teleop(self) -> None:
        """Stop teleoperation thread."""
        with self._lock:
            self._require_state("teleoperating")
            self._stop_teleop_locked()
            self._state = "connected"
            logger.info("Teleoperation stopped")

    def _stop_teleop_locked(self) -> None:
        """Internal: stop teleop thread (caller holds lock)."""
        self._teleop_stop.set()
        if self._teleop_thread is not None:
            self._teleop_thread.join(timeout=5.0)
            self._teleop_thread = None

    def _teleop_loop(self) -> None:
        """Background thread: read leader, send to follower at target fps."""
        from lerobot.utils.robot_utils import busy_wait

        dt = 1.0 / self._teleop_fps
        while not self._teleop_stop.is_set():
            start = time.perf_counter()
            obs = self._robot.get_observation()
            action = self._teleop.get_action()
            action_processed = self._teleop_proc((action, obs))
            robot_action = self._robot_proc((action_processed, obs))
            self._robot.send_action(robot_action)
            self._latest_obs = obs
            busy_wait(dt - (time.perf_counter() - start))

    # -- Recording -----------------------------------------------------------

    def start_recording(
        self,
        dataset_name: str,
        task: str,
        fps: int = 30,
        num_episodes: int = 10,
    ) -> None:
        """Create a LeRobotDataset and start the recording thread."""
        with self._lock:
            self._require_state("connected", "teleoperating")
            was_teleoperating = self._state == "teleoperating"
            if was_teleoperating:
                self._stop_teleop_locked()

            self._task = task
            self._target_episodes = num_episodes
            self._episode_count = 0
            self._frame_count = 0
            self._recording_fps = fps

            self._dataset = self._create_dataset(dataset_name, fps)

            self._record_stop.clear()
            self._record_thread = threading.Thread(
                target=self._record_loop, daemon=True, name="record",
            )
            self._state = "recording"
            self._record_thread.start()
            logger.info(
                "Recording started: dataset={}, task={}, fps={}, episodes={}",
                dataset_name, task, fps, num_episodes,
            )

    def stop_recording(self) -> None:
        """Stop recording thread. Does NOT save the current episode."""
        with self._lock:
            self._require_state("recording")
            self._stop_recording_locked()
            self._state = "connected"
            logger.info("Recording stopped")

    def _stop_recording_locked(self) -> None:
        """Internal: stop recording thread (caller holds lock)."""
        self._record_stop.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=5.0)
            self._record_thread = None

    def save_episode(self) -> None:
        """Save the current recording episode and prepare for the next."""
        with self._lock:
            self._require_state("recording")
            if self._dataset is None:
                raise RuntimeError("No dataset available")
            self._dataset.save_episode()
            self._episode_count += 1
            self._frame_count = 0
            logger.info("Episode {} saved", self._episode_count)

    def discard_episode(self) -> None:
        """Discard the current recording episode."""
        with self._lock:
            self._require_state("recording")
            if self._dataset is None:
                raise RuntimeError("No dataset available")
            self._dataset.clear_episode_buffer()
            self._frame_count = 0
            logger.info("Episode discarded")

    def _create_dataset(self, dataset_name: str, fps: int) -> Any:
        """Create a new LeRobotDataset for recording."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import (
            aggregate_pipeline_dataset_features,
            create_initial_features,
        )
        from lerobot.datasets.utils import combine_feature_dicts

        datasets_root = self._setup.get("datasets", {}).get("root", "")
        if not datasets_root:
            raise ValueError("No datasets root configured in setup.json")

        root = Path(datasets_root)
        root.mkdir(parents=True, exist_ok=True)

        robot_features = create_initial_features(self._robot)
        teleop_features = create_initial_features(self._teleop)
        features = aggregate_pipeline_dataset_features(
            combine_feature_dicts(robot_features, teleop_features)
        )

        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=fps,
            root=str(root / dataset_name),
            features=features,
        )
        return dataset

    def _record_loop(self) -> None:
        """Background thread: capture obs + action, build frames, add to dataset."""
        from lerobot.datasets.utils import build_dataset_frame
        from lerobot.utils.robot_utils import busy_wait

        dt = 1.0 / self._recording_fps
        while not self._record_stop.is_set():
            start = time.perf_counter()
            obs = self._robot.get_observation()
            action = self._teleop.get_action()

            obs_processed = self._obs_proc(obs)
            action_processed = self._teleop_proc((action, obs))
            robot_action = self._robot_proc((action_processed, obs))
            self._robot.send_action(robot_action)

            obs_frame = build_dataset_frame(
                self._dataset.features, obs_processed, prefix="observation.images.",
            )
            act_frame = build_dataset_frame(
                self._dataset.features, robot_action, prefix="action.",
            )
            frame = {**obs_frame, **act_frame, "task": self._task}
            self._dataset.add_frame(frame)
            self._frame_count += 1
            self._latest_obs = obs

            busy_wait(dt - (time.perf_counter() - start))

    # -- Status & Camera -----------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current session status."""
        return {
            "state": self._state,
            "episode_count": self._episode_count,
            "frame_count": self._frame_count,
            "target_episodes": self._target_episodes,
            "recording_fps": self._recording_fps,
            "teleop_fps": self._teleop_fps,
            "task": self._task,
            "dataset": self._dataset.repo_id if self._dataset else None,
        }

    def get_camera_frame(self, camera_name: str) -> bytes:
        """Return the latest camera frame as JPEG bytes."""
        import cv2
        import numpy as np

        obs = self._latest_obs
        key = f"observation.images.{camera_name}"
        if key not in obs:
            raise KeyError(f"Camera '{camera_name}' not found in latest observation")
        frame = obs[key]
        if isinstance(frame, np.ndarray):
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        raise TypeError(f"Unexpected frame type: {type(frame)}")
