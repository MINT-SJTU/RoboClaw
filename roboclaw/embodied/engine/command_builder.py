"""LeRobot CLI command builder for robot arms."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from roboclaw.embodied.embodiment.arm.base import ServoArmSpec
from roboclaw.embodied.embodiment.arm.registry import get_arm_spec


class ArmCommandBuilder:
    """Builds LeRobot CLI commands for any supported robot arm spec.

    All methods take explicit params — the caller resolves
    the manifest config into concrete values before calling these.
    """

    def __init__(self, spec: ServoArmSpec | None = None) -> None:
        self._spec = spec

    def doctor(self) -> list[str]:
        """Check lerobot, list supported robots, motors, and connected devices."""
        script = (
            "import lerobot, glob, os; "
            "print(f'lerobot_version: {lerobot.__version__}'); "
            "print(f'supported_robots: {lerobot.available_robots}'); "
            "print(f'supported_motors: {lerobot.available_motors}'); "
            "print(f'supported_cameras: {lerobot.available_cameras}'); "
            "by_id = sorted(glob.glob('/dev/serial/by-id/*')); "
            "pairs = [(p, os.path.realpath(p)) for p in by_id]; "
            "print(f'connected_ports_by_id: {pairs}')"
        )
        return ["python3", "-c", script]

    def calibrate(
        self, arm_type: str, arm_port: str, calibration_dir: str, arm_id: str,
    ) -> list[str]:
        """Build calibration command for one arm.

        For follower uses --robot.* prefix, for leader uses --teleop.* prefix.
        """
        prefix = self._arm_prefix(arm_type)
        return [
            *self._wrapper_args("calibrate"),
            *self._arm_args(prefix, arm_type, arm_port, calibration_dir, arm_id),
        ]

    def teleoperate(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        robot_id: str,
        teleop_type: str, teleop_port: str, teleop_cal_dir: str,
        teleop_id: str,
        cameras: dict[str, dict] | None = None,
        display_data: bool = False,
        display_ip: str = "",
        display_port: int = 0,
    ) -> list[str]:
        """Build teleoperation command (single follower + single leader)."""
        argv = [
            *self._wrapper_args("teleoperate"),
            *self._arm_args("robot", robot_type, robot_port, robot_cal_dir, robot_id),
            *self._arm_args("teleop", teleop_type, teleop_port, teleop_cal_dir, teleop_id),
        ]
        if cameras:
            argv.append(f"--robot.cameras={json.dumps(cameras)}")
        argv.extend(self._display_args(display_data, display_ip, display_port))
        return argv

    def teleoperate_bimanual(
        self,
        robot_id: str, robot_cal_dir: str,
        left_robot: dict, right_robot: dict,
        teleop_id: str, teleop_cal_dir: str,
        left_teleop: dict, right_teleop: dict,
        cameras: dict[str, dict] | None = None,
        display_data: bool = False,
        display_ip: str = "",
        display_port: int = 0,
    ) -> list[str]:
        """Build bimanual teleoperation command (2 followers + 2 leaders)."""
        spec = self._require_bimanual_spec()
        argv = [
            *self._wrapper_args("teleoperate"),
            f"--robot.type={spec.bimanual_follower_type}",
            f"--robot.id={robot_id}",
            f"--robot.calibration_dir={Path(robot_cal_dir).expanduser()}",
            *self._bimanual_arm_args("robot", left_robot, right_robot, cameras),
            f"--teleop.type={spec.bimanual_leader_type}",
            f"--teleop.id={teleop_id}",
            f"--teleop.calibration_dir={Path(teleop_cal_dir).expanduser()}",
            *self._bimanual_arm_args("teleop", left_teleop, right_teleop),
        ]
        argv.extend(self._display_args(display_data, display_ip, display_port))
        return argv

    def record(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        robot_id: str,
        teleop_type: str, teleop_port: str, teleop_cal_dir: str,
        teleop_id: str,
        cameras: dict[str, dict],
        repo_id: str, task: str,
        dataset_root: str,
        push_to_hub: bool = False,
        fps: int = 30, num_episodes: int = 10,
        episode_time_s: int | None = None,
        reset_time_s: int | None = None,
        resume: bool = False,
        display_data: bool = False,
        display_ip: str = "",
        display_port: int = 0,
    ) -> list[str]:
        """Build recording command (follower + leader + cameras + dataset)."""
        argv = self._wrapper_args("record")
        argv.extend(self._arm_args("robot", robot_type, robot_port, robot_cal_dir, robot_id))
        argv.extend(self._arm_args("teleop", teleop_type, teleop_port, teleop_cal_dir, teleop_id))
        if cameras:
            argv.append(f"--robot.cameras={json.dumps(cameras)}")
        argv.extend(self._dataset_args(
            repo_id, dataset_root, task, push_to_hub, fps, num_episodes, episode_time_s,
            reset_time_s=reset_time_s, resume=resume,
        ))
        argv.extend(self._display_args(display_data, display_ip, display_port))
        return argv

    def record_bimanual(
        self,
        robot_id: str, robot_cal_dir: str,
        left_robot: dict, right_robot: dict,
        teleop_id: str, teleop_cal_dir: str,
        left_teleop: dict, right_teleop: dict,
        cameras: dict[str, dict],
        repo_id: str, task: str,
        dataset_root: str,
        push_to_hub: bool = False,
        fps: int = 30, num_episodes: int = 10,
        episode_time_s: int | None = None,
        reset_time_s: int | None = None,
        resume: bool = False,
        display_data: bool = False,
        display_ip: str = "",
        display_port: int = 0,
    ) -> list[str]:
        """Build bimanual recording command (2 followers + 2 leaders + cameras)."""
        spec = self._require_bimanual_spec()
        argv = [
            *self._wrapper_args("record"),
            f"--robot.type={spec.bimanual_follower_type}",
            f"--robot.id={robot_id}",
            f"--robot.calibration_dir={Path(robot_cal_dir).expanduser()}",
            *self._bimanual_arm_args("robot", left_robot, right_robot, cameras),
            f"--teleop.type={spec.bimanual_leader_type}",
            f"--teleop.id={teleop_id}",
            f"--teleop.calibration_dir={Path(teleop_cal_dir).expanduser()}",
            *self._bimanual_arm_args("teleop", left_teleop, right_teleop),
            *self._dataset_args(
                repo_id, dataset_root, task, push_to_hub, fps, num_episodes, episode_time_s,
                reset_time_s=reset_time_s, resume=resume,
            ),
        ]
        argv.extend(self._display_args(display_data, display_ip, display_port))
        return argv

    def replay(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        robot_id: str,
        repo_id: str,
        dataset_root: str,
        episode: int,
        fps: int = 30,
    ) -> list[str]:
        """Build replay command for one follower arm."""
        return [
            *self._wrapper_args("replay"),
            *self._arm_args("robot", robot_type, robot_port, robot_cal_dir, robot_id),
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={Path(dataset_root).expanduser()}",
            f"--dataset.episode={episode}",
            f"--dataset.fps={fps}",
        ]

    def replay_bimanual(
        self,
        robot_id: str,
        robot_cal_dir: str,
        left_robot: dict,
        right_robot: dict,
        repo_id: str,
        dataset_root: str,
        episode: int,
        fps: int = 30,
    ) -> list[str]:
        """Build replay command for two follower arms."""
        spec = self._require_bimanual_spec()
        return [
            *self._wrapper_args("replay"),
            f"--robot.type={spec.bimanual_follower_type}",
            f"--robot.id={robot_id}",
            f"--robot.calibration_dir={Path(robot_cal_dir).expanduser()}",
            *self._bimanual_arm_args("robot", left_robot, right_robot),
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={Path(dataset_root).expanduser()}",
            f"--dataset.episode={episode}",
            f"--dataset.fps={fps}",
        ]

    def run_policy(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        robot_id: str,
        cameras: dict[str, dict],
        policy_path: str,
        repo_id: str = "local/eval",
        dataset_root: str = "",
        task: str = "eval",
        num_episodes: int = 1,
        resume: bool = False,
    ) -> list[str]:
        """Build policy execution command (follower only, no teleop)."""
        argv = self._wrapper_args("record")
        argv.extend(self._arm_args("robot", robot_type, robot_port, robot_cal_dir, robot_id))
        if cameras:
            argv.append(f"--robot.cameras={json.dumps(cameras)}")
        argv.extend(self._policy_args(
            policy_path, repo_id, task, num_episodes, dataset_root, resume=resume,
        ))
        return argv

    def run_policy_bimanual(
        self,
        robot_id: str, robot_cal_dir: str,
        left_robot: dict, right_robot: dict,
        cameras: dict[str, dict],
        policy_path: str,
        repo_id: str = "local/eval",
        dataset_root: str = "",
        task: str = "eval",
        num_episodes: int = 1,
        resume: bool = False,
    ) -> list[str]:
        """Build bimanual policy execution command (2 followers, no teleop)."""
        spec = self._require_bimanual_spec()
        argv = [
            *self._wrapper_args("record"),
            f"--robot.type={spec.bimanual_follower_type}",
            f"--robot.id={robot_id}",
            f"--robot.calibration_dir={Path(robot_cal_dir).expanduser()}",
            *self._bimanual_arm_args("robot", left_robot, right_robot, cameras),
            *self._policy_args(
                policy_path, repo_id, task, num_episodes, dataset_root, resume=resume,
            ),
        ]
        return argv

    # -- Private helpers ---------------------------------------------------

    def _require_bimanual_spec(self) -> ServoArmSpec:
        if self._spec is None:
            raise ValueError("ArmCommandBuilder needs a spec for bimanual commands.")
        if not self._spec.supports_bimanual:
            raise ValueError(f"{self._spec.name} arms do not support bimanual mode.")
        return self._spec

    def _policy_args(
        self,
        policy_path: str, repo_id: str, task: str,
        num_episodes: int, dataset_root: str,
        resume: bool = False,
    ) -> list[str]:
        args = [
            f"--policy.path={Path(policy_path).expanduser()}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.single_task={task}",
            "--dataset.push_to_hub=false",
            f"--dataset.num_episodes={num_episodes}",
        ]
        if dataset_root:
            args.append(f"--dataset.root={Path(dataset_root).expanduser()}")
        if resume:
            args.append("--resume=true")
        return args

    def _dataset_args(
        self,
        repo_id: str, dataset_root: str, task: str,
        push_to_hub: bool, fps: int, num_episodes: int,
        episode_time_s: int | None = None,
        reset_time_s: int | None = None,
        resume: bool = False,
    ) -> list[str]:
        args = [
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={Path(dataset_root).expanduser()}",
            f"--dataset.push_to_hub={str(push_to_hub).lower()}",
            f"--dataset.single_task={task}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={num_episodes}",
            "--dataset.vcodec=h264",
            "--dataset.streaming_encoding=true",
        ]
        if episode_time_s is not None:
            args.append(f"--dataset.episode_time_s={episode_time_s}")
        if reset_time_s is not None:
            args.append(f"--dataset.reset_time_s={reset_time_s}")
        if resume:
            args.append("--resume=true")
        return args

    def _wrapper_args(self, action: str) -> list[str]:
        return [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", action]

    def _arm_prefix(self, arm_type: str) -> str:
        if "leader" in arm_type:
            return "teleop"
        if "follower" in arm_type:
            return "robot"
        raise ValueError(f"Unsupported arm type: {arm_type}")

    def _arm_args(
        self, prefix: str, arm_type: str, port: str, cal_dir: str, arm_id: str,
    ) -> list[str]:
        return [
            f"--{prefix}.type={arm_type}",
            f"--{prefix}.id={arm_id}",
            f"--{prefix}.port={port}",
            f"--{prefix}.calibration_dir={Path(cal_dir).expanduser()}",
        ]

    def _display_args(
        self, display_data: bool, display_ip: str, display_port: int,
    ) -> list[str]:
        if not display_data:
            return []
        args = ["--display_data=true"]
        if display_ip:
            args.append(f"--display_ip={display_ip}")
        if display_port:
            args.append(f"--display_port={display_port}")
        return args

    def _bimanual_arm_args(
        self,
        prefix: str,
        left: dict,
        right: dict,
        cameras: dict[str, dict] | None = None,
    ) -> list[str]:
        args: list[str] = []
        for side, arm in [("left", left), ("right", right)]:
            args.append(f"--{prefix}.{side}_arm_config.port={arm['port']}")
        if cameras:
            args.append(f"--{prefix}.left_arm_config.cameras={json.dumps(cameras)}")
        return args



def builder_for_arms(arms: list[dict[str, Any]]) -> ArmCommandBuilder:
    """Create an ArmCommandBuilder with the spec derived from arm types."""
    if not arms:
        return ArmCommandBuilder()
    spec = get_arm_spec(arms[0]["type"])
    return ArmCommandBuilder(spec=spec)
