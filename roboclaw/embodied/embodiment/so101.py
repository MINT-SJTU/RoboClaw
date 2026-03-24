"""SO101 LeRobot 0.5.0 command builder."""

from __future__ import annotations

import json
from pathlib import Path


class SO101Controller:
    """Builds LeRobot CLI commands for SO101 robot arm.

    All methods take explicit params — the caller (tool.py) resolves
    setup.json into concrete values before calling these.
    """

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

    def calibrate(self, arm_type: str, arm_port: str, calibration_dir: str) -> list[str]:
        """Build calibration command for one arm.

        arm_type: "so101_follower" or "so101_leader"
        For follower uses --robot.* prefix, for leader uses --teleop.* prefix.
        """
        prefix = self._arm_prefix(arm_type)
        return [
            "lerobot-calibrate",
            *self._arm_args(prefix, arm_type, arm_port, calibration_dir),
        ]

    def teleoperate(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        teleop_type: str, teleop_port: str, teleop_cal_dir: str,
    ) -> list[str]:
        """Build teleoperation command (single follower + single leader)."""
        return [
            "lerobot-teleoperate",
            *self._arm_args("robot", robot_type, robot_port, robot_cal_dir),
            *self._arm_args("teleop", teleop_type, teleop_port, teleop_cal_dir),
        ]

    def teleoperate_bimanual(
        self,
        left_robot: dict, right_robot: dict,
        left_teleop: dict, right_teleop: dict,
    ) -> list[str]:
        """Build bimanual teleoperation command (2 followers + 2 leaders)."""
        return [
            "lerobot-teleoperate",
            "--robot.type=bi_so_follower",
            *self._bimanual_arm_args("robot", left_robot, right_robot),
            "--teleop.type=bi_so_leader",
            *self._bimanual_arm_args("teleop", left_teleop, right_teleop),
        ]

    def record(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        teleop_type: str, teleop_port: str, teleop_cal_dir: str,
        cameras: dict[str, dict],
        repo_id: str, task: str,
        fps: int = 30, num_episodes: int = 10,
    ) -> list[str]:
        """Build recording command (follower + leader + cameras + dataset)."""
        return [
            "lerobot-record",
            *self._arm_args("robot", robot_type, robot_port, robot_cal_dir),
            *self._arm_args("teleop", teleop_type, teleop_port, teleop_cal_dir),
            f"--robot.cameras={json.dumps(cameras)}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.single_task={task}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={num_episodes}",
        ]

    def record_bimanual(
        self,
        left_robot: dict, right_robot: dict,
        left_teleop: dict, right_teleop: dict,
        cameras: dict[str, dict],
        repo_id: str, task: str,
        fps: int = 30, num_episodes: int = 10,
    ) -> list[str]:
        """Build bimanual recording command (2 followers + 2 leaders + cameras)."""
        return [
            "lerobot-record",
            "--robot.type=bi_so_follower",
            *self._bimanual_arm_args("robot", left_robot, right_robot),
            f"--robot.cameras={json.dumps(cameras)}",
            "--teleop.type=bi_so_leader",
            *self._bimanual_arm_args("teleop", left_teleop, right_teleop),
            f"--dataset.repo_id={repo_id}",
            f"--dataset.single_task={task}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={num_episodes}",
        ]

    def run_policy(
        self,
        robot_type: str, robot_port: str, robot_cal_dir: str,
        cameras: dict[str, dict],
        policy_path: str,
        repo_id: str = "local/eval",
        num_episodes: int = 1,
    ) -> list[str]:
        """Build policy execution command (follower only, no teleop)."""
        return [
            "lerobot-record",
            *self._arm_args("robot", robot_type, robot_port, robot_cal_dir),
            f"--robot.cameras={json.dumps(cameras)}",
            f"--policy.path={Path(policy_path).expanduser()}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.num_episodes={num_episodes}",
        ]

    def _arm_prefix(self, arm_type: str) -> str:
        if "leader" in arm_type:
            return "teleop"
        if "follower" in arm_type:
            return "robot"
        raise ValueError(f"Unsupported arm type: {arm_type}")

    def _arm_args(self, prefix: str, arm_type: str, port: str, cal_dir: str) -> list[str]:
        return [
            f"--{prefix}.type={arm_type}",
            f"--{prefix}.port={port}",
            f"--{prefix}.calibration_dir={Path(cal_dir).expanduser()}",
        ]

    def _bimanual_arm_args(self, prefix: str, left: dict, right: dict) -> list[str]:
        """Build --{prefix}.left_arm_config.* and --{prefix}.right_arm_config.* args."""
        args: list[str] = []
        for side, arm in [("left", left), ("right", right)]:
            base = f"--{prefix}.{side}_arm_config"
            args.append(f"{base}.port={arm['port']}")
            args.append(f"{base}.calibration_dir={Path(arm['calibration_dir']).expanduser()}")
        return args
