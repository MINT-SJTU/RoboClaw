"""Nav2 CLI wrappers for the isolated navigation slice."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from roboclaw.embodied.ros2.discovery import CommandResult


RosCommandRunner = Callable[[Sequence[str], Path, Mapping[str, str], float | None], CommandResult]

_GOAL_STATUS_RE = re.compile(r"Goal finished with status:\s*([A-Z_]+)")
_INT_FIELD_RES = {
    "number_of_recoveries": re.compile(r"number_of_recoveries:\s*(\d+)"),
    "current_waypoint": re.compile(r"current_waypoint:\s*(\d+)"),
}
_FLOAT_FIELD_RES = {
    "distance_remaining": re.compile(r"distance_remaining:\s*([-+0-9.eE]+)"),
}
_MISSED_RE = re.compile(r"missed_waypoints:\s*\[([^\]]*)\]")
_CANCEL_RETURN_RE = re.compile(r"return_code:\s*(\d+)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_shell_runner(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float | None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=dict(env),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(124, stdout, stderr)


def _yaw_to_quaternion(yaw: float) -> dict[str, float]:
    half = yaw / 2.0
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half),
        "w": math.cos(half),
    }


class Nav2Client:
    """Best-effort Nav2 interaction layer using ROS 2 CLI commands."""

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        shell_runner: RosCommandRunner | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
        self._shell_runner = shell_runner or _default_shell_runner

    def action_names(self) -> list[str]:
        result = self._run_ros_command(["ros2", "action", "list"])
        if not result.ok:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def action_available(self, action_name: str) -> bool:
        return action_name in set(self.action_names())

    def navigate_to_pose(
        self,
        *,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        behavior_tree: str = "",
        feedback: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        goal = {
            "pose": {
                "header": {"frame_id": frame_id},
                "pose": {
                    "position": {"x": float(x), "y": float(y), "z": 0.0},
                    "orientation": _yaw_to_quaternion(float(yaw)),
                },
            },
            "behavior_tree": behavior_tree,
        }
        return self._send_goal(
            action_name="/navigate_to_pose",
            action_type="nav2_msgs/action/NavigateToPose",
            goal=goal,
            feedback=feedback,
            timeout_s=timeout_s,
        )

    def follow_waypoints(
        self,
        *,
        poses: Sequence[Mapping[str, Any]],
        frame_id: str = "map",
        feedback: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        goal = {
            "poses": [self._pose_stamped_dict(pose, frame_id=frame_id) for pose in poses],
        }
        return self._send_goal(
            action_name="/follow_waypoints",
            action_type="nav2_msgs/action/FollowWaypoints",
            goal=goal,
            feedback=feedback,
            timeout_s=timeout_s,
        )

    def cancel_goals(
        self,
        *,
        action_names: Sequence[str] = ("/navigate_to_pose", "/follow_waypoints"),
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        payload = {
            "goal_info": {
                "goal_id": {"uuid": [0] * 16},
                "stamp": {"sec": 0, "nanosec": 0},
            }
        }
        attempts: list[dict[str, Any]] = []
        any_ok = False
        for action_name in action_names:
            service_name = f"{action_name}/_action/cancel_goal"
            result = self._run_ros_command(
                [
                    "ros2",
                    "service",
                    "call",
                    service_name,
                    "action_msgs/srv/CancelGoal",
                    json.dumps(payload),
                ],
                timeout_s=timeout_s,
            )
            output = f"{result.stdout}\n{result.stderr}"
            match = _CANCEL_RETURN_RE.search(output)
            cancel_code = int(match.group(1)) if match else None
            ok = result.ok and cancel_code == 0
            any_ok = any_ok or ok
            attempts.append({
                "action_name": action_name,
                "service_name": service_name,
                "ok": ok,
                "returncode": result.returncode,
                "cancel_return_code": cancel_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            })
        return {
            "ok": any_ok,
            "attempts": attempts,
        }

    def _send_goal(
        self,
        *,
        action_name: str,
        action_type: str,
        goal: dict[str, Any],
        feedback: bool,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        argv = ["ros2", "action", "send_goal"]
        if feedback:
            argv.append("-f")
        argv.extend([action_name, action_type, json.dumps(goal)])
        result = self._run_ros_command(argv, timeout_s=timeout_s)
        output = f"{result.stdout}\n{result.stderr}"
        goal_status = _last_group(_GOAL_STATUS_RE, output)
        metrics = self._parse_metrics(output)
        accepted = "Goal accepted" in output
        succeeded = result.ok and accepted and goal_status in {"SUCCEEDED", "STATUS_SUCCEEDED"}
        return {
            "ok": result.ok,
            "goal_succeeded": succeeded,
            "goal_accepted": accepted,
            "goal_status": goal_status,
            "action_name": action_name,
            "action_type": action_type,
            "goal": goal,
            "command": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "metrics": metrics,
        }

    def _run_ros_command(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float | None = None,
    ) -> CommandResult:
        ros_setup = self._repo_root / "robotics" / "ros_ws" / "install" / "setup.bash"
        shell_parts = [
            "set +u",
            "source /opt/ros/humble/setup.bash",
            "source /usr/share/gazebo/setup.sh",
            f"source {shlex.quote(str(ros_setup))}",
            " ".join(shlex.quote(part) for part in argv),
        ]
        shell_command = " && ".join(shell_parts)
        return self._shell_runner(
            ["bash", "-lc", shell_command],
            self._repo_root,
            dict(os.environ),
            timeout_s,
        )

    @staticmethod
    def _pose_stamped_dict(pose: Mapping[str, Any], *, frame_id: str) -> dict[str, Any]:
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        yaw = float(pose.get("yaw", 0.0))
        local_frame = str(pose.get("frame_id", frame_id))
        return {
            "header": {"frame_id": local_frame},
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": _yaw_to_quaternion(yaw),
            },
        }

    @staticmethod
    def _parse_metrics(output: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key, pattern in _INT_FIELD_RES.items():
            value = _last_group(pattern, output)
            if value is not None:
                metrics[key] = int(value)
        for key, pattern in _FLOAT_FIELD_RES.items():
            value = _last_group(pattern, output)
            if value is not None:
                metrics[key] = float(value)
        missed = _MISSED_RE.search(output)
        if missed is not None:
            content = missed.group(1).strip()
            if not content:
                metrics["missed_waypoints"] = []
            else:
                metrics["missed_waypoints"] = [int(part.strip()) for part in content.split(",") if part.strip()]
        return metrics


def _last_group(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1)
