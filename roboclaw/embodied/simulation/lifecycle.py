"""Simulation lifecycle helpers for isolated navigation workflows.

This module manages the simulation runtime boundary without touching the arm
manifest or EmbodiedService. It wraps the repo-local shell entrypoints for
bringup and uses ROS 2 CLI calls for reset operations.
"""

from __future__ import annotations

import os
import pwd
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from roboclaw.embodied.ros2.discovery import CommandResult
from roboclaw.embodied.simulation.state import get_roboclaw_home


ProcessFactory = Callable[[Sequence[str], Path, Mapping[str, str]], subprocess.Popen[str]]
ShellRunner = Callable[[Sequence[str], Path, Mapping[str, str], float | None], CommandResult]
_ORPHAN_MATCHES = (
    "/opt/ros/humble/bin/ros2 launch turtlebot3_gazebo",
    "/opt/ros/humble/bin/ros2 launch turtlebot3_navigation2",
    "/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher",
    "/opt/ros/humble/lib/rviz2/rviz2",
    "gzserver ",
    "gzclient ",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_process_factory(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": dict(env),
        "text": True,
        "start_new_session": True,
    }
    log_path = env.get("ROBOCLAW_SIM_LOG_PATH")
    log_handle = None
    if log_path:
        log_file = Path(log_path).expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a", encoding="utf-8")
        popen_kwargs["stdout"] = log_handle
        popen_kwargs["stderr"] = subprocess.STDOUT
    try:
        return subprocess.Popen(list(argv), **popen_kwargs)
    finally:
        if log_handle is not None:
            log_handle.close()


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


class SimulationLifecycle:
    """Manage isolated simulation bringup and runtime control."""

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        process_factory: ProcessFactory | None = None,
        shell_runner: ShellRunner | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
        self._process_factory = process_factory or _default_process_factory
        self._shell_runner = shell_runner or _default_shell_runner
        self._process: subprocess.Popen[str] | Any | None = None
        self._command: tuple[str, ...] = ()
        self._mode: str | None = None
        self._log_path: Path | None = None

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def status(self) -> dict[str, Any]:
        process = self._process
        running = process is not None and process.poll() is None
        return {
            "tracked": process is not None,
            "running": running,
            "pid": getattr(process, "pid", None) if process is not None else None,
            "mode": self._mode,
            "command": list(self._command),
            "log_path": str(self._log_path) if self._log_path is not None else None,
            "returncode": process.poll() if process is not None else None,
        }

    def bringup(
        self,
        *,
        mode: str = "nav",
        map_path: str | Path | None = None,
        world_launch: str | None = None,
        model: str | None = None,
        ros_domain_id: int | None = None,
        rviz: bool = True,
    ) -> dict[str, Any]:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"gazebo", "nav", "nav-only"}:
            return {
                "ok": False,
                "message": f"Unsupported bringup mode: {mode}",
                "process": self.status(),
            }

        if self.status()["running"]:
            return {
                "ok": True,
                "already_running": True,
                "message": "A tracked simulation process is already running.",
                "process": self.status(),
            }

        script_path = self._repo_root / "robotics" / "scripts" / "run_sim.sh"
        if not script_path.is_file():
            return {
                "ok": False,
                "message": f"Simulation entrypoint not found: {script_path}",
                "process": self.status(),
            }

        argv = ["bash", str(script_path), "--mode", normalized_mode]
        if world_launch:
            argv.extend(["--world", world_launch])
        if map_path and normalized_mode in {"nav", "nav-only"}:
            argv.extend(["--map", str(self._resolve_path(map_path))])
        if model:
            argv.extend(["--model", str(model)])
        if ros_domain_id is not None:
            argv.extend(["--ros-domain-id", str(ros_domain_id)])
        if normalized_mode in {"nav", "nav-only"}:
            argv.append("--rviz" if rviz else "--no-rviz")

        env = dict(os.environ)
        log_path = self._new_log_path(normalized_mode)
        env["ROBOCLAW_SIM_LOG_PATH"] = str(log_path)
        process = self._process_factory(argv, self._repo_root, env)
        self._process = process
        self._command = tuple(argv)
        self._mode = normalized_mode
        self._log_path = log_path
        return {
            "ok": True,
            "message": "Simulation bringup started.",
            "process": self.status(),
        }

    def shutdown(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        process = self._process
        previous = self.status()
        if process is None:
            orphan_cleanup = self._cleanup_orphans(grace_s=min(timeout_s, 2.0))
            message = "No tracked simulation process is running."
            if orphan_cleanup["terminated"] or orphan_cleanup["killed"]:
                message = "No tracked simulation process was running. Cleaned up orphaned simulation processes."
            return {
                "ok": True,
                "message": message,
                "process": previous,
                "orphan_cleanup": orphan_cleanup,
            }

        if process.poll() is not None:
            self._clear_process()
            orphan_cleanup = self._cleanup_orphans(grace_s=min(timeout_s, 2.0))
            message = "Tracked simulation process had already exited."
            if orphan_cleanup["terminated"] or orphan_cleanup["killed"]:
                message = "Tracked simulation process had already exited. Cleaned up orphaned simulation processes."
            return {
                "ok": True,
                "message": message,
                "process": previous,
                "orphan_cleanup": orphan_cleanup,
            }

        try:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                process.terminate()
            process.wait(timeout=timeout_s)
            message = "Simulation process stopped."
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (AttributeError, ProcessLookupError):
                process.kill()
            process.wait()
            message = "Simulation process was force-stopped after timeout."

        self._clear_process()
        orphan_cleanup = self._cleanup_orphans(grace_s=min(timeout_s, 2.0))
        if orphan_cleanup["terminated"] or orphan_cleanup["killed"]:
            message += " Orphaned simulation processes were also cleaned up."
        return {
            "ok": True,
            "message": message,
            "process": previous,
            "orphan_cleanup": orphan_cleanup,
        }

    def reset_world(
        self,
        *,
        service_name: str = "/reset_simulation",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        attempted: list[dict[str, Any]] = []
        for candidate in self._reset_service_candidates(service_name):
            result = self._call_ros_command(
                ["ros2", "service", "call", candidate, "std_srvs/srv/Empty", "{}"],
                timeout_s=timeout_s,
            )
            attempted.append({
                "service_name": candidate,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            })
            if result.ok:
                return {
                    "ok": True,
                    "message": f"Reset service succeeded: {candidate}",
                    "service_name": candidate,
                    "attempts": attempted,
                }

        last = attempted[-1] if attempted else {"service_name": service_name, "stderr": "No attempts made."}
        return {
            "ok": False,
            "message": f"Reset service failed: {last['service_name']}",
            "service_name": service_name,
            "attempts": attempted,
        }

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (self._repo_root / candidate).resolve()

    def _call_ros_command(
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
    def _reset_service_candidates(service_name: str) -> tuple[str, ...]:
        requested = service_name or "/reset_simulation"
        if requested == "/reset_simulation":
            return ("/reset_simulation", "/reset_world")
        return (requested,)

    def _new_log_path(self, mode: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = get_roboclaw_home() / "workspace" / "embodied" / "logs" / "simulation"
        return log_dir / f"{timestamp}-{mode}.log"

    def _cleanup_orphans(self, *, grace_s: float = 1.0) -> dict[str, list[int]]:
        candidates = self._list_orphan_pids()
        terminated: list[int] = []
        killed: list[int] = []

        for pid in candidates:
            try:
                os.kill(pid, signal.SIGTERM)
                terminated.append(pid)
            except ProcessLookupError:
                continue

        if terminated and grace_s > 0:
            time.sleep(min(grace_s, 1.0))

        remaining = set(self._list_orphan_pids())
        for pid in terminated:
            if pid not in remaining:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                continue

        return {
            "terminated": terminated,
            "killed": killed,
        }

    def _list_orphan_pids(self) -> list[int]:
        current_user = pwd.getpwuid(os.getuid()).pw_name
        tracked_pid = getattr(self._process, "pid", None)
        try:
            completed = subprocess.run(
                ["ps", "-eo", "pid=,user=,args="],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []
        if completed.returncode != 0:
            return []

        pids: list[int] = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue
            pid_text, user, args = parts
            if user != current_user:
                continue
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if tracked_pid is not None and pid == tracked_pid:
                continue
            if self._matches_orphan_args(args):
                pids.append(pid)
        return pids

    @staticmethod
    def _matches_orphan_args(args: str) -> bool:
        if any(pattern in args for pattern in _ORPHAN_MATCHES):
            return True
        if "component_container_isolated" in args and "nav2_container" in args:
            return True
        return False

    def _clear_process(self) -> None:
        self._process = None
        self._command = ()
        self._mode = None
        self._log_path = None
