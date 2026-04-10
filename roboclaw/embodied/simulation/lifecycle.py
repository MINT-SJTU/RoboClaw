"""Simulation lifecycle helpers for isolated navigation workflows.

This module manages the simulation runtime boundary without touching the arm
manifest or EmbodiedService. It wraps the repo-local shell entrypoints for
bringup and uses ROS 2 CLI calls for reset operations.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from roboclaw.embodied.ros2.discovery import CommandResult


ProcessFactory = Callable[[Sequence[str], Path, Mapping[str, str]], subprocess.Popen[str]]
ShellRunner = Callable[[Sequence[str], Path, Mapping[str, str], float | None], CommandResult]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_process_factory(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        start_new_session=True,
    )


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
        process = self._process_factory(argv, self._repo_root, env)
        self._process = process
        self._command = tuple(argv)
        self._mode = normalized_mode
        return {
            "ok": True,
            "message": "Simulation bringup started.",
            "process": self.status(),
        }

    def shutdown(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        process = self._process
        previous = self.status()
        if process is None:
            return {
                "ok": True,
                "message": "No tracked simulation process is running.",
                "process": previous,
            }

        if process.poll() is not None:
            self._clear_process()
            return {
                "ok": True,
                "message": "Tracked simulation process had already exited.",
                "process": previous,
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
        return {
            "ok": True,
            "message": message,
            "process": previous,
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

    def _clear_process(self) -> None:
        self._process = None
        self._command = ()
        self._mode = None
