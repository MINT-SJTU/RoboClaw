"""ROS 2 environment discovery helpers for simulation workflows."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result wrapper used by ROS 2 discovery."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class Ros2ListResult:
    """Structured result for ROS graph list commands."""

    command: tuple[str, ...]
    items: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def error(self) -> str:
        if self.ok:
            return ""
        message = self.stderr.strip() or self.stdout.strip()
        if message:
            return message
        return f"Command exited with code {self.returncode}."


Runner = Callable[[Sequence[str], float | None], CommandResult]


def _default_runner(argv: Sequence[str], timeout_s: float | None = None) -> CommandResult:
    """Run a command and normalize errors into CommandResult."""
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            124,
            _timeout_text(exc.stdout),
            _timeout_text(exc.stderr),
        )


def _timeout_text(value: str | bytes | None) -> str:
    """Return timeout output as text across Python/subprocess variants."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _split_lines(text: str) -> list[str]:
    """Split CLI output into non-empty stripped lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


class Ros2Discovery:
    """Stateless ROS 2 discovery workflow.

    This mirrors the HardwareDiscovery shape at the orchestration boundary, but
    each method is intentionally side-effect-light: it runs one ROS 2 CLI query
    and returns structured data for the simulation doctor to aggregate.
    """

    def __init__(self, runner: Runner | None = None, *, default_timeout_s: float = 5.0) -> None:
        self._runner = runner or _default_runner
        self._default_timeout_s = default_timeout_s

    def ros2_cli_available(self) -> bool:
        """Return whether the ros2 CLI is available on PATH."""
        return shutil.which("ros2") is not None

    def package_available(self, package_name: str) -> bool:
        """Return whether a ROS 2 package is visible in the sourced environment."""
        if not package_name:
            return False
        return self._run(["ros2", "pkg", "prefix", package_name]).ok

    def package_prefix(self, package_name: str) -> str | None:
        """Return a package prefix, or None when the package is unavailable."""
        if not package_name:
            return None
        result = self._run(["ros2", "pkg", "prefix", package_name])
        if not result.ok:
            return None
        lines = _split_lines(result.stdout)
        return lines[0] if lines else None

    def nodes(self) -> list[str]:
        """Return currently visible ROS graph node names."""
        return list(self.node_list().items)

    def node_list(self) -> Ros2ListResult:
        """Return node names plus command status metadata."""
        return self._list(["ros2", "node", "list"])

    def topics(self) -> list[str]:
        """Return currently visible ROS topic names."""
        return list(self.topic_list().items)

    def topic_list(self) -> Ros2ListResult:
        """Return topic names plus command status metadata."""
        return self._list(["ros2", "topic", "list"])

    def actions(self) -> list[str]:
        """Return currently visible ROS action names."""
        return list(self.action_list().items)

    def action_list(self) -> Ros2ListResult:
        """Return action names plus command status metadata."""
        return self._list(["ros2", "action", "list"])

    def services(self) -> list[str]:
        """Return currently visible ROS service names."""
        return list(self.service_list().items)

    def service_list(self) -> Ros2ListResult:
        """Return service names plus command status metadata."""
        return self._list(["ros2", "service", "list"])

    def has_topic(self, topic_name: str) -> bool:
        """Return whether a topic is visible in the ROS graph."""
        return topic_name in self.topics()

    def has_action(self, action_name: str) -> bool:
        """Return whether an action is visible in the ROS graph."""
        return action_name in self.actions()

    def has_service(self, service_name: str) -> bool:
        """Return whether a service is visible in the ROS graph."""
        return service_name in self.services()

    def topics_present(self, topic_names: Sequence[str]) -> dict[str, bool]:
        """Return per-topic presence for a set of expected topics."""
        discovered = set(self.topics())
        return {topic: topic in discovered for topic in topic_names}

    def actions_present(self, action_names: Sequence[str]) -> dict[str, bool]:
        """Return per-action presence for a set of expected actions."""
        discovered = set(self.actions())
        return {action: action in discovered for action in action_names}

    def services_present(self, service_names: Sequence[str]) -> dict[str, bool]:
        """Return per-service presence for a set of expected services."""
        discovered = set(self.services())
        return {service: service in discovered for service in service_names}

    def packages_present(self, package_names: Sequence[str]) -> dict[str, bool]:
        """Return per-package availability for a set of expected packages."""
        return {package: self.package_available(package) for package in package_names}

    def tf_topics_ready(self) -> bool:
        """Return whether TF topics are visible in the ROS graph."""
        discovered = set(self.topics())
        return "/tf" in discovered or "/tf_static" in discovered

    def transform_available(
        self,
        target_frame: str,
        source_frame: str,
        *,
        timeout_s: float = 2.0,
    ) -> bool:
        """Best-effort check for a transform using tf2_echo.

        tf2_echo is a streaming command, so a timeout can still mean success if
        it produced transform data before the timeout fired.
        """
        if not target_frame or not source_frame:
            return False
        result = self._run(
            ["ros2", "run", "tf2_ros", "tf2_echo", target_frame, source_frame],
            timeout_s=timeout_s,
        )
        output = f"{result.stdout}\n{result.stderr}"
        if "Translation:" in output or "Rotation:" in output:
            return True
        return result.ok

    def _list(self, argv: Sequence[str]) -> Ros2ListResult:
        """Run a ROS 2 list command and return normalized lines with metadata."""
        result = self._run(argv)
        items = tuple(_split_lines(result.stdout)) if result.ok else ()
        return Ros2ListResult(
            command=tuple(argv),
            items=items,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _run(self, argv: Sequence[str], timeout_s: float | None = None) -> CommandResult:
        """Run a ROS 2 command with the configured default timeout."""
        return self._runner(
            argv,
            self._default_timeout_s if timeout_s is None else timeout_s,
        )
