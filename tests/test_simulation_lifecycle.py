"""Tests for isolated simulation lifecycle helpers."""

from __future__ import annotations

import signal
from types import SimpleNamespace

from roboclaw.embodied.ros2.discovery import CommandResult
from roboclaw.embodied.simulation import lifecycle as lifecycle_module
from roboclaw.embodied.simulation.lifecycle import SimulationLifecycle


class _FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self._returncode = None
        self.wait_calls: list[float | None] = []

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0

    def kill(self) -> None:
        self._returncode = -9


def test_lifecycle_bringup_builds_nav_command(tmp_path) -> None:
    script = tmp_path / "robotics" / "scripts" / "run_sim.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    map_path = tmp_path / "map.yaml"
    map_path.write_text("image: map.pgm\n", encoding="utf-8")

    seen = {}
    fake_process = _FakeProcess()

    def process_factory(argv, cwd, env):
        seen["argv"] = list(argv)
        seen["cwd"] = cwd
        seen["env"] = env
        return fake_process

    lifecycle = SimulationLifecycle(repo_root=tmp_path, process_factory=process_factory)
    result = lifecycle.bringup(
        mode="nav",
        map_path=map_path,
        world_launch="custom_world.launch.py",
        model="burger",
        ros_domain_id=7,
        rviz=False,
    )

    assert result["ok"] is True
    assert seen["cwd"] == tmp_path
    assert seen["argv"] == [
        "bash",
        str(script),
        "--mode",
        "nav",
        "--world",
        "custom_world.launch.py",
        "--map",
        str(map_path.resolve()),
        "--model",
        "burger",
        "--ros-domain-id",
        "7",
        "--no-rviz",
    ]
    assert seen["env"]["ROBOCLAW_SIM_LOG_PATH"].endswith("-nav.log")
    assert lifecycle.status()["running"] is True
    assert lifecycle.status()["log_path"].endswith("-nav.log")


def test_lifecycle_shutdown_stops_tracked_process(tmp_path, monkeypatch) -> None:
    script = tmp_path / "robotics" / "scripts" / "run_sim.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_process = _FakeProcess(pid=5151)

    lifecycle = SimulationLifecycle(
        repo_root=tmp_path,
        process_factory=lambda argv, cwd, env: fake_process,
    )
    lifecycle.bringup(mode="gazebo")

    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(lifecycle_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(lifecycle_module.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        lifecycle_module.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_name="tester"),
    )
    monkeypatch.setattr(lifecycle_module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(
        lifecycle_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = lifecycle.shutdown(timeout_s=3.0)

    assert result["ok"] is True
    assert fake_process.wait_calls == [3.0]
    assert calls == [(5151, signal.SIGTERM)]
    assert lifecycle.status()["tracked"] is False


def test_lifecycle_shutdown_cleans_orphaned_simulation_processes(monkeypatch) -> None:
    lifecycle = SimulationLifecycle(repo_root=".")
    kills: list[tuple[int, signal.Signals]] = []
    snapshots = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "2000 tester /usr/bin/python3 /opt/ros/humble/bin/ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py\n"
                    "2001 tester /opt/ros/humble/lib/rclcpp_components/component_container_isolated --ros-args -r __node:=nav2_container\n"
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="2001 tester /opt/ros/humble/lib/rclcpp_components/component_container_isolated --ros-args -r __node:=nav2_container\n",
                stderr="",
            ),
        ]
    )

    monkeypatch.setattr(lifecycle_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        lifecycle_module.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_name="tester"),
    )
    monkeypatch.setattr(lifecycle_module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(lifecycle_module.subprocess, "run", lambda *args, **kwargs: next(snapshots))
    monkeypatch.setattr(lifecycle_module.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    result = lifecycle.shutdown(timeout_s=1.0)

    assert result["ok"] is True
    assert result["orphan_cleanup"] == {
        "terminated": [2000, 2001],
        "killed": [2001],
    }
    assert kills == [
        (2000, signal.SIGTERM),
        (2001, signal.SIGTERM),
        (2001, signal.SIGKILL),
    ]


def test_lifecycle_reset_world_falls_back_to_reset_world_service(tmp_path) -> None:
    attempts: list[str] = []

    def shell_runner(argv, cwd, env, timeout_s):
        attempts.append(argv[2])
        if "/reset_simulation" in argv[2]:
            return CommandResult(1, "", "service unavailable")
        return CommandResult(0, "success", "")

    lifecycle = SimulationLifecycle(repo_root=tmp_path, shell_runner=shell_runner)
    result = lifecycle.reset_world()

    assert result["ok"] is True
    assert result["service_name"] == "/reset_world"
    assert len(attempts) == 2
    assert "/reset_simulation" in attempts[0]
    assert "/reset_world" in attempts[1]
