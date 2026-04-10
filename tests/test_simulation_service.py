"""Tests for the isolated simulation service layer."""

from __future__ import annotations

from roboclaw.embodied.simulation.service import SimulationService
from roboclaw.embodied.simulation.state import load_simulation_state


def _manifest(
    *,
    environment_installed: bool = True,
    runtime_up: bool = False,
    tf_ready: bool = False,
    nav_ready: bool = False,
    decision: str = "blocked",
    errors: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "profile_id": "turtlebot3_gazebo_nav2",
        "mode": "simulation",
        "robot": "turtlebot3",
        "simulator": "gazebo",
        "checks": {
            "system": {"ros2_cli": environment_installed},
            "packages": {"nav2_bringup": environment_installed},
            "nodes": {"/bt_navigator": nav_ready},
            "topics": {"/cmd_vel": runtime_up, "/odom": runtime_up, "/scan": runtime_up, "/tf": runtime_up},
            "actions": {"/navigate_to_pose": nav_ready},
            "services": {},
            "tf": {"map->odom": tf_ready, "odom->base_footprint": tf_ready},
        },
        "status": {
            "environment_installed": environment_installed,
            "runtime_up": runtime_up,
            "tf_ready": tf_ready,
            "nav_ready": nav_ready,
        },
        "decision": decision,
        "next_steps": ["next"],
        "errors": errors or [],
    }


class _FakeLifecycle:
    def __init__(self) -> None:
        self._running = False
        self.bringup_calls: list[dict] = []
        self.shutdown_calls = 0
        self.reset_calls: list[dict] = []

    def status(self) -> dict:
        return {
            "tracked": self._running,
            "running": self._running,
            "pid": 999 if self._running else None,
            "mode": "nav" if self._running else None,
            "command": [],
            "returncode": None,
        }

    def bringup(self, **kwargs) -> dict:
        self.bringup_calls.append(kwargs)
        self._running = True
        return {
            "ok": True,
            "message": "started",
            "process": self.status(),
        }

    def shutdown(self) -> dict:
        self.shutdown_calls += 1
        self._running = False
        return {
            "ok": True,
            "message": "stopped",
            "process": self.status(),
        }

    def reset_world(self, *, service_name: str, timeout_s: float) -> dict:
        self.reset_calls.append({"service_name": service_name, "timeout_s": timeout_s})
        return {
            "ok": True,
            "message": "reset",
            "service_name": service_name,
            "attempts": [],
        }


def test_simulation_service_doctor_syncs_state(tmp_path) -> None:
    state_path = tmp_path / "simulation_state.json"
    service = SimulationService(
        lifecycle=_FakeLifecycle(),
        doctor_runner=lambda profile_id: _manifest(
            runtime_up=True,
            tf_ready=True,
            nav_ready=True,
            decision="ready_for_smoke_test",
        ),
        state_path=state_path,
    )

    result = service.doctor()

    assert result["ok"] is True
    assert state_path.exists()
    loaded = load_simulation_state(state_path)
    assert loaded["last_doctor"]["decision"] == "ready_for_smoke_test"
    assert not (tmp_path / "manifest.json").exists()


def test_simulation_service_bringup_blocks_when_environment_missing(tmp_path) -> None:
    lifecycle = _FakeLifecycle()
    service = SimulationService(
        lifecycle=lifecycle,
        doctor_runner=lambda profile_id: _manifest(
            environment_installed=False,
            decision="blocked",
            errors=[{"category": "environment", "message": "missing deps"}],
        ),
        state_path=tmp_path / "simulation_state.json",
    )

    result = service.bringup()

    assert result["ok"] is False
    assert lifecycle.bringup_calls == []


def test_simulation_service_bringup_switches_to_nav_only_when_runtime_exists(tmp_path) -> None:
    lifecycle = _FakeLifecycle()
    state_path = tmp_path / "simulation_state.json"
    service = SimulationService(
        lifecycle=lifecycle,
        doctor_runner=lambda profile_id: _manifest(
            environment_installed=True,
            runtime_up=True,
            tf_ready=False,
            nav_ready=False,
            decision="needs_reconfiguration",
            errors=[{"category": "navigation", "message": "nav missing"}],
        ),
        state_path=state_path,
    )

    result = service.bringup(
        mode="nav",
        map_path="robotics/ros_ws/src/roboclaw_tb3_sim/maps/map.yaml",
        world_launch="turtlebot3_world.launch.py",
    )

    assert result["ok"] is True
    assert result["lifecycle_mode"] == "nav-only"
    assert lifecycle.bringup_calls[0]["mode"] == "nav-only"
    loaded = load_simulation_state(state_path)
    assert loaded["paths"]["launch"] == "robotics/scripts/run_sim.sh"
    assert loaded["paths"]["map"].endswith("maps/map.yaml")


def test_simulation_service_shutdown_and_reset_delegate(tmp_path) -> None:
    lifecycle = _FakeLifecycle()
    service = SimulationService(
        lifecycle=lifecycle,
        doctor_runner=lambda profile_id: _manifest(),
        state_path=tmp_path / "simulation_state.json",
    )

    reset_result = service.reset_world(service_name="/reset_world", timeout_s=2.5)
    shutdown_result = service.shutdown()

    assert reset_result["ok"] is True
    assert lifecycle.reset_calls == [{"service_name": "/reset_world", "timeout_s": 2.5}]
    assert shutdown_result["ok"] is True
    assert lifecycle.shutdown_calls == 1
