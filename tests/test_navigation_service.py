"""Tests for the isolated navigation service layer."""

from __future__ import annotations

from roboclaw.embodied.navigation.service import NavigationService


def _doctor_result(*, nav_ready: bool = True, tf_ready: bool = True, runtime_up: bool = True) -> dict:
    return {
        "action": "doctor",
        "ok": True,
        "state_path": "/tmp/simulation_state.json",
        "lifecycle": {"running": runtime_up},
        "manifest": {
            "mode": "simulation",
            "robot": "turtlebot3",
            "simulator": "gazebo",
            "checks": {
                "topics": {"/cmd_vel": runtime_up, "/odom": runtime_up, "/scan": runtime_up, "/tf": runtime_up},
                "actions": {"/navigate_to_pose": nav_ready},
            },
            "status": {
                "environment_installed": True,
                "runtime_up": runtime_up,
                "tf_ready": tf_ready,
                "nav_ready": nav_ready,
            },
        },
    }


class _FakeSimulationService:
    def __init__(self, doctor_result: dict):
        self._doctor_result = doctor_result

    def doctor(self, *, profile_id=None):
        return self._doctor_result


class _FakeNavClient:
    def __init__(self):
        self.navigate_calls = []
        self.follow_calls = []
        self.cancel_calls = []

    def action_names(self):
        return ["/navigate_to_pose", "/follow_waypoints"]

    def navigate_to_pose(self, **kwargs):
        self.navigate_calls.append(kwargs)
        return {
            "ok": True,
            "goal_succeeded": True,
            "goal_accepted": True,
            "goal_status": "SUCCEEDED",
            "goal": {"pose": kwargs},
            "command": ["ros2", "action", "send_goal"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "metrics": {"number_of_recoveries": 0, "distance_remaining": 0.0},
        }

    def follow_waypoints(self, **kwargs):
        self.follow_calls.append(kwargs)
        return {
            "ok": True,
            "goal_succeeded": True,
            "goal_accepted": True,
            "goal_status": "SUCCEEDED",
            "goal": {"poses": kwargs["poses"]},
            "command": ["ros2", "action", "send_goal"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "metrics": {"current_waypoint": 1, "missed_waypoints": []},
        }

    def cancel_goals(self, **kwargs):
        self.cancel_calls.append(kwargs)
        return {"ok": True, "attempts": []}


def test_navigation_service_nav_status_reports_available_actions() -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    result = service.nav_status()

    assert result["ok"] is True
    assert result["checks"]["navigate_to_pose"] is True
    assert result["checks"]["follow_waypoints"] is True


def test_navigation_service_blocks_goal_when_nav_not_ready() -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result(nav_ready=False)),
        nav_client=_FakeNavClient(),
    )

    result = service.navigate_to_pose(x=1.0, y=2.0)

    assert result["ok"] is False
    assert result["decision"] == "blocked"


def test_navigation_service_smoke_test_passive_ready() -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    result = service.smoke_test()

    assert result["ok"] is True
    assert result["passed"] is True
    assert result["mode"] == "passive"
    assert result["decision"] == "ready_for_navigation_task"


def test_navigation_service_follow_waypoints_and_collect_metrics() -> None:
    nav_client = _FakeNavClient()
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=nav_client,
    )

    result = service.follow_waypoints(waypoints=[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}])
    metrics = service.collect_metrics()

    assert result["succeeded"] is True
    assert nav_client.follow_calls[0]["poses"][0]["x"] == 1.0
    assert metrics["ok"] is True
    assert metrics["source_action"] == "follow_waypoints"


def test_navigation_service_marks_missing_terminal_status_inconclusive() -> None:
    class _InconclusiveNavClient(_FakeNavClient):
        def navigate_to_pose(self, **kwargs):
            self.navigate_calls.append(kwargs)
            return {
                "ok": True,
                "goal_succeeded": False,
                "goal_accepted": True,
                "goal_status": None,
                "goal": {"pose": kwargs},
                "command": ["ros2", "action", "send_goal"],
                "returncode": 0,
                "stdout": "Goal accepted with ID: 123\n",
                "stderr": "",
                "metrics": {},
            }

    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_InconclusiveNavClient(),
    )

    result = service.navigate_to_pose(x=1.0, y=2.0)

    assert result["ok"] is True
    assert result["succeeded"] is False
    assert result["decision"] == "goal_result_inconclusive"
    assert result["metrics"]["goal_accepted"] is True
    assert "goal_status" not in result["metrics"]


def test_navigation_service_cancel_nav_delegates_to_client() -> None:
    nav_client = _FakeNavClient()
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=nav_client,
    )

    result = service.cancel_nav(timeout_s=2.0)

    assert result["ok"] is True
    assert nav_client.cancel_calls == [{"timeout_s": 2.0}]
