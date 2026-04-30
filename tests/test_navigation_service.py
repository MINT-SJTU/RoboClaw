"""Tests for the isolated navigation service layer."""

from __future__ import annotations

import json
from pathlib import Path

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


def _write_semantic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "map.pgm").write_text(
        "\n".join(
            [
                "P2",
                "8 8",
                "255",
                "0 0 0 0 0 0 0 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 0 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    map_path = tmp_path / "map.yaml"
    map_path.write_text(
        "\n".join(
            [
                "image: map.pgm",
                "resolution: 1.0",
                "origin: [0.0, 0.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    graph_path = tmp_path / "map.semantic.json"
    graph_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "test_graph",
                "map_id": "test_map",
                "map_path": "map.yaml",
                "places": [
                    {
                        "id": "bedroom",
                        "type": "room",
                        "regions": [
                            {
                                "id": "main",
                                "frame_id": "map",
                                "polygon": [
                                    {"x": 1.0, "y": 1.0},
                                    {"x": 6.0, "y": 1.0},
                                    {"x": 6.0, "y": 6.0},
                                    {"x": 1.0, "y": 6.0},
                                ],
                            }
                        ],
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return graph_path, map_path


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


def test_navigation_service_resolves_place_from_semantic_graph(tmp_path: Path) -> None:
    graph_path, map_path = _write_semantic_fixture(tmp_path)
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    result = service.resolve_place(
        place="bedroom",
        map_path=map_path,
        semantic_graph_path=graph_path,
    )

    assert result["ok"] is True
    assert result["decision"] == "place_grounded"
    assert result["goal"]["place_id"] == "bedroom"
    assert result["goal"]["source"] == "region_free_space"


def test_navigation_service_resolves_project_paths_from_non_repo_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    monkeypatch.chdir(tmp_path)
    resolved = service._resolve_project_path("pyproject.toml")

    assert resolved.name == "pyproject.toml"
    assert resolved.is_file()


def test_navigation_service_ignores_blank_semantic_path_overrides() -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    result = service.resolve_place(
        place="kitchen",
        map_id="house",
        map_path="",
        semantic_graph_path="   ",
    )

    assert result["ok"] is True
    assert result["decision"] == "place_grounded"
    assert result["semantic_graph"]["path"].endswith("map_house.semantic.json")


def test_navigation_service_rejects_empty_project_path() -> None:
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_FakeNavClient(),
    )

    try:
        service._resolve_project_path("")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected empty path to be rejected.")


def test_navigation_service_navigates_to_resolved_place(tmp_path: Path) -> None:
    graph_path, map_path = _write_semantic_fixture(tmp_path)
    nav_client = _FakeNavClient()
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=nav_client,
    )

    result = service.navigate_to_place(
        place="bedroom",
        map_path=map_path,
        semantic_graph_path=graph_path,
        feedback=False,
    )

    assert result["ok"] is True
    assert result["succeeded"] is True
    assert result["grounding"]["goal"]["place_id"] == "bedroom"
    assert nav_client.navigate_calls[0]["x"] == result["grounding"]["goal"]["pose"]["x"]
    assert nav_client.navigate_calls[0]["feedback"] is False


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


def test_navigation_service_compacts_long_nav_output() -> None:
    class _VerboseNavClient(_FakeNavClient):
        def navigate_to_pose(self, **kwargs):
            self.navigate_calls.append(kwargs)
            return {
                "ok": True,
                "goal_succeeded": True,
                "goal_accepted": True,
                "goal_status": "SUCCEEDED",
                "goal": {"pose": kwargs},
                "command": ["ros2", "action", "send_goal", "-f"],
                "returncode": 0,
                "stdout": "feedback\n" * 500 + "Goal finished with status: SUCCEEDED\n",
                "stderr": "warning\n" * 300,
                "metrics": {"number_of_recoveries": 0},
            }

    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=_VerboseNavClient(),
    )

    result = service.navigate_to_pose(x=1.0, y=2.0)

    assert result["ok"] is True
    assert "stdout" not in result
    assert "stderr" not in result
    assert result["output"]["truncated"] is True
    assert result["output"]["stdout_chars"] > len(result["output"]["stdout_tail"])
    assert len(result["output"]["stdout_tail"]) <= 800


def test_navigation_service_cancel_nav_delegates_to_client() -> None:
    nav_client = _FakeNavClient()
    service = NavigationService(
        simulation_service=_FakeSimulationService(_doctor_result()),
        nav_client=nav_client,
    )

    result = service.cancel_nav(timeout_s=2.0)

    assert result["ok"] is True
    assert nav_client.cancel_calls == [{"timeout_s": 2.0}]
