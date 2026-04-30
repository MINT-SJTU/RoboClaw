"""Tests for ROS 2 simulation doctor manifests."""

from __future__ import annotations

from roboclaw.embodied.ros2.discovery import CommandResult, Ros2Discovery, Ros2ListResult
from roboclaw.embodied.simulation.doctor import (
    DEFAULT_PROFILE,
    SimulationDoctor,
    SimulationDoctorProfile,
    TransformCheck,
)


_PROFILE = SimulationDoctorProfile(
    packages=("nav2_bringup",),
    nodes=("/bt_navigator",),
    topics=("/cmd_vel", "/odom", "/scan", "/tf"),
    actions=("/navigate_to_pose",),
    services=("/reset_simulation",),
    transforms=(
        TransformCheck("map", "odom"),
        TransformCheck("odom", "base_footprint"),
    ),
)


class _FakeDiscovery:
    def __init__(
        self,
        *,
        ros2_cli: bool = True,
        packages: dict[str, bool] | None = None,
        nodes: tuple[str, ...] = ("/bt_navigator",),
        topics: tuple[str, ...] = ("/cmd_vel", "/odom", "/scan", "/tf"),
        actions: tuple[str, ...] = ("/navigate_to_pose",),
        services: tuple[str, ...] = ("/reset_simulation",),
        transforms: dict[str, bool] | None = None,
        failed_queries: set[str] | None = None,
    ) -> None:
        self._ros2_cli = ros2_cli
        self._packages = packages or {"nav2_bringup": True}
        self._nodes = nodes
        self._topics = topics
        self._actions = actions
        self._services = services
        self._transforms = transforms or {
            "map->odom": True,
            "odom->base_footprint": True,
        }
        self._failed_queries = failed_queries or set()

    def ros2_cli_available(self) -> bool:
        return self._ros2_cli

    def packages_present(self, package_names):
        return {package: self._packages.get(package, False) for package in package_names}

    def node_list(self) -> Ros2ListResult:
        return self._list_result("nodes", ("ros2", "node", "list"), self._nodes)

    def topic_list(self) -> Ros2ListResult:
        return self._list_result("topics", ("ros2", "topic", "list"), self._topics)

    def action_list(self) -> Ros2ListResult:
        return self._list_result("actions", ("ros2", "action", "list"), self._actions)

    def service_list(self) -> Ros2ListResult:
        return self._list_result("services", ("ros2", "service", "list"), self._services)

    def transform_available(self, target_frame: str, source_frame: str) -> bool:
        return self._transforms.get(f"{target_frame}->{source_frame}", False)

    def _list_result(
        self,
        query: str,
        command: tuple[str, ...],
        items: tuple[str, ...],
    ) -> Ros2ListResult:
        if query in self._failed_queries:
            return Ros2ListResult(
                command=command,
                items=(),
                returncode=2,
                stdout="",
                stderr=f"{query} query failed",
            )
        return Ros2ListResult(
            command=command,
            items=items,
            returncode=0,
            stdout="\n".join(items),
            stderr="",
        )


def test_simulation_doctor_ready_manifest() -> None:
    manifest = SimulationDoctor(discovery=_FakeDiscovery(), profile=_PROFILE).check()

    assert manifest["profile_id"] == "turtlebot3_gazebo_nav2"
    assert manifest["status"] == {
        "environment_installed": True,
        "runtime_up": True,
        "tf_ready": True,
        "nav_ready": True,
    }
    assert manifest["decision"] == "ready_for_smoke_test"
    assert manifest["next_steps"] == ["Run the navigation smoke test."]
    assert manifest["errors"] == []


def test_simulation_doctor_nav_ready_requires_services() -> None:
    manifest = SimulationDoctor(
        discovery=_FakeDiscovery(services=()),
        profile=_PROFILE,
    ).check()

    assert manifest["checks"]["services"] == {"/reset_simulation": False}
    assert manifest["status"]["nav_ready"] is False
    assert manifest["decision"] == "needs_reconfiguration"
    assert manifest["errors"] == [{
        "category": "navigation",
        "message": "Navigation capabilities are missing: /reset_simulation",
    }]


def test_simulation_doctor_default_tf_checks_include_map_to_odom() -> None:
    keys = {transform.key for transform in DEFAULT_PROFILE.transforms}

    assert "map->odom" in keys
    assert "odom->base_footprint" in keys


def test_simulation_doctor_reports_query_failure_without_runtime_false_positive() -> None:
    manifest = SimulationDoctor(
        discovery=_FakeDiscovery(failed_queries={"topics"}),
        profile=_PROFILE,
    ).check()

    assert manifest["status"]["runtime_up"] is False
    assert manifest["decision"] == "blocked"
    assert manifest["errors"] == [{
        "category": "discovery",
        "query": "topics",
        "message": "Failed to query ROS 2 topics: topics query failed",
        "command": "ros2 topic list",
    }]


def test_simulation_doctor_blocks_when_ros2_cli_missing() -> None:
    manifest = SimulationDoctor(
        discovery=_FakeDiscovery(ros2_cli=False),
        profile=_PROFILE,
    ).check()

    assert manifest["status"] == {
        "environment_installed": False,
        "runtime_up": False,
        "tf_ready": False,
        "nav_ready": False,
    }
    assert manifest["decision"] == "blocked"
    assert manifest["errors"][0]["category"] == "environment"


def test_ros2_discovery_list_result_preserves_query_errors() -> None:
    def runner(argv, timeout_s):
        assert argv == ["ros2", "topic", "list"]
        return CommandResult(7, "", "daemon unavailable")

    result = Ros2Discovery(runner=runner).topic_list()

    assert result.items == ()
    assert result.returncode == 7
    assert result.error == "daemon unavailable"
