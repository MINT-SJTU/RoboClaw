"""Standalone navigation service for the isolated simulation slice."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from roboclaw.embodied.navigation.evaluator import NavigationEvaluator
from roboclaw.embodied.navigation.nav2_client import Nav2Client
from roboclaw.embodied.navigation.semantic_goal import DEFAULT_CLEARANCE_M, SemanticGoalResolver
from roboclaw.embodied.navigation.semantic_graph import load_semantic_graph
from roboclaw.embodied.navigation.smoke_test import SmokeTestRunner
from roboclaw.embodied.simulation.maps import get_simulation_map
from roboclaw.embodied.simulation.service import SimulationService


class NavigationService:
    """Navigation runtime helpers built on top of the simulation slice."""

    def __init__(
        self,
        *,
        simulation_service: SimulationService | None = None,
        nav_client: Nav2Client | None = None,
        smoke_test_runner: SmokeTestRunner | None = None,
        evaluator: NavigationEvaluator | None = None,
        semantic_resolver: SemanticGoalResolver | None = None,
    ) -> None:
        self._simulation = simulation_service or SimulationService()
        self._nav_client = nav_client or Nav2Client()
        self._smoke_test = smoke_test_runner or SmokeTestRunner()
        self._evaluator = evaluator or NavigationEvaluator()
        self._semantic_resolver = semantic_resolver or SemanticGoalResolver()
        self._last_report: dict[str, Any] | None = None

    @property
    def simulation(self) -> SimulationService:
        return self._simulation

    def nav_status(self, *, profile_id: str | None = None) -> dict[str, Any]:
        doctor = self._simulation.doctor(profile_id=profile_id)
        manifest = doctor["manifest"]
        available_actions = set(self._nav_client.action_names())
        return {
            "action": "nav_status",
            "ok": True,
            "environment": self._environment_summary(manifest),
            "state_path": doctor["state_path"],
            "lifecycle": doctor["lifecycle"],
            "checks": {
                "navigate_to_pose": "/navigate_to_pose" in available_actions,
                "follow_waypoints": "/follow_waypoints" in available_actions,
            },
            "last_metrics": deepcopy(self._last_report["metrics"]) if self._last_report else {},
            "manifest": manifest,
        }

    def smoke_test(
        self,
        *,
        profile_id: str | None = None,
        goal_x: float | None = None,
        goal_y: float | None = None,
        goal_yaw: float = 0.0,
        frame_id: str = "map",
        feedback: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        doctor = self._simulation.doctor(profile_id=profile_id)
        active_result = None
        if goal_x is not None and goal_y is not None:
            active_result = self._run_nav_action(
                doctor_result=doctor,
                action="navigate_to_pose",
                command_result=self._nav_client.navigate_to_pose(
                    x=goal_x,
                    y=goal_y,
                    yaw=goal_yaw,
                    frame_id=frame_id,
                    feedback=feedback,
                    timeout_s=timeout_s,
                ),
            )
        result = self._smoke_test.run(doctor_result=doctor, active_result=active_result)
        if active_result is not None:
            self._last_report = deepcopy(active_result)
        return result

    def navigate_to_pose(
        self,
        *,
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        behavior_tree: str = "",
        feedback: bool = True,
        timeout_s: float | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        doctor = self._simulation.doctor(profile_id=profile_id)
        if not doctor["manifest"]["status"].get("nav_ready", False):
            return self._blocked("navigate_to_pose", doctor, "Navigation stack is not ready.")
        result = self._run_nav_action(
            doctor_result=doctor,
            action="navigate_to_pose",
            command_result=self._nav_client.navigate_to_pose(
                x=x,
                y=y,
                yaw=yaw,
                frame_id=frame_id,
                behavior_tree=behavior_tree,
                feedback=feedback,
                timeout_s=timeout_s,
            ),
        )
        self._last_report = deepcopy(result)
        return result

    def resolve_place(
        self,
        *,
        place: str,
        map_id: str | None = None,
        map_path: str | Path | None = None,
        semantic_graph_path: str | Path | None = None,
        clearance_m: float = DEFAULT_CLEARANCE_M,
        goal_stride_m: float = 0.10,
    ) -> dict[str, Any]:
        try:
            graph_path, occupancy_map_path = self._semantic_paths(
                map_id=map_id,
                map_path=map_path,
                semantic_graph_path=semantic_graph_path,
            )
            graph = load_semantic_graph(graph_path)
            goal = self._semantic_resolver.resolve(
                graph=graph,
                place_label=place,
                occupancy_map_path=occupancy_map_path,
                clearance_m=clearance_m,
                goal_stride_m=goal_stride_m,
            )
        except (OSError, ValueError) as exc:
            return {
                "action": "resolve_place",
                "ok": False,
                "decision": "semantic_grounding_blocked",
                "message": str(exc),
                "next_steps": [
                    "Create or select a semantic graph for the active map.",
                    "Verify the requested place label exists in the semantic graph.",
                ],
            }

        return {
            "action": "resolve_place",
            "ok": True,
            "decision": "place_grounded",
            "requested_place": place,
            "semantic_graph": {
                "path": str(graph_path),
                "id": graph.graph_id,
                "map_id": graph.map_id,
            },
            "map": {
                "requested_map_id": map_id,
                "occupancy_map_path": (
                    str(occupancy_map_path) if occupancy_map_path else str(graph.resolve_map_path())
                ),
            },
            "goal": goal.to_dict(),
        }

    def navigate_to_place(
        self,
        *,
        place: str,
        map_id: str | None = None,
        map_path: str | Path | None = None,
        semantic_graph_path: str | Path | None = None,
        clearance_m: float = DEFAULT_CLEARANCE_M,
        goal_stride_m: float = 0.10,
        behavior_tree: str = "",
        feedback: bool = True,
        timeout_s: float | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        grounding = self.resolve_place(
            place=place,
            map_id=map_id,
            map_path=map_path,
            semantic_graph_path=semantic_graph_path,
            clearance_m=clearance_m,
            goal_stride_m=goal_stride_m,
        )
        if not grounding["ok"]:
            return {
                "action": "navigate_to_place",
                "ok": False,
                "succeeded": False,
                "decision": grounding["decision"],
                "grounding": grounding,
                "message": grounding["message"],
                "next_steps": grounding["next_steps"],
            }

        pose = grounding["goal"]["pose"]
        navigation = self.navigate_to_pose(
            profile_id=profile_id,
            x=float(pose["x"]),
            y=float(pose["y"]),
            yaw=float(pose.get("yaw", 0.0)),
            frame_id=str(pose.get("frame_id", "map") or "map"),
            behavior_tree=behavior_tree,
            feedback=feedback,
            timeout_s=timeout_s,
        )
        return {
            "action": "navigate_to_place",
            "ok": bool(navigation.get("ok")),
            "succeeded": bool(navigation.get("succeeded", False)),
            "decision": navigation.get("decision"),
            "grounding": grounding,
            "navigation": navigation,
        }

    def follow_waypoints(
        self,
        *,
        waypoints: Sequence[Mapping[str, Any]],
        frame_id: str = "map",
        feedback: bool = True,
        timeout_s: float | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        doctor = self._simulation.doctor(profile_id=profile_id)
        if not doctor["manifest"]["status"].get("nav_ready", False):
            return self._blocked("follow_waypoints", doctor, "Navigation stack is not ready.")
        result = self._run_nav_action(
            doctor_result=doctor,
            action="follow_waypoints",
            command_result=self._nav_client.follow_waypoints(
                poses=waypoints,
                frame_id=frame_id,
                feedback=feedback,
                timeout_s=timeout_s,
            ),
        )
        self._last_report = deepcopy(result)
        return result

    def cancel_nav(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        result = self._nav_client.cancel_goals(timeout_s=timeout_s)
        return {
            "action": "cancel_nav",
            "ok": bool(result.get("ok")),
            "message": "Cancel request sent." if result.get("ok") else "No active navigation goal was canceled.",
            "details": result,
        }

    def collect_metrics(self) -> dict[str, Any]:
        return self._evaluator.collect_metrics(self._last_report)

    def _semantic_paths(
        self,
        *,
        map_id: str | None,
        map_path: str | Path | None,
        semantic_graph_path: str | Path | None,
    ) -> tuple[Path, Path | None]:
        requested_map_id = self._optional_string(map_id)
        selected_map = get_simulation_map(requested_map_id) if requested_map_id else None
        active_paths = self._active_state_paths() if selected_map is None else {}
        if selected_map is None and active_paths.get("map_id"):
            selected_map = get_simulation_map(active_paths["map_id"])

        graph_path = self._optional_path_value(semantic_graph_path)
        if graph_path is None and selected_map is not None:
            graph_path = selected_map.semantic_graph_path
        if graph_path is None:
            graph_path = active_paths.get("semantic_graph") or None
        if graph_path is None:
            raise ValueError("No semantic graph is selected for the active navigation map.")

        occupancy_map_path = self._optional_path_value(map_path)
        if occupancy_map_path is None and selected_map is not None:
            occupancy_map_path = selected_map.path
        if occupancy_map_path is None and active_paths.get("map"):
            occupancy_map_path = active_paths["map"]

        return self._resolve_project_path(graph_path), (
            self._resolve_project_path(occupancy_map_path)
            if occupancy_map_path is not None
            else None
        )

    def _resolve_project_path(self, value: str | Path) -> Path:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Path value must not be empty.")
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        return self._repo_root() / path

    @staticmethod
    def _optional_string(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _optional_path_value(value: str | Path | None) -> str | Path | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    def _repo_root(self) -> Path:
        lifecycle = getattr(self._simulation, "lifecycle", None)
        repo_root = getattr(lifecycle, "repo_root", None)
        if repo_root is not None:
            return Path(repo_root).resolve()
        return Path(__file__).resolve().parents[3]

    def _active_state_paths(self) -> dict[str, str]:
        state_show = getattr(self._simulation, "state_show", None)
        if state_show is None:
            return {}
        result = state_show()
        state = result.get("state", {}) if isinstance(result, dict) else {}
        paths = state.get("paths", {}) if isinstance(state, dict) else {}
        if not isinstance(paths, dict):
            return {}
        return {
            str(key): str(value).strip()
            for key, value in paths.items()
            if value is not None and str(value).strip()
        }

    def _run_nav_action(
        self,
        *,
        doctor_result: dict[str, Any],
        action: str,
        command_result: dict[str, Any],
    ) -> dict[str, Any]:
        return self._evaluator.action_report(
            action=action,
            command_result=command_result,
            environment=self._environment_summary(doctor_result["manifest"]),
        )

    def _blocked(
        self,
        action: str,
        doctor_result: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "ok": False,
            "succeeded": False,
            "environment": self._environment_summary(doctor_result["manifest"]),
            "decision": "blocked",
            "message": message,
            "next_steps": ["Run doctor and bringup until navigation is ready."],
        }

    @staticmethod
    def _environment_summary(manifest: dict[str, Any]) -> dict[str, Any]:
        status = manifest.get("status", {})
        return {
            "mode": manifest.get("mode"),
            "robot": manifest.get("robot"),
            "simulator": manifest.get("simulator"),
            "environment_installed": status.get("environment_installed", False),
            "runtime_up": status.get("runtime_up", False),
            "tf_ready": status.get("tf_ready", False),
            "nav_ready": status.get("nav_ready", False),
        }
