"""Standalone navigation service for the isolated simulation slice."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from roboclaw.embodied.navigation.evaluator import NavigationEvaluator
from roboclaw.embodied.navigation.nav2_client import Nav2Client
from roboclaw.embodied.navigation.smoke_test import SmokeTestRunner
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
    ) -> None:
        self._simulation = simulation_service or SimulationService()
        self._nav_client = nav_client or Nav2Client()
        self._smoke_test = smoke_test_runner or SmokeTestRunner()
        self._evaluator = evaluator or NavigationEvaluator()
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
