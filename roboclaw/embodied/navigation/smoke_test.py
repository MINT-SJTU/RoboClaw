"""Smoke test evaluation for the isolated navigation slice."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class SmokeTestRunner:
    """Combine doctor readiness checks and optional short-goal execution."""

    def run(
        self,
        *,
        doctor_result: dict[str, Any],
        active_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = doctor_result["manifest"]
        checks = self._smoke_checks(manifest)
        environment = {
            "mode": manifest.get("mode"),
            "robot": manifest.get("robot"),
            "simulator": manifest.get("simulator"),
            "nav_ready": manifest.get("status", {}).get("nav_ready", False),
            "tf_ready": manifest.get("status", {}).get("tf_ready", False),
        }

        if not manifest.get("status", {}).get("nav_ready", False):
            return {
                "action": "smoke_test",
                "ok": False,
                "passed": False,
                "mode": "passive",
                "environment": environment,
                "checks": checks,
                "decision": "blocked",
                "next_steps": ["Run doctor and bringup until navigation is ready."],
                "active_result": active_result,
            }

        if active_result is None:
            return {
                "action": "smoke_test",
                "ok": True,
                "passed": True,
                "mode": "passive",
                "environment": environment,
                "checks": checks,
                "decision": "ready_for_navigation_task",
                "next_steps": ["Send a goal pose or waypoint task."],
                "active_result": None,
            }

        passed = bool(active_result.get("succeeded"))
        return {
            "action": "smoke_test",
            "ok": bool(active_result.get("ok")),
            "passed": passed,
            "mode": "active",
            "environment": environment,
            "checks": checks,
            "decision": "smoke_test_passed" if passed else "smoke_test_failed",
            "next_steps": ["Proceed to the user navigation task."] if passed else [
                "Inspect the failed active smoke test result.",
                "Fix localization or planner issues before retrying.",
            ],
            "active_result": deepcopy(active_result),
        }

    @staticmethod
    def _smoke_checks(manifest: dict[str, Any]) -> dict[str, bool]:
        topics = manifest.get("checks", {}).get("topics", {})
        actions = manifest.get("checks", {}).get("actions", {})
        return {
            "cmd_vel": bool(topics.get("/cmd_vel", False)),
            "odom": bool(topics.get("/odom", False)),
            "scan": bool(topics.get("/scan", False)),
            "navigate_to_pose": bool(actions.get("/navigate_to_pose", False)),
        }
