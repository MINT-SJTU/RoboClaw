"""Structured evaluation helpers for navigation actions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_OUTPUT_TAIL_CHARS = 800


class NavigationEvaluator:
    """Turn raw Nav2 command results into compact structured reports."""

    def action_report(
        self,
        *,
        action: str,
        command_result: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = self._metrics_from_result(command_result)
        command_ok = bool(command_result.get("ok"))
        goal_accepted = bool(command_result.get("goal_accepted"))
        goal_status = command_result.get("goal_status")
        succeeded = bool(command_result.get("goal_succeeded"))
        if succeeded:
            decision = "goal_reached"
            next_steps = ["Collect metrics or issue another navigation task."]
        elif command_ok and goal_accepted and goal_status is None:
            decision = "goal_result_inconclusive"
            next_steps = [
                "Inspect Nav2 CLI output and runtime logs for the missing terminal status.",
                "Confirm the robot pose or rerun the goal if the stack is healthy.",
            ]
        else:
            decision = "goal_failed"
            next_steps = [
                "Inspect localization and planner status.",
                "Try cancel_nav or reset_world if the stack is stuck.",
            ]
        return {
            "action": action,
            "ok": command_ok,
            "succeeded": succeeded,
            "environment": deepcopy(environment),
            "goal": deepcopy(command_result.get("goal")),
            "goal_status": goal_status,
            "metrics": metrics,
            "command": list(command_result.get("command", [])),
            "returncode": command_result.get("returncode"),
            "output": self._output_summary(command_result),
            "decision": decision,
            "next_steps": next_steps,
        }

    def collect_metrics(self, last_report: dict[str, Any] | None) -> dict[str, Any]:
        if not last_report:
            return {
                "action": "collect_metrics",
                "ok": False,
                "message": "No navigation metrics have been collected yet.",
                "metrics": {},
            }
        return {
            "action": "collect_metrics",
            "ok": True,
            "source_action": last_report.get("action"),
            "succeeded": last_report.get("succeeded"),
            "goal_status": last_report.get("goal_status"),
            "metrics": deepcopy(last_report.get("metrics", {})),
            "decision": last_report.get("decision"),
        }

    @staticmethod
    def _metrics_from_result(command_result: dict[str, Any]) -> dict[str, Any]:
        metrics = deepcopy(command_result.get("metrics", {}))
        metrics["goal_accepted"] = bool(command_result.get("goal_accepted"))
        goal_status = command_result.get("goal_status")
        if goal_status is not None:
            metrics["goal_status"] = goal_status
        metrics["command_ok"] = bool(command_result.get("ok"))
        return metrics

    @staticmethod
    def _output_summary(command_result: dict[str, Any]) -> dict[str, Any]:
        stdout = str(command_result.get("stdout", "") or "")
        stderr = str(command_result.get("stderr", "") or "")
        return {
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "truncated": len(stdout) > _OUTPUT_TAIL_CHARS or len(stderr) > _OUTPUT_TAIL_CHARS,
        }


def _tail(value: str) -> str:
    if len(value) <= _OUTPUT_TAIL_CHARS:
        return value
    return value[-_OUTPUT_TAIL_CHARS:]
