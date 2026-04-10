"""Standalone navigation tool group for the isolated simulation slice."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.embodied.navigation.service import NavigationService
from roboclaw.embodied.standalone_tool import StandaloneTool


_NAVIGATION_ACTIONS = [
    "nav_status",
    "smoke_test",
    "navigate_to_pose",
    "follow_waypoints",
    "cancel_nav",
    "collect_metrics",
]


class NavigationToolGroup(StandaloneTool):
    """Expose isolated navigation actions without touching main embodied tools."""

    def __init__(self, service: NavigationService | None = None):
        self._service = service or NavigationService()

    @property
    def name(self) -> str:
        return "embodied_navigation"

    @property
    def description(self) -> str:
        return "Navigation-only status, smoke testing, and Nav2 task execution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _NAVIGATION_ACTIONS,
                    "description": "The navigation action to perform.",
                },
                "profile_id": {
                    "type": "string",
                    "description": "Capability profile id. Defaults to turtlebot3_gazebo_nav2.",
                },
                "x": {
                    "type": "number",
                    "description": "Goal x position in map frame.",
                },
                "y": {
                    "type": "number",
                    "description": "Goal y position in map frame.",
                },
                "yaw": {
                    "type": "number",
                    "description": "Goal yaw in radians.",
                },
                "frame_id": {
                    "type": "string",
                    "description": "Target frame id. Defaults to map.",
                },
                "behavior_tree": {
                    "type": "string",
                    "description": "Optional Nav2 behavior tree path for navigate_to_pose.",
                },
                "feedback": {
                    "type": "boolean",
                    "description": "Whether to stream ROS action feedback through CLI.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Optional timeout in seconds for navigation commands.",
                },
                "waypoints": {
                    "type": "array",
                    "description": "Waypoint list for follow_waypoints.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "yaw": {"type": "number"},
                            "frame_id": {"type": "string"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str | list:
        action = kwargs.get("action", "")
        if action not in _NAVIGATION_ACTIONS:
            return f"Unknown action '{action}' for tool {self.name}."

        if action == "nav_status":
            result = self._service.nav_status(profile_id=kwargs.get("profile_id"))
        elif action == "smoke_test":
            result = self._service.smoke_test(
                profile_id=kwargs.get("profile_id"),
                goal_x=kwargs.get("x"),
                goal_y=kwargs.get("y"),
                goal_yaw=float(kwargs.get("yaw", 0.0)),
                frame_id=kwargs.get("frame_id", "map"),
                feedback=kwargs.get("feedback", True),
                timeout_s=kwargs.get("timeout_s"),
            )
        elif action == "navigate_to_pose":
            if "x" not in kwargs or "y" not in kwargs:
                return "navigate_to_pose requires x and y."
            result = self._service.navigate_to_pose(
                profile_id=kwargs.get("profile_id"),
                x=float(kwargs["x"]),
                y=float(kwargs["y"]),
                yaw=float(kwargs.get("yaw", 0.0)),
                frame_id=kwargs.get("frame_id", "map"),
                behavior_tree=kwargs.get("behavior_tree", ""),
                feedback=kwargs.get("feedback", True),
                timeout_s=kwargs.get("timeout_s"),
            )
        elif action == "follow_waypoints":
            if not kwargs.get("waypoints"):
                return "follow_waypoints requires a non-empty waypoints array."
            result = self._service.follow_waypoints(
                profile_id=kwargs.get("profile_id"),
                waypoints=kwargs.get("waypoints", []),
                frame_id=kwargs.get("frame_id", "map"),
                feedback=kwargs.get("feedback", True),
                timeout_s=kwargs.get("timeout_s"),
            )
        elif action == "cancel_nav":
            result = self._service.cancel_nav(timeout_s=float(kwargs.get("timeout_s", 10.0)))
        else:
            result = self._service.collect_metrics()

        return json.dumps(result, indent=2, ensure_ascii=False)


def create_navigation_tools(service: NavigationService | None = None) -> list[NavigationToolGroup]:
    """Return the isolated navigation tool group list."""
    return [NavigationToolGroup(service=service)]
