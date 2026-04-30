"""Standalone simulation tool group for the Phase 4 vertical slice."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.embodied.standalone_tool import StandaloneTool
from roboclaw.embodied.simulation.service import SimulationService


_SIMULATION_ACTIONS = ["state_show", "doctor", "bringup", "shutdown", "reset_world"]


class SimulationToolGroup(StandaloneTool):
    """Expose isolated simulation actions without touching main embodied tools."""

    def __init__(self, service: SimulationService | None = None):
        self._service = service or SimulationService()

    @property
    def name(self) -> str:
        return "embodied_simulation"

    @property
    def description(self) -> str:
        return "Simulation-only bringup, doctor, and runtime control for navigation workflows."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _SIMULATION_ACTIONS,
                    "description": "The simulation action to perform.",
                },
                "profile_id": {
                    "type": "string",
                    "description": "Capability profile id. Defaults to turtlebot3_gazebo_nav2.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["gazebo", "nav"],
                    "description": "Bringup mode. nav means full navigation bringup.",
                },
                "map_path": {
                    "type": "string",
                    "description": "Map path for navigation bringup.",
                },
                "map_id": {
                    "type": "string",
                    "description": (
                        "Named map/world id for navigation bringup. Use house for semantic/house navigation demos."
                    ),
                },
                "world_launch": {
                    "type": "string",
                    "description": "Gazebo launch file for turtlebot3_gazebo.",
                },
                "model": {
                    "type": "string",
                    "description": "Robot model, for example burger or waffle.",
                },
                "ros_domain_id": {
                    "type": "integer",
                    "description": "ROS_DOMAIN_ID override for bringup.",
                },
                "rviz": {
                    "type": "boolean",
                    "description": "Whether navigation bringup should include RViz.",
                },
                "service_name": {
                    "type": "string",
                    "description": "ROS service name for reset_world. Defaults to /reset_simulation.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Timeout in seconds for reset_world or shutdown helpers.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str | list:
        action = kwargs.get("action", "")
        if action not in _SIMULATION_ACTIONS:
            return f"Unknown action '{action}' for tool {self.name}."

        if action == "state_show":
            result = self._service.state_show()
        elif action == "doctor":
            result = self._service.doctor(profile_id=kwargs.get("profile_id"))
        elif action == "bringup":
            result = self._service.bringup(
                profile_id=kwargs.get("profile_id"),
                mode=kwargs.get("mode", "nav"),
                map_id=kwargs.get("map_id"),
                map_path=kwargs.get("map_path"),
                world_launch=kwargs.get("world_launch"),
                model=kwargs.get("model"),
                ros_domain_id=kwargs.get("ros_domain_id"),
                rviz=kwargs.get("rviz", True),
            )
        elif action == "shutdown":
            result = self._service.shutdown()
        else:
            result = self._service.reset_world(
                service_name=kwargs.get("service_name", "/reset_simulation"),
                timeout_s=float(kwargs.get("timeout_s", 10.0)),
            )

        return json.dumps(result, indent=2, ensure_ascii=False)


def create_simulation_tools(service: SimulationService | None = None) -> list[SimulationToolGroup]:
    """Return the isolated simulation tool group list."""
    return [SimulationToolGroup(service=service)]
