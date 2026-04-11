"""Demo-only tool aggregation for the isolated simulation/navigation slice."""

from __future__ import annotations

from typing import Any

from roboclaw.embodied.navigation.tool import create_navigation_tools
from roboclaw.embodied.simulation.tool import create_simulation_tools


def create_demo_tools() -> list[Any]:
    """Return the isolated demo tool groups without touching arm embodied tools."""
    return [*create_simulation_tools(), *create_navigation_tools()]


def register_demo_tools(registry: Any) -> list[str]:
    """Register demo tools into a ToolRegistry-like object and return their names."""
    names: list[str] = []
    for tool in create_demo_tools():
        registry.register(tool)
        names.append(tool.name)
    return names
