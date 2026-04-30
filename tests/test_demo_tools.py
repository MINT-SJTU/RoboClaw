"""Tests for demo-only tool aggregation."""

from __future__ import annotations

from roboclaw.embodied.demo_tools import create_demo_tools, register_demo_tools


class _FakeRegistry:
    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def register(self, tool: object) -> None:
        self.registered[getattr(tool, "name")] = tool


def test_create_demo_tools_returns_simulation_and_navigation_groups() -> None:
    tools = create_demo_tools()

    assert [tool.name for tool in tools] == ["embodied_simulation", "embodied_navigation"]


def test_register_demo_tools_registers_expected_names() -> None:
    registry = _FakeRegistry()

    names = register_demo_tools(registry)

    assert names == ["embodied_simulation", "embodied_navigation"]
    assert set(registry.registered) == {"embodied_simulation", "embodied_navigation"}
