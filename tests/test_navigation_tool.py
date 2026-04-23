"""Tests for the isolated navigation tool group."""

from __future__ import annotations

import asyncio
import json

from roboclaw.embodied.navigation.tool import NavigationToolGroup, create_navigation_tools


class _FakeNavigationService:
    def nav_status(self, *, profile_id=None):
        return {"action": "nav_status", "ok": True, "profile_id": profile_id}

    def smoke_test(self, **kwargs):
        return {"action": "smoke_test", "ok": True, "kwargs": kwargs}

    def navigate_to_pose(self, **kwargs):
        return {"action": "navigate_to_pose", "ok": True, "kwargs": kwargs}

    def resolve_place(self, **kwargs):
        return {"action": "resolve_place", "ok": True, "kwargs": kwargs}

    def navigate_to_place(self, **kwargs):
        return {"action": "navigate_to_place", "ok": True, "kwargs": kwargs}

    def follow_waypoints(self, **kwargs):
        return {"action": "follow_waypoints", "ok": True, "kwargs": kwargs}

    def cancel_nav(self, *, timeout_s):
        return {"action": "cancel_nav", "ok": True, "timeout_s": timeout_s}

    def collect_metrics(self):
        return {"action": "collect_metrics", "ok": True}


def test_create_navigation_tools_returns_one_group() -> None:
    tools = create_navigation_tools(service=_FakeNavigationService())

    assert len(tools) == 1
    assert tools[0].name == "embodied_navigation"


def test_navigation_tool_dispatches_navigate_to_pose() -> None:
    tool = NavigationToolGroup(service=_FakeNavigationService())

    raw = asyncio.run(tool.execute(action="navigate_to_pose", x=1.25, y=-0.5, yaw=0.3, feedback=False))
    data = json.loads(raw)

    assert data["action"] == "navigate_to_pose"
    assert data["kwargs"]["x"] == 1.25
    assert data["kwargs"]["feedback"] is False


def test_navigation_tool_dispatches_navigate_to_place() -> None:
    tool = NavigationToolGroup(service=_FakeNavigationService())

    raw = asyncio.run(tool.execute(action="navigate_to_place", place="bedroom", map_id="house", feedback=False))
    data = json.loads(raw)

    assert data["action"] == "navigate_to_place"
    assert data["kwargs"]["place"] == "bedroom"
    assert data["kwargs"]["map_id"] == "house"
    assert data["kwargs"]["feedback"] is False


def test_navigation_tool_dispatches_collect_metrics() -> None:
    tool = NavigationToolGroup(service=_FakeNavigationService())

    raw = asyncio.run(tool.execute(action="collect_metrics"))
    data = json.loads(raw)

    assert data["action"] == "collect_metrics"
    assert data["ok"] is True
