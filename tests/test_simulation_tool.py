"""Tests for the isolated simulation tool group."""

from __future__ import annotations

import asyncio
import json

from roboclaw.embodied.simulation.tool import SimulationToolGroup, create_simulation_tools


class _FakeService:
    def state_show(self):
        return {"action": "state_show", "ok": True}

    def doctor(self, *, profile_id=None):
        return {"action": "doctor", "profile_id": profile_id, "ok": True}

    def bringup(self, **kwargs):
        return {"action": "bringup", "ok": True, "kwargs": kwargs}

    def shutdown(self):
        return {"action": "shutdown", "ok": True}

    def reset_world(self, *, service_name, timeout_s):
        return {"action": "reset_world", "service_name": service_name, "timeout_s": timeout_s, "ok": True}


def test_create_simulation_tools_returns_one_group() -> None:
    tools = create_simulation_tools(service=_FakeService())

    assert len(tools) == 1
    assert tools[0].name == "embodied_simulation"


def test_simulation_tool_dispatches_bringup() -> None:
    tool = SimulationToolGroup(service=_FakeService())

    raw = asyncio.run(tool.execute(action="bringup", mode="nav", rviz=False))
    data = json.loads(raw)

    assert data["action"] == "bringup"
    assert data["kwargs"]["mode"] == "nav"
    assert data["kwargs"]["rviz"] is False


def test_simulation_tool_dispatches_reset_world() -> None:
    tool = SimulationToolGroup(service=_FakeService())

    raw = asyncio.run(tool.execute(action="reset_world", service_name="/reset_world", timeout_s=3))
    data = json.loads(raw)

    assert data["action"] == "reset_world"
    assert data["service_name"] == "/reset_world"
    assert data["timeout_s"] == 3.0
