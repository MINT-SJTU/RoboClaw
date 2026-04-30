from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.app import AppTool
from roboclaw.bus.queue import MessageBus


def test_agent_loop_registers_app_tool(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")

    assert loop.tools.get("app") is not None


def test_app_tool_lists_pages() -> None:
    result = json.loads(asyncio.run(AppTool().execute(action="list_pages")))

    page_ids = {page["id"] for page in result["pages"]}
    assert "control" in page_ids
    assert "curation_quality" in page_ids
    assert "training" in page_ids


def test_app_tool_uses_current_web_context() -> None:
    tool = AppTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/quality",
                "selected_dataset": "demo-dataset",
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_current_context")))

    assert result["context_available"] is True
    assert result["context"]["selected_dataset"] == "demo-dataset"
    assert result["page"]["id"] == "curation_quality"


def test_app_tool_navigation_emits_web_event() -> None:
    bus = MessageBus()
    tool = AppTool(send_callback=bus.publish_outbound)
    tool.set_context("web", "chat-1")

    result = json.loads(asyncio.run(tool.execute(action="navigate", page="training")))
    message = asyncio.run(bus.consume_outbound())

    assert result["status"] == "navigation_requested"
    assert result["event_sent"] is True
    assert message.channel == "web"
    assert message.chat_id == "chat-1"
    assert message.metadata["app_event"]["type"] == "app.navigate"
    assert message.metadata["app_event"]["route"] == "/training"


def test_app_tool_data_overview_exposes_episode_workspace_action() -> None:
    result = json.loads(
        asyncio.run(AppTool().execute(action="list_page_actions", page="curation_data_overview"))
    )

    action_ids = {item["id"] for item in result["actions"]}
    assert "pipeline.get_current_page_data" in action_ids
    assert "pipeline.get_data_overview" in action_ids
    assert "pipeline.get_alignment_overview" in action_ids
    assert "pipeline.get_episode_workspace" in action_ids


def test_app_tool_dataset_explorer_exposes_read_actions() -> None:
    result = json.loads(
        asyncio.run(AppTool().execute(action="list_page_actions", page="curation_datasets"))
    )

    action_ids = {item["id"] for item in result["actions"]}
    assert "pipeline.get_current_page_data" in action_ids
    assert "pipeline.get_explorer_summary" in action_ids
    assert "pipeline.get_explorer_details" in action_ids
    assert "pipeline.get_explorer_episodes" in action_ids


def test_app_tool_text_alignment_exposes_read_actions() -> None:
    result = json.loads(
        asyncio.run(AppTool().execute(action="list_page_actions", page="curation_text_alignment"))
    )

    action_ids = {item["id"] for item in result["actions"]}
    assert "pipeline.get_current_page_data" in action_ids
    assert "pipeline.get_alignment_overview" in action_ids
    assert "pipeline.get_prototype_results" in action_ids
    assert "pipeline.get_propagation_results" in action_ids
    assert "pipeline.get_episode_workspace" in action_ids
