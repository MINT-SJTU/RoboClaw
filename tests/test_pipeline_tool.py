from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.pipeline import PipelineTool
from roboclaw.bus.queue import MessageBus


def test_agent_loop_registers_pipeline_tool(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")

    assert loop.tools.get("pipeline") is not None


def test_pipeline_tool_lists_datasets(monkeypatch) -> None:
    from roboclaw.http.routes import curation as curation_routes

    monkeypatch.setattr(
        curation_routes,
        "list_curation_dataset_summaries",
        lambda: [{"id": "demo", "name": "demo"}],
    )

    result = json.loads(asyncio.run(PipelineTool().execute(action="list_datasets")))

    assert result["datasets"] == [{"id": "demo", "name": "demo"}]


def test_pipeline_tool_prepare_remote_dataset_emits_frontend_event(monkeypatch) -> None:
    from roboclaw.data import dataset_sessions

    bus = MessageBus()
    tool = PipelineTool(send_callback=bus.publish_outbound)
    tool.set_context(
        "web",
        "chat-1",
        metadata={"app_context": {"route": "/curation/datasets"}},
    )

    def fake_register_remote_dataset_session(dataset_id, *, include_videos=False, force=False):
        assert dataset_id == "imstevenpmwork/thanos_picking_power_gem"
        assert include_videos is True
        assert force is False
        return {
            "dataset_id": dataset_id,
            "dataset_name": "session:remote:thanos",
            "display_name": dataset_id,
            "local_path": "/tmp/thanos",
            "summary": {"name": "session:remote:thanos"},
        }

    monkeypatch.setattr(
        dataset_sessions,
        "register_remote_dataset_session",
        fake_register_remote_dataset_session,
    )

    result = json.loads(
        asyncio.run(
            tool.execute(
                action="prepare_remote_dataset",
                dataset="imstevenpmwork/thanos_picking_power_gem",
                include_videos=True,
            )
        )
    )
    message = asyncio.run(bus.consume_outbound())

    assert result["dataset_name"] == "session:remote:thanos"
    assert result["event_sent"] is True
    assert message.channel == "web"
    assert message.chat_id == "chat-1"
    app_event = message.metadata["app_event"]
    assert app_event["type"] == "pipeline.dataset_prepared"
    assert app_event["dataset_id"] == "imstevenpmwork/thanos_picking_power_gem"
    assert app_event["dataset_name"] == "session:remote:thanos"
    assert app_event["context"]["route"] == "/curation/datasets"


def test_pipeline_tool_merges_quality_threshold_defaults(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    captured: dict[str, object] = {}

    class FakeService:
        def get_quality_defaults(self, dataset_path, dataset):
            return {
                "selected_validators": ["metadata", "visual"],
                "threshold_overrides": {
                    "metadata_min_duration_s": 1.0,
                    "visual_min_resolution_width": 640.0,
                },
            }

        async def start_quality_run(
            self,
            dataset_path,
            dataset,
            selected_validators,
            episode_indices,
            threshold_overrides,
        ):
            captured["validators"] = selected_validators
            captured["thresholds"] = threshold_overrides
            return {"status": "started"}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    result = json.loads(
        asyncio.run(
            PipelineTool().execute(
                action="run_quality",
                dataset="demo",
                threshold_overrides={"metadata_min_duration_s": 0.2},
            )
        )
    )

    assert result["status"] == "started"
    assert captured["validators"] == ["metadata", "visual"]
    assert captured["thresholds"] == {
        "metadata_min_duration_s": 0.2,
        "visual_min_resolution_width": 640.0,
    }


def test_pipeline_tool_run_quality_emits_frontend_event(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_quality_defaults(self, dataset_path, dataset):
            return {
                "selected_validators": ["metadata", "visual"],
                "threshold_overrides": {"metadata_min_duration_s": 1.0},
            }

        async def start_quality_run(
            self,
            dataset_path,
            dataset,
            selected_validators,
            episode_indices,
            threshold_overrides,
        ):
            return {"status": "started"}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    bus = MessageBus()
    tool = PipelineTool(send_callback=bus.publish_outbound)
    tool.set_context(
        "web",
        "chat-1",
        metadata={"app_context": {"route": "/curation/quality"}},
    )

    result = json.loads(
        asyncio.run(
            tool.execute(
                action="run_quality",
                dataset="session:remote:thanos",
                episode_indices=[0, 1, 2],
            )
        )
    )
    message = asyncio.run(bus.consume_outbound())

    assert result["status"] == "started"
    assert result["event_sent"] is True
    app_event = message.metadata["app_event"]
    assert app_event["type"] == "pipeline.quality_run_started"
    assert app_event["dataset"] == "session:remote:thanos"
    assert app_event["episode_indices"] == [0, 1, 2]
    assert app_event["selected_validators"] == ["metadata", "visual"]
    assert app_event["context"]["route"] == "/curation/quality"
