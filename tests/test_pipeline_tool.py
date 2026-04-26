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
