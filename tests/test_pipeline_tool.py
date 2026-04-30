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


def test_pipeline_tool_get_alignment_overview(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_alignment_overview(self, dataset_path):
            assert dataset_path == tmp_path
            return {"summary": {"total_checked": 2}, "rows": [{"episode_index": 0}]}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    result = json.loads(
        asyncio.run(PipelineTool().execute(action="get_alignment_overview", dataset="demo"))
    )

    assert result["summary"]["total_checked"] == 2
    assert result["rows"][0]["episode_index"] == 0


def test_pipeline_tool_get_data_overview_bundle(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_workflow_state(self, dataset_path):
            assert dataset_path == tmp_path
            return {"stages": {"annotation": {"status": "completed"}}}

        def get_quality_results(self, dataset_path):
            assert dataset_path == tmp_path
            return {"total": 271, "passed": 120}

        def get_alignment_overview(self, dataset_path):
            assert dataset_path == tmp_path
            return {"summary": {"aligned_count": 271}}

        def get_prototype_results(self, dataset_path):
            assert dataset_path == tmp_path
            return {"cluster_count": 2}

        def get_propagation_results(self, dataset_path):
            assert dataset_path == tmp_path
            return {"target_count": 269}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    result = json.loads(
        asyncio.run(PipelineTool().execute(action="get_data_overview", dataset="demo"))
    )

    assert result["state"]["stages"]["annotation"]["status"] == "completed"
    assert result["quality_results"]["passed"] == 120
    assert result["alignment_overview"]["summary"]["aligned_count"] == 271
    assert result["prototype_results"]["cluster_count"] == 2
    assert result["propagation_results"]["target_count"] == 269
    assert "episodes" not in result["quality_results"]
    assert "rows" not in result["alignment_overview"]


def test_pipeline_tool_defaults_data_overview_dataset_from_web_context(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    captured: dict[str, str] = {}

    class FakeService:
        def get_workflow_state(self, dataset_path):
            return {"stages": {"annotation": {"status": "completed"}}}

        def get_quality_results(self, dataset_path):
            return {"total": 271}

        def get_alignment_overview(self, dataset_path):
            return {"summary": {"aligned_count": 271}}

        def get_prototype_results(self, dataset_path):
            return {"cluster_count": 2}

        def get_propagation_results(self, dataset_path):
            return {"target_count": 269}

    def fake_resolve_dataset_path(dataset):
        captured["dataset"] = dataset
        return tmp_path

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", fake_resolve_dataset_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    tool = PipelineTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/data-overview",
                "selected_dataset": "session:remote:demo",
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_data_overview")))

    assert captured["dataset"] == "session:remote:demo"
    assert result["alignment_overview"]["summary"]["aligned_count"] == 271


def test_pipeline_tool_get_current_page_data_for_data_overview(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_workflow_state(self, dataset_path):
            return {"stages": {"semantic_propagation": {"status": "completed"}}}

        def get_quality_results(self, dataset_path):
            return {"total": 271, "passed": 120, "failed": 151}

        def get_alignment_overview(self, dataset_path):
            return {"summary": {"aligned_count": 271, "propagated_count": 269}}

        def get_prototype_results(self, dataset_path):
            return {"cluster_count": 2}

        def get_propagation_results(self, dataset_path):
            return {"target_count": 269}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    tool = PipelineTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/data-overview",
                "selected_dataset": "session:remote:demo",
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_current_page_data")))

    assert result["page"] == "curation_data_overview"
    assert result["dataset"] == "session:remote:demo"
    assert result["quality_results"]["failed"] == 151
    assert result["alignment_overview"]["summary"]["propagated_count"] == 269
    assert "rows" not in result["alignment_overview"]
    assert "episodes" not in result["quality_results"]


def test_pipeline_tool_get_current_page_data_for_text_alignment(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_workflow_state(self, dataset_path):
            return {"stages": {"prototype_discovery": {"status": "completed"}}}

        def get_alignment_overview(self, dataset_path):
            return {"summary": {"aligned_count": 271}}

        def get_prototype_results(self, dataset_path):
            return {"cluster_count": 2}

        def get_propagation_results(self, dataset_path):
            return {"target_count": 269}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    tool = PipelineTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/text-alignment",
                "selected_dataset": "session:remote:demo",
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_current_page_data")))

    assert result["page"] == "curation_text_alignment"
    assert result["prototype_results"]["cluster_count"] == 2
    assert result["propagation_results"]["target_count"] == 269
    assert "propagated" not in result["propagation_results"]


def test_pipeline_tool_get_explorer_summary(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import explorer as explorer_routes

    monkeypatch.setattr(
        explorer_routes,
        "_resolve_dataset_context",
        lambda **_kwargs: ("remote", "demo/remote", None),
    )
    monkeypatch.setattr(
        explorer_routes,
        "build_remote_explorer_summary",
        lambda dataset: {"dataset": dataset, "summary": {"total_episodes": 12}},
    )

    result = json.loads(
        asyncio.run(
            PipelineTool().execute(
                action="get_explorer_summary",
                dataset="demo/remote",
                source="remote",
            )
        )
    )

    assert result["source"] == "remote"
    assert result["dataset"] == "demo/remote"
    assert result["payload"]["summary"]["total_episodes"] == 12


def test_pipeline_tool_defaults_explorer_summary_from_web_context(monkeypatch) -> None:
    from roboclaw.http.routes import explorer as explorer_routes

    captured: dict[str, object] = {}

    def fake_resolve_dataset_context(**kwargs):
        captured.update(kwargs)
        return "remote", kwargs["dataset"], None

    monkeypatch.setattr(explorer_routes, "_resolve_dataset_context", fake_resolve_dataset_context)
    monkeypatch.setattr(
        explorer_routes,
        "build_remote_explorer_summary",
        lambda dataset: {"dataset": dataset, "summary": {"total_episodes": 271}},
    )

    tool = PipelineTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/datasets",
                "explorer": {
                    "active_dataset_ref": {
                        "source": "remote",
                        "dataset": "Elvinky/bi-so101-insert-screw-271ep",
                    }
                },
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_explorer_summary")))

    assert captured["source"] == "remote"
    assert captured["dataset"] == "Elvinky/bi-so101-insert-screw-271ep"
    assert result["payload"]["summary"]["total_episodes"] == 271


def test_pipeline_tool_get_current_page_data_for_dataset_explorer(monkeypatch) -> None:
    from roboclaw.http.routes import explorer as explorer_routes

    monkeypatch.setattr(
        explorer_routes,
        "_resolve_dataset_context",
        lambda **kwargs: ("remote", kwargs["dataset"], None),
    )
    monkeypatch.setattr(
        explorer_routes,
        "build_remote_explorer_summary",
        lambda dataset: {"dataset": dataset, "summary": {"total_episodes": 271}},
    )
    monkeypatch.setattr(
        explorer_routes,
        "build_remote_episode_page",
        lambda dataset, page, page_size: {
            "dataset": dataset,
            "page": page,
            "page_size": page_size,
            "episodes": [{"episode_index": 0}],
        },
    )

    tool = PipelineTool()
    tool.set_context(
        "web",
        "chat-1",
        metadata={
            "app_context": {
                "route": "/curation/datasets",
                "explorer": {
                    "active_dataset_ref": {
                        "source": "remote",
                        "dataset": "Elvinky/bi-so101-insert-screw-271ep",
                    }
                },
            }
        },
    )

    result = json.loads(asyncio.run(tool.execute(action="get_current_page_data")))

    assert result["page"] == "curation_datasets"
    assert result["explorer"]["dataset"] == "Elvinky/bi-so101-insert-screw-271ep"
    assert result["explorer"]["summary"]["summary"]["total_episodes"] == 271
    assert result["explorer"]["episodes"]["episodes"][0]["episode_index"] == 0


def test_pipeline_tool_get_explorer_episodes_for_path_source(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import explorer as explorer_routes

    monkeypatch.setattr(
        explorer_routes,
        "_resolve_dataset_context",
        lambda **_kwargs: ("path", "demo-local", tmp_path),
    )
    monkeypatch.setattr(
        explorer_routes,
        "_build_local_episode_page",
        lambda dataset_path, dataset, page, page_size: {
            "dataset_path": str(dataset_path),
            "dataset": dataset,
            "page": page,
            "page_size": page_size,
            "episodes": [{"episode_index": 0}],
        },
    )

    result = json.loads(
        asyncio.run(
            PipelineTool().execute(
                action="get_explorer_episodes",
                source="path",
                path=str(tmp_path),
                page=2,
                page_size=25,
            )
        )
    )

    assert result["source"] == "path"
    assert result["payload"]["dataset"] == "demo-local"
    assert result["payload"]["page"] == 2
    assert result["payload"]["page_size"] == 25


def test_pipeline_tool_run_prototype_defaults_to_all_candidates(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    captured: dict[str, object] = {}

    class FakeService:
        async def start_prototype_run(
            self,
            dataset_path,
            dataset,
            cluster_count,
            candidate_limit,
            episode_indices,
            quality_filter_mode,
        ):
            captured["candidate_limit"] = candidate_limit
            captured["quality_filter_mode"] = quality_filter_mode
            return {"status": "started"}

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    result = json.loads(
        asyncio.run(
            PipelineTool().execute(
                action="run_prototype",
                dataset="demo",
                quality_filter_mode="raw",
            )
        )
    )

    assert result["status"] == "started"
    assert captured["candidate_limit"] is None
    assert captured["quality_filter_mode"] == "raw"


def test_pipeline_tool_get_episode_workspace(monkeypatch, tmp_path) -> None:
    from roboclaw.http.routes import curation as curation_routes

    class FakeService:
        def get_workspace_payload(self, dataset, dataset_path, episode_index):
            assert dataset == "demo"
            assert dataset_path == tmp_path
            assert episode_index == 7
            return {
                "episode_index": 7,
                "videos": [{"url": "/api/curation/video/demo.mp4", "stream": "front"}],
                "joint_trajectory": {
                    "time_values": [0.0, 0.1],
                    "frame_values": [0, 3],
                    "joint_trajectories": [
                        {
                            "joint_name": "shoulder",
                            "action_values": [0.1, 0.2],
                            "state_values": [0.0, 0.15],
                        }
                    ],
                },
            }

    monkeypatch.setattr(curation_routes, "resolve_dataset_path", lambda _dataset: tmp_path)
    monkeypatch.setattr(curation_routes, "_service", FakeService())

    result = json.loads(
        asyncio.run(
            PipelineTool().execute(
                action="get_episode_workspace",
                dataset="demo",
                episode_index=7,
            )
        )
    )

    assert result["episode_index"] == 7
    assert result["videos"][0]["stream"] == "front"
    assert result["joint_trajectory"]["joint_trajectories"][0]["joint_name"] == "shoulder"
