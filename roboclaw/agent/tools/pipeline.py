"""Pipeline tool for the in-app RoboClaw AI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.bus.events import OutboundMessage

SendCallback = Callable[[OutboundMessage], Awaitable[None]]


class PipelineTool(Tool):
    """Let RoboClaw AI inspect and trigger curation Pipeline stages."""

    def __init__(self, send_callback: SendCallback | None = None):
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""
        self._context_by_session: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "pipeline"

    @property
    def description(self) -> str:
        return (
            "Control RoboClaw's built-in curation Pipeline: list datasets, inspect workflow state, "
            "prepare remote datasets, get dataset-aware quality defaults, and start/pause/resume "
            "quality/prototype/propagation runs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_datasets",
                        "prepare_remote_dataset",
                        "load_remote_dataset",
                        "get_state",
                        "get_quality_defaults",
                        "get_quality_results",
                        "run_quality",
                        "pause_quality",
                        "resume_quality",
                        "run_prototype",
                        "run_propagation",
                    ],
                    "description": "Pipeline operation to perform.",
                },
                "dataset": {
                    "type": "string",
                    "description": "Dataset id/name/session handle. Required except for list_datasets.",
                },
                "include_videos": {
                    "type": "boolean",
                    "description": "Whether to include videos when preparing a remote dataset. Defaults to false.",
                },
                "force": {
                    "type": "boolean",
                    "description": "Whether to rebuild an existing prepared remote dataset session.",
                },
                "selected_validators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quality validators to run. Defaults to dataset-aware quality defaults.",
                },
                "threshold_overrides": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Quality threshold overrides.",
                },
                "episode_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional episode indices to run.",
                },
                "cluster_count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cluster count for prototype discovery.",
                },
                "candidate_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Candidate limit for prototype discovery.",
                },
                "quality_filter_mode": {
                    "type": "string",
                    "enum": ["passed", "failed", "all"],
                    "description": "Which quality rows prototype discovery should use.",
                },
                "source_episode_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Source episode for semantic propagation.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        action: str,
        dataset: str = "",
        selected_validators: list[str] | None = None,
        threshold_overrides: dict[str, float] | None = None,
        episode_indices: list[int] | None = None,
        include_videos: bool = False,
        force: bool = False,
        cluster_count: int | None = None,
        candidate_limit: int = 50,
        quality_filter_mode: str = "passed",
        source_episode_index: int | None = None,
    ) -> str:
        from roboclaw.data.curation.state import set_stage_pause_requested
        from roboclaw.http.routes import curation as curation_routes

        try:
            if action == "list_datasets":
                return _json({
                    "datasets": curation_routes.list_curation_dataset_summaries(),
                })

            if not dataset:
                return _json({"error": "dataset is required"})

            if action in {"prepare_remote_dataset", "load_remote_dataset"}:
                from roboclaw.data.dataset_sessions import register_remote_dataset_session

                payload = await asyncio.to_thread(
                    register_remote_dataset_session,
                    dataset,
                    include_videos=include_videos,
                    force=force,
                )
                event_sent = await self._send_app_event({
                    "type": "pipeline.dataset_prepared",
                    "dataset_id": dataset,
                    "dataset_name": payload.get("dataset_name"),
                    "display_name": payload.get("display_name"),
                    "source_dataset": payload.get("dataset_id") or dataset,
                    "local_path": payload.get("local_path"),
                    "summary": payload.get("summary"),
                    "include_videos": include_videos,
                })
                return _json({**payload, "event_sent": event_sent})

            dataset_path = curation_routes.resolve_dataset_path(dataset)
            service = curation_routes._service

            if action == "get_state":
                return _json(service.get_workflow_state(dataset_path))

            if action == "get_quality_defaults":
                return _json(service.get_quality_defaults(dataset_path, dataset))

            if action == "get_quality_results":
                return _json(service.get_quality_results(dataset_path))

            if action == "run_quality":
                defaults = service.get_quality_defaults(dataset_path, dataset)
                validators = selected_validators or defaults["selected_validators"]
                thresholds = {
                    **defaults["threshold_overrides"],
                    **(threshold_overrides or {}),
                }
                result = await service.start_quality_run(
                    dataset_path,
                    dataset,
                    validators,
                    episode_indices,
                    thresholds,
                )
                event_sent = await self._send_app_event({
                    "type": "pipeline.quality_run_started",
                    "dataset": dataset,
                    "status": result.get("status"),
                    "selected_validators": validators,
                    "episode_indices": episode_indices or [],
                })
                return _json({**result, "event_sent": event_sent})

            if action == "pause_quality":
                state = service.get_workflow_state(dataset_path)
                if state["stages"]["quality_validation"].get("status") != "running":
                    return _json({"error": "Quality validation is not running"})
                set_stage_pause_requested(dataset_path, "quality_validation", True)
                event_sent = await self._send_app_event({
                    "type": "pipeline.quality_state_changed",
                    "dataset": dataset,
                    "status": "pause_requested",
                })
                return _json({"status": "pause_requested", "event_sent": event_sent})

            if action == "resume_quality":
                defaults = service.get_quality_defaults(dataset_path, dataset)
                thresholds = {
                    **defaults["threshold_overrides"],
                    **(threshold_overrides or {}),
                }
                result = await service.start_quality_resume(
                    dataset_path,
                    dataset,
                    selected_validators or defaults["selected_validators"],
                    episode_indices,
                    thresholds,
                )
                event_sent = await self._send_app_event({
                    "type": "pipeline.quality_run_started",
                    "dataset": dataset,
                    "status": result.get("status"),
                    "selected_validators": selected_validators or defaults["selected_validators"],
                    "episode_indices": episode_indices or [],
                    "resumed": True,
                })
                return _json({**result, "event_sent": event_sent})

            if action == "run_prototype":
                result = await service.start_prototype_run(
                    dataset_path,
                    dataset,
                    cluster_count,
                    candidate_limit,
                    episode_indices,
                    quality_filter_mode,
                )
                return _json(result)

            if action == "run_propagation":
                if source_episode_index is None:
                    return _json({"error": "source_episode_index is required"})
                result = await service.start_propagation_run(
                    dataset_path,
                    dataset,
                    source_episode_index,
                )
                return _json(result)

            return _json({"error": f"Unknown pipeline action: {action}"})
        except Exception as exc:
            return _json({"error": str(exc)})

    def set_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        app_context = _extract_app_context(metadata or {})
        if app_context:
            self._context_by_session[self._session_key(channel, chat_id)] = app_context

    async def _send_app_event(self, app_event: dict[str, Any]) -> bool:
        if not self._send_callback or self._channel != "web" or not self._chat_id:
            return False
        context = deepcopy(self._context_by_session.get(self._session_key(self._channel, self._chat_id), {}))
        if context:
            app_event.setdefault("context", context)
        await self._send_callback(
            OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content="",
                metadata={"app_event": app_event},
            )
        )
        return True

    @staticmethod
    def _session_key(channel: str, chat_id: str) -> str:
        return f"{channel}:{chat_id}"


def _extract_app_context(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("app_context") or metadata.get("appContext") or metadata.get("app")
    if not isinstance(raw, dict):
        return {}
    return deepcopy(raw)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
