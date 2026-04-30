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
            "quality/prototype/propagation runs. Use get_current_page_data to read the user's "
            "current 数据集读取, 质量验证, 文本对齐, or 数据总览 page from live app context."
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
                        "get_current_page_data",
                        "get_state",
                        "get_quality_defaults",
                        "get_quality_results",
                        "get_prototype_results",
                        "get_propagation_results",
                        "get_alignment_overview",
                        "get_data_overview",
                        "get_explorer_summary",
                        "get_explorer_details",
                        "get_explorer_episodes",
                        "get_episode_workspace",
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
                    "description": (
                        "Dataset id/name/session handle. Defaults to the selected dataset from "
                        "the current web app context for curation read actions."
                    ),
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
                "episode_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Episode index for inspecting video, joint trajectory, annotations, and quality context.",
                },
                "cluster_count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cluster count for prototype discovery.",
                },
                "candidate_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional candidate limit for prototype discovery. Omit to use all selected episodes.",
                },
                "quality_filter_mode": {
                    "type": "string",
                    "enum": ["passed", "failed", "all", "raw"],
                    "description": "Which quality rows prototype discovery should use.",
                },
                "source_episode_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Source episode for semantic propagation.",
                },
                "source": {
                    "type": "string",
                    "enum": ["remote", "local", "path"],
                    "description": "Dataset explorer source for get_explorer_* actions.",
                },
                "path": {
                    "type": "string",
                    "description": "Local dataset path for dataset explorer path-source actions.",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Episode page number for get_explorer_episodes.",
                },
                "page_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Episode page size for get_explorer_episodes.",
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
        episode_index: int | None = None,
        cluster_count: int | None = None,
        candidate_limit: int | None = None,
        quality_filter_mode: str = "passed",
        source_episode_index: int | None = None,
        source: str = "remote",
        path: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> str:
        from roboclaw.data.curation.state import set_stage_pause_requested
        from roboclaw.http.routes import curation as curation_routes
        from roboclaw.http.routes import explorer as explorer_routes

        try:
            context = self._current_context()
            if not dataset:
                if action in {"prepare_remote_dataset", "load_remote_dataset"}:
                    source, dataset, path = self._default_explorer_args(context, source, "", path)
                elif action not in {
                    "list_datasets",
                    "get_explorer_summary",
                    "get_explorer_details",
                    "get_explorer_episodes",
                }:
                    dataset = self._default_dataset(context)
            dataset = dataset.strip()

            if action == "list_datasets":
                return _json({
                    "datasets": curation_routes.list_curation_dataset_summaries(),
                })

            if action == "get_current_page_data":
                return _json(
                    await self._get_current_page_data(
                        context,
                        curation_routes,
                        explorer_routes,
                        dataset,
                        page,
                        page_size,
                    )
                )

            if action in {"get_explorer_summary", "get_explorer_details", "get_explorer_episodes"}:
                source, dataset, path = self._default_explorer_args(context, source, dataset, path)
                return _json(
                    await self._read_explorer_action(
                        explorer_routes,
                        action,
                        source,
                        dataset,
                        path,
                        page,
                        page_size,
                    )
                )

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

            if action == "get_prototype_results":
                return _json(service.get_prototype_results(dataset_path))

            if action == "get_propagation_results":
                return _json(service.get_propagation_results(dataset_path))

            if action == "get_alignment_overview":
                return _json(service.get_alignment_overview(dataset_path))

            if action == "get_data_overview":
                quality_results = service.get_quality_results(dataset_path)
                alignment_overview = service.get_alignment_overview(dataset_path)
                return _json({
                    "state": service.get_workflow_state(dataset_path),
                    "quality_results": _summarize_quality_results(quality_results),
                    "alignment_overview": _summarize_alignment_overview(alignment_overview),
                    "prototype_results": _summarize_prototype_results(service.get_prototype_results(dataset_path)),
                    "propagation_results": _summarize_propagation_results(service.get_propagation_results(dataset_path)),
                })

            if action == "get_episode_workspace":
                if episode_index is None:
                    return _json({"error": "episode_index is required"})
                return _json(service.get_workspace_payload(dataset, dataset_path, episode_index))

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

    def _current_context(self) -> dict[str, Any]:
        return deepcopy(self._context_by_session.get(self._session_key(self._channel, self._chat_id), {}))

    def _default_dataset(self, context: dict[str, Any]) -> str:
        value = context.get("selected_dataset") or context.get("dataset")
        return str(value or "")

    def _default_explorer_args(
        self,
        context: dict[str, Any],
        source: str,
        dataset: str,
        path: str,
    ) -> tuple[str, str, str]:
        if dataset or path:
            return source, dataset, path
        explorer = context.get("explorer")
        if not isinstance(explorer, dict):
            return source, dataset, path

        active_ref = explorer.get("active_dataset_ref")
        if isinstance(active_ref, dict):
            ref_source = str(active_ref.get("source") or source or "remote")
            ref_dataset = str(active_ref.get("dataset") or "")
            ref_path = str(active_ref.get("path") or "")
            if ref_dataset or ref_path:
                return ref_source, ref_dataset, ref_path

        summary_dataset = str(explorer.get("summary_dataset") or "")
        if summary_dataset:
            return str(explorer.get("source") or source or "remote"), summary_dataset, path
        return source, dataset, path

    async def _get_current_page_data(
        self,
        context: dict[str, Any],
        curation_routes: Any,
        explorer_routes: Any,
        dataset: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        route = _normalize_route(str(context.get("route") or context.get("pathname") or ""))

        if route.startswith("/curation/datasets"):
            source, explorer_dataset, explorer_path = self._default_explorer_args(
                context,
                "remote",
                "",
                "",
            )
            if not explorer_dataset and not explorer_path:
                return {
                    "page": "curation_datasets",
                    "context": context,
                    "error": "No active dataset explorer selection is available",
                }
            summary = await self._read_explorer_action(
                explorer_routes,
                "get_explorer_summary",
                source,
                explorer_dataset,
                explorer_path,
                page,
                page_size,
            )
            episodes = await self._read_explorer_action(
                explorer_routes,
                "get_explorer_episodes",
                source,
                explorer_dataset,
                explorer_path,
                page,
                page_size,
            )
            return {
                "page": "curation_datasets",
                "context": context,
                "explorer": {
                    "source": summary["source"],
                    "dataset": summary["dataset"],
                    "path": summary["path"],
                    "summary": summary["payload"],
                    "episodes": episodes["payload"],
                },
            }

        if not dataset:
            return {
                "page": route or "unknown",
                "context": context,
                "error": "No selected curation dataset is available",
            }

        dataset_path = curation_routes.resolve_dataset_path(dataset)
        service = curation_routes._service

        if route.startswith("/curation/quality"):
            quality_results = service.get_quality_results(dataset_path)
            return {
                "page": "curation_quality",
                "context": context,
                "dataset": dataset,
                "state": service.get_workflow_state(dataset_path),
                "quality_defaults": service.get_quality_defaults(dataset_path, dataset),
                "quality_results": _summarize_quality_results(quality_results),
            }

        if route.startswith("/curation/text-alignment"):
            alignment_overview = service.get_alignment_overview(dataset_path)
            return {
                "page": "curation_text_alignment",
                "context": context,
                "dataset": dataset,
                "state": service.get_workflow_state(dataset_path),
                "alignment_overview": _summarize_alignment_overview(alignment_overview),
                "prototype_results": _summarize_prototype_results(service.get_prototype_results(dataset_path)),
                "propagation_results": _summarize_propagation_results(service.get_propagation_results(dataset_path)),
            }

        if route.startswith("/curation/data-overview"):
            quality_results = service.get_quality_results(dataset_path)
            alignment_overview = service.get_alignment_overview(dataset_path)
            return {
                "page": "curation_data_overview",
                "context": context,
                "dataset": dataset,
                "state": service.get_workflow_state(dataset_path),
                "quality_results": _summarize_quality_results(quality_results),
                "alignment_overview": _summarize_alignment_overview(alignment_overview),
                "prototype_results": _summarize_prototype_results(service.get_prototype_results(dataset_path)),
                "propagation_results": _summarize_propagation_results(service.get_propagation_results(dataset_path)),
            }

        alignment_overview = service.get_alignment_overview(dataset_path)
        return {
            "page": route or "curation_dataset",
            "context": context,
            "dataset": dataset,
            "state": service.get_workflow_state(dataset_path),
            "alignment_overview": _summarize_alignment_overview(alignment_overview),
        }

    async def _read_explorer_action(
        self,
        explorer_routes: Any,
        action: str,
        source: str,
        dataset: str,
        path: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        explorer_source, explorer_dataset, explorer_path = explorer_routes._resolve_dataset_context(
            source=source,
            dataset=dataset or None,
            path=path or None,
        )
        if action == "get_explorer_summary":
            payload = await _build_explorer_payload(
                explorer_source,
                explorer_dataset,
                explorer_path,
                remote_func=explorer_routes.build_remote_explorer_summary,
                local_func=explorer_routes._build_local_explorer_summary,
            )
        elif action == "get_explorer_details":
            payload = await _build_explorer_payload(
                explorer_source,
                explorer_dataset,
                explorer_path,
                remote_func=explorer_routes.build_remote_explorer_details,
                local_func=explorer_routes._build_local_explorer_details,
            )
        else:
            safe_page_size = max(1, min(page_size, 200))
            if explorer_source == "remote":
                payload = await asyncio.to_thread(
                    explorer_routes.build_remote_episode_page,
                    explorer_dataset,
                    page,
                    safe_page_size,
                )
            else:
                payload = await asyncio.to_thread(
                    explorer_routes._build_local_episode_page,
                    explorer_path,
                    explorer_dataset,
                    page,
                    safe_page_size,
                )
        return {
            "source": explorer_source,
            "dataset": explorer_dataset,
            "path": str(explorer_path) if explorer_path else "",
            "payload": payload,
        }

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
    context = deepcopy(raw)
    route = str(context.get("route") or context.get("pathname") or "").strip()
    if route:
        context["route"] = _normalize_route(route)
    return context


def _normalize_route(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    if "://" in clean:
        from urllib.parse import urlparse

        parsed = urlparse(clean)
        clean = parsed.path or "/"
    if "?" in clean:
        clean = clean.split("?", 1)[0]
    if "#" in clean:
        clean = clean.split("#", 1)[0]
    if clean.startswith("/"):
        return clean.rstrip("/") or "/"
    return clean


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _summarize_quality_results(payload: dict[str, Any]) -> dict[str, Any]:
    episodes = payload.get("episodes", [])
    sample = episodes[:5] if isinstance(episodes, list) else []
    return {
        "total": payload.get("total"),
        "passed": payload.get("passed"),
        "failed": payload.get("failed"),
        "overall_score": payload.get("overall_score"),
        "decision_counts": payload.get("decision_counts"),
        "training_weight_sum": payload.get("training_weight_sum"),
        "selected_validators": payload.get("selected_validators"),
        "threshold_overrides": payload.get("threshold_overrides"),
        "working_parquet_path": payload.get("working_parquet_path"),
        "published_parquet_path": payload.get("published_parquet_path"),
        "episode_count": len(episodes) if isinstance(episodes, list) else None,
        "sample_episodes": sample,
    }


def _summarize_alignment_overview(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    sample_rows = rows[:8] if isinstance(rows, list) else []
    return {
        "summary": payload.get("summary"),
        "distribution": payload.get("distribution"),
        "row_count": len(rows) if isinstance(rows, list) else None,
        "sample_rows": sample_rows,
    }


def _summarize_prototype_results(payload: dict[str, Any]) -> dict[str, Any]:
    clusters = payload.get("clusters", [])
    prototypes = payload.get("prototypes", [])
    return {
        key: value
        for key, value in {
            "status": payload.get("status"),
            "cluster_count": payload.get("cluster_count"),
            "candidate_count": payload.get("candidate_count"),
            "quality_filter_mode": payload.get("quality_filter_mode"),
            "source_episode_indices": payload.get("source_episode_indices"),
            "created_at": payload.get("created_at"),
            "cluster_sample": clusters[:5] if isinstance(clusters, list) else None,
            "prototype_sample": prototypes[:5] if isinstance(prototypes, list) else None,
        }.items()
        if value is not None
    }


def _summarize_propagation_results(payload: dict[str, Any]) -> dict[str, Any]:
    propagated = payload.get("propagated", [])
    return {
        key: value
        for key, value in {
            "status": payload.get("status"),
            "source_episode_index": payload.get("source_episode_index"),
            "source_episode_indices": payload.get("source_episode_indices"),
            "target_count": payload.get("target_count"),
            "propagated_count": len(propagated) if isinstance(propagated, list) else None,
            "created_at": payload.get("created_at"),
            "published_parquet_path": payload.get("published_parquet_path"),
            "sample_propagated": propagated[:8] if isinstance(propagated, list) else None,
        }.items()
        if value is not None
    }


async def _build_explorer_payload(
    source: str,
    dataset: str,
    dataset_path: Any,
    *,
    remote_func: Callable[..., Any],
    local_func: Callable[..., Any],
) -> Any:
    if source == "remote":
        return await asyncio.to_thread(remote_func, dataset)
    return await asyncio.to_thread(local_func, dataset_path, dataset)
