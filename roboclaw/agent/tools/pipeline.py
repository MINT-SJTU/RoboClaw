"""Pipeline tool for the in-app RoboClaw AI."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool


class PipelineTool(Tool):
    """Let RoboClaw AI inspect and trigger curation Pipeline stages."""

    @property
    def name(self) -> str:
        return "pipeline"

    @property
    def description(self) -> str:
        return (
            "Control RoboClaw's built-in curation Pipeline: list datasets, inspect workflow state, "
            "get dataset-aware quality defaults, and start/pause/resume quality/prototype/propagation runs."
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
                return _json(result)

            if action == "pause_quality":
                state = service.get_workflow_state(dataset_path)
                if state["stages"]["quality_validation"].get("status") != "running":
                    return _json({"error": "Quality validation is not running"})
                set_stage_pause_requested(dataset_path, "quality_validation", True)
                return _json({"status": "pause_requested"})

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
                return _json(result)

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


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
