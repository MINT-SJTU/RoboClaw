"""Workflow routes — unified embodied workflow planning and execution."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException

from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.workflow import WorkflowSpec

WorkflowPhase = Literal["record", "train", "infer"]


def register_workflow_routes(app: FastAPI, service: EmbodiedService) -> None:
    @app.post(
        "/api/workflows/plan",
        summary="Preview the compiled workflow plan",
        description=(
            "Resolve cross-stage dataset flow, checkpoint paths, and concrete commands "
            "for a unified embodied workflow spec."
        ),
    )
    async def workflow_plan(body: WorkflowSpec) -> dict[str, Any]:
        return service.plan_workflow(body).model_dump(by_alias=True)

    @app.post(
        "/api/workflows/run/{phase}",
        summary="Run one phase from a validated workflow",
        description=(
            "Execute a single workflow phase using the same derived dataset and "
            "checkpoint values shown in the workflow plan."
        ),
    )
    async def workflow_run(phase: WorkflowPhase, body: WorkflowSpec) -> dict[str, Any]:
        try:
            return await service.start_workflow_phase(body, phase)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
