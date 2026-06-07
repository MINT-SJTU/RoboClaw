"""Evo Studio agent-consult routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingService
from roboclaw.training.agent_consult import EvoStudioAgentConsultRequest, EvoStudioAgentConsultService

_log = logging.getLogger(__name__)


class AgentConsultBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task: str = ""
    mode: str = "plan"
    username: str = ""
    provider: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""
    job_id: str = ""
    confirmed: bool = False
    automation_mode: str = Field(default="ask", alias="automationMode")
    automation_policy: dict[str, Any] = Field(default_factory=dict)


def _bridge_error_status(exc: RuntimeError) -> int:
    return 503 if "bridge is not enabled" in str(exc).lower() else 502


def register_agent_consult_routes(
    app: FastAPI,
    service: EmbodiedService,
    *,
    llm_provider: Any | None = None,
) -> None:
    training = TrainingService(service)

    @app.post("/api/evo-studio/agent-consult")
    async def evo_studio_agent_consult(body: AgentConsultBody, request: Request) -> dict[str, Any]:
        task = body.task.strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        provider = llm_provider or getattr(request.app.state, "llm_provider", None)
        consult = EvoStudioAgentConsultService(training, llm_provider=provider)
        automation_policy = dict(body.automation_policy or {})
        if body.automation_mode and "mode" not in automation_policy:
            automation_policy["mode"] = body.automation_mode
        try:
            return await consult.consult(
                EvoStudioAgentConsultRequest(
                    task=task,
                    mode=body.mode,
                    username=body.username,
                    provider=body.provider,
                    workflow=body.workflow,
                    params=body.params,
                    context=body.context,
                    sku_id=body.sku_id,
                    image_id=body.image_id,
                    job_id=body.job_id,
                    confirmed=body.confirmed or body.automation_mode == "full_auto",
                    automation_mode=body.automation_mode,
                    automation_policy=automation_policy,
                )
            )
        except RuntimeError as exc:
            _log.warning("Agent consult failed: %s", exc)
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
