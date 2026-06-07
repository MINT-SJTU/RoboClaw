"""Agent-facing Evo Studio consult tool."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.training.agent_consult import EvoStudioAgentConsultRequest, EvoStudioAgentConsultService
from roboclaw.training.service import TrainingService


class EvoStudioAgentConsultTool(Tool):
    """Delegate product tasks to Evo Studio's backend control-plane agent."""

    def __init__(self, *, embodied_service: Any = None, llm_provider: Any = None) -> None:
        self.embodied_service = embodied_service
        self.llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "evo_studio_agent_consult"

    @property
    def description(self) -> str:
        return (
            "Delegate Evo Studio training/data/cloud operations to RoboClaw's backend agent. "
            "Use this as the stable OpenClaw-style consult surface for natural-language "
            "cloud training, runtime checks, status, and repair. The backend performs real "
            "EVO_Train checks; the outer model should not call local exec for cloud/GPU/SSH probes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "mode": {"type": "string", "enum": ["plan", "execute", "repair", "status"]},
                "username": {"type": "string"},
                "provider": {"type": "string"},
                "workflow": {"type": "string"},
                "params": {"type": "object"},
                "context": {"type": "object"},
                "sku_id": {"type": "string"},
                "image_id": {"type": "string"},
                "job_id": {"type": "string"},
                "confirmed": {"type": "boolean"},
                "automation_mode": {"type": "string", "enum": ["ask", "safe_auto", "full_auto"]},
                "automationMode": {"type": "string", "enum": ["ask", "safe_auto", "full_auto"]},
                "automation_policy": {"type": "object"},
                "automationPolicy": {"type": "object"},
            },
            "required": ["task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        task = str(kwargs.get("task") or "").strip()
        if not task:
            return "Error: task is required."
        automation_mode = str(kwargs.get("automation_mode") or kwargs.get("automationMode") or "").strip()
        automation_policy = dict(kwargs.get("automation_policy") or kwargs.get("automationPolicy") or {})
        if automation_mode and "mode" not in automation_policy:
            automation_policy["mode"] = automation_mode
        training = self._training_service()
        consult = EvoStudioAgentConsultService(training, llm_provider=self.llm_provider)
        result = await consult.consult(
            EvoStudioAgentConsultRequest(
                task=task,
                mode=str(kwargs.get("mode") or "plan"),
                username=str(kwargs.get("username") or ""),
                provider=str(kwargs.get("provider") or ""),
                workflow=str(kwargs.get("workflow") or ""),
                params=dict(kwargs.get("params") or {}),
                context=dict(kwargs.get("context") or {}),
                sku_id=str(kwargs.get("sku_id") or ""),
                image_id=str(kwargs.get("image_id") or ""),
                job_id=str(kwargs.get("job_id") or ""),
                confirmed=bool(kwargs.get("confirmed", False)) or automation_mode == "full_auto",
                automation_mode=automation_mode,
                automation_policy=automation_policy,
            )
        )
        return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)

    def _training_service(self) -> TrainingService:
        if self.embodied_service is not None:
            return TrainingService(self.embodied_service)
        from roboclaw.embodied.service import EmbodiedService

        return TrainingService(EmbodiedService())
