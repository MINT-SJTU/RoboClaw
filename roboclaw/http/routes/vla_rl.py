"""VLA-RL planning and artifact review routes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingService
from roboclaw.training.vla_rl import VLAPlanRequest, VLARLService, deployability_gate


class VLARLPlanBody(BaseModel):
    username: str = ""
    message: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    provider: str = "autodl"
    sku_id: str = ""
    image_id: str = ""


class ArtifactReviewBody(BaseModel):
    contract_path: str = ""
    contract: dict[str, Any] | None = None


class DeployabilityBody(BaseModel):
    contract: dict[str, Any] = Field(default_factory=dict)
    robot_embodiment: str = ""
    observation_schema: str = ""
    action_schema: str = ""


def _bridge_error_status(exc: RuntimeError) -> int:
    return 503 if "bridge is not enabled" in str(exc).lower() else 502


def register_vla_rl_routes(app: FastAPI, service: EmbodiedService, llm_provider: Any | None = None) -> None:
    vla_rl = VLARLService(TrainingService(service), llm_provider=llm_provider)

    @app.get("/api/vla-rl/profiles")
    async def vla_rl_profiles() -> dict[str, Any]:
        return vla_rl.profiles()

    @app.get("/api/vla-rl/rlinf-catalog")
    async def vla_rl_rlinf_catalog() -> dict[str, Any]:
        return vla_rl.rlinf_catalog()

    @app.get("/api/vla-rl/playground")
    async def vla_rl_playground() -> dict[str, Any]:
        return vla_rl.playground()

    @app.post("/api/vla-rl/plan")
    async def vla_rl_plan(body: VLARLPlanBody, request: Request) -> dict[str, Any]:
        try:
            vla_rl._llm_provider = llm_provider or getattr(request.app.state, "llm_provider", None)
            return await vla_rl.plan(
                VLAPlanRequest(
                    username=body.username,
                    message=body.message,
                    workflow=body.workflow,
                    params=body.params,
                    provider=body.provider,
                    sku_id=body.sku_id,
                    image_id=body.image_id,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.post("/api/vla-rl/artifact-review")
    async def vla_rl_artifact_review(body: ArtifactReviewBody) -> dict[str, Any]:
        try:
            return vla_rl.review_artifact(contract_path=body.contract_path, contract=body.contract)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/vla-rl/deployability")
    async def vla_rl_deployability(body: DeployabilityBody) -> dict[str, Any]:
        return deployability_gate(
            body.contract,
            robot_embodiment=body.robot_embodiment,
            observation_schema=body.observation_schema,
            action_schema=body.action_schema,
        )
