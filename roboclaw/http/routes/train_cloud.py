"""Cloud training routes backed by EVO_Train."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingPlanSpec, TrainingService, TrainingStartSpec, TrainingStopSpec


class CloudTrainStartRequest(BaseModel):
    dataset_name: str = ""
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"
    username: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""
    task_name: str = ""


class CloudTrainStopRequest(BaseModel):
    job_id: str
    username: str = ""


class CloudTrainPlanRequest(BaseModel):
    username: str = ""
    message: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    sku_id: str = ""
    image_id: str = ""


class RuntimeMatchRequest(BaseModel):
    username: str = ""
    provider: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""


def _bridge_error_status(exc: RuntimeError) -> int:
    return 503 if "bridge is not enabled" in str(exc).lower() else 502


def register_train_cloud_routes(app: FastAPI, service: EmbodiedService) -> None:
    training = TrainingService(service)

    @app.post("/api/train/cloud/start")
    async def train_cloud_start(body: CloudTrainStartRequest) -> dict[str, Any]:
        try:
            result = await training.start(
                TrainingStartSpec(
                    dataset_name=body.dataset_name,
                    policy_type=body.policy_type,
                    steps=body.steps,
                    device=body.device,
                    username=body.username,
                    workflow=body.workflow,
                    params=body.params,
                    sku_id=body.sku_id,
                    image_id=body.image_id,
                    task_name=body.task_name,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/api/train/cloud/stop")
    async def train_cloud_stop(body: CloudTrainStopRequest) -> dict[str, Any]:
        try:
            result = await training.stop(TrainingStopSpec(job_id=body.job_id, username=body.username))
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/api/train/cloud/current")
    async def train_cloud_current(username: str = "") -> dict[str, Any]:
        try:
            result = await training.current(username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/api/train/cloud/status/{job_id}")
    async def train_cloud_status(job_id: str, username: str = "") -> dict[str, Any]:
        try:
            result = await training.status(job_id=job_id, username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/api/train/plan")
    async def train_plan(body: CloudTrainPlanRequest) -> dict[str, Any]:
        try:
            return await training.plan(
                TrainingPlanSpec(
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

    @app.get("/api/train/gpu-skus")
    async def train_gpu_skus(provider: str = "", include_incomplete: bool = False) -> dict[str, Any]:
        try:
            return await training.gpu_skus(provider=provider, include_incomplete=include_incomplete)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.get("/api/train/images")
    async def train_images(include_incomplete: bool = False) -> dict[str, Any]:
        try:
            return await training.images(include_incomplete=include_incomplete)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.post("/api/train/runtime-match")
    async def train_runtime_match(body: RuntimeMatchRequest) -> dict[str, Any]:
        try:
            return await training.runtime_match(
                username=body.username,
                provider=body.provider,
                params=body.params,
                sku_id=body.sku_id,
                image_id=body.image_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
