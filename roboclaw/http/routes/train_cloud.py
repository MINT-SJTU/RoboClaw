"""Cloud training routes backed by EVO_Train."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from roboclaw.account import AccountLedger, apply_service_fee_cents, estimate_training_hold_cents, hourly_cost_from_params
from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingPlanSpec, TrainingService, TrainingStartSpec, TrainingStopSpec

_ledger: AccountLedger | None = None


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
    hourly_cost_cents: int = Field(
        default=0,
        description="Provider hourly compute cost in cents before service fee.",
    )
    service_fee_bps: int = Field(default=1_000, description="Service fee in basis points. 1000 = 10%.")


class CloudTrainStopRequest(BaseModel):
    job_id: str
    username: str = ""


class CloudTrainBillingSettleRequest(BaseModel):
    username: str
    job_id: str
    provider_cost_cents: int = Field(..., description="Actual provider compute cost in cents before service fee.")
    service_fee_bps: int = Field(default=1_000, description="Service fee in basis points. 1000 = 10%.")
    task_name: str = ""


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


def get_ledger() -> AccountLedger:
    global _ledger
    if _ledger is None:
        _ledger = AccountLedger()
    return _ledger


def set_ledger_for_tests(ledger: AccountLedger | None) -> None:
    global _ledger
    _ledger = ledger


def register_train_cloud_routes(app: FastAPI, service: EmbodiedService) -> None:
    training = TrainingService(service)

    @app.post("/api/train/cloud/start")
    async def train_cloud_start(body: CloudTrainStartRequest) -> dict[str, Any]:
        username = body.username.strip()
        hourly_cost_cents = body.hourly_cost_cents or hourly_cost_from_params(body.params)
        hold_cents = 0
        freeze_record = None
        if username and hourly_cost_cents:
            try:
                hold_cents = estimate_training_hold_cents(
                    hourly_cost_cents=hourly_cost_cents,
                    service_fee_bps=body.service_fee_bps,
                )
                _wallet, freeze_record = get_ledger().freeze(
                    username,
                    hold_cents,
                    reason="cloud training first-hour hold",
                    task_name=body.task_name or body.dataset_name,
                    job_id=body.task_name or body.dataset_name or "pending-cloud-train",
                )
            except ValueError as exc:
                raise HTTPException(status_code=409 if "insufficient" in str(exc) else 400, detail=str(exc)) from exc
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
            if username and freeze_record is not None:
                try:
                    get_ledger().release(
                        username,
                        hold_cents,
                        reason="release hold after cloud training start failure",
                        task_name=body.task_name or body.dataset_name,
                        job_id=freeze_record.job_id,
                    )
                except ValueError:
                    pass
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = result.to_dict()
        if username and freeze_record is not None:
            job_id = payload.get("job_id") or freeze_record.job_id
            if job_id != freeze_record.job_id:
                try:
                    freeze_record = get_ledger().reassign_job_hold(
                        username,
                        freeze_record.job_id,
                        str(job_id),
                    )
                except ValueError:
                    pass
            payload["billing"] = {
                "holdCents": hold_cents,
                "hourlyCostCents": hourly_cost_cents,
                "serviceFeeBps": body.service_fee_bps,
                "record": freeze_record.to_dict(),
            }
        return payload

    @app.post("/api/train/cloud/billing/settle")
    async def train_cloud_billing_settle(body: CloudTrainBillingSettleRequest) -> dict[str, Any]:
        try:
            charge_cents = apply_service_fee_cents(
                body.provider_cost_cents,
                service_fee_bps=body.service_fee_bps,
            )
            wallet, settle_record, release_record = await asyncio.to_thread(
                get_ledger().settle_job,
                body.username,
                body.job_id,
                charge_cents,
                reason="cloud training final settlement",
                task_name=body.task_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "exceeds" in str(exc) or "no frozen" in str(exc) else 400, detail=str(exc)) from exc
        return {
            "wallet": wallet.to_dict(),
            "chargeCents": charge_cents,
            "providerCostCents": body.provider_cost_cents,
            "serviceFeeBps": body.service_fee_bps,
            "settleRecord": settle_record.to_dict(),
            "releaseRecord": release_record.to_dict() if release_record else None,
        }

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
