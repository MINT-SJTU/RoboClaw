"""Account credit and billing routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from roboclaw.account import AccountLedger

_ledger: AccountLedger | None = None


class RechargeRequest(BaseModel):
    username: str
    amount_cents: int
    reason: str = "admin recharge"


class TopupOrderRequest(BaseModel):
    username: str
    amount_cents: int = Field(..., description="Training credit in cents. 100 cents = 1 CNY.")
    bonus_points: int = Field(
        default=0,
        description="Small non-cash reward points granted as a top-up bonus, e.g. 5-20.",
    )
    provider: str = "mock"
    reason: str = "credit topup"


class CompleteTopupOrderRequest(BaseModel):
    order_id: str
    provider_order_id: str = ""


class DatasetRewardRequest(BaseModel):
    username: str
    dataset_id: str
    reward_points: int = Field(
        ...,
        description="Small non-cash contribution points for an accepted dataset, e.g. 10-100.",
    )
    reason: str = "dataset upload reward"


class BillingAmountRequest(BaseModel):
    username: str
    amount_cents: int
    reason: str = ""
    task_name: str = ""
    job_id: str = ""


def get_ledger() -> AccountLedger:
    global _ledger
    if _ledger is None:
        _ledger = AccountLedger()
    return _ledger


def set_ledger_for_tests(ledger: AccountLedger | None) -> None:
    global _ledger
    _ledger = ledger


def register_account_routes(app: FastAPI) -> None:
    @app.get("/api/account/balance")
    async def account_balance(username: str) -> dict[str, Any]:
        try:
            wallet = await asyncio.to_thread(get_ledger().wallet, username)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict()}

    @app.get("/api/account/billing-records")
    async def billing_records(username: str = "", limit: int = 50) -> dict[str, Any]:
        records = await asyncio.to_thread(get_ledger().records, username, limit=limit)
        return {"records": [record.to_dict() for record in records]}

    @app.get("/api/account/topup-orders")
    async def account_topup_orders(username: str = "", limit: int = 50) -> dict[str, Any]:
        orders = await asyncio.to_thread(get_ledger().orders, username, limit=limit)
        return {"orders": [order.to_dict() for order in orders]}

    @app.post("/api/account/topup-orders")
    async def create_topup_order(body: TopupOrderRequest) -> dict[str, Any]:
        try:
            order = await asyncio.to_thread(
                get_ledger().create_topup_order,
                body.username,
                body.amount_cents,
                bonus_points=body.bonus_points,
                provider=body.provider,
                reason=body.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"order": order.to_dict()}

    @app.post("/api/account/topup-orders/complete")
    async def complete_topup_order(body: CompleteTopupOrderRequest) -> dict[str, Any]:
        try:
            order, wallet, record = await asyncio.to_thread(
                get_ledger().complete_topup_order,
                body.order_id,
                provider_order_id=body.provider_order_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404 if "not found" in str(exc) else 400, detail=str(exc)) from exc
        return {
            "order": order.to_dict(),
            "wallet": wallet.to_dict(),
            "record": record.to_dict() if record else None,
        }

    @app.post("/api/account/rewards/dataset-upload")
    async def grant_dataset_upload_reward(body: DatasetRewardRequest) -> dict[str, Any]:
        try:
            wallet, record, granted = await asyncio.to_thread(
                get_ledger().grant_dataset_reward,
                body.username,
                body.dataset_id,
                body.reward_points,
                reason=body.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict(), "record": record.to_dict(), "granted": granted}

    @app.post("/api/admin/account/recharge")
    async def admin_account_recharge(body: RechargeRequest) -> dict[str, Any]:
        try:
            wallet, record = await asyncio.to_thread(
                get_ledger().admin_recharge,
                body.username,
                body.amount_cents,
                reason=body.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict(), "record": record.to_dict()}

    @app.post("/api/billing/freeze")
    async def billing_freeze(body: BillingAmountRequest) -> dict[str, Any]:
        try:
            wallet, record = await asyncio.to_thread(
                get_ledger().freeze,
                body.username,
                body.amount_cents,
                reason=body.reason or "freeze credits",
                task_name=body.task_name,
                job_id=body.job_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "insufficient" in str(exc) else 400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict(), "record": record.to_dict()}

    @app.post("/api/billing/settle")
    async def billing_settle(body: BillingAmountRequest) -> dict[str, Any]:
        try:
            wallet, record = await asyncio.to_thread(
                get_ledger().settle,
                body.username,
                body.amount_cents,
                reason=body.reason or "settle credits",
                task_name=body.task_name,
                job_id=body.job_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "exceeds" in str(exc) else 400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict(), "record": record.to_dict()}

    @app.post("/api/billing/release")
    async def billing_release(body: BillingAmountRequest) -> dict[str, Any]:
        try:
            wallet, record = await asyncio.to_thread(
                get_ledger().release,
                body.username,
                body.amount_cents,
                reason=body.reason or "release frozen credits",
                task_name=body.task_name,
                job_id=body.job_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "exceeds" in str(exc) else 400, detail=str(exc)) from exc
        return {"wallet": wallet.to_dict(), "record": record.to_dict()}


def ledger_for_path(path: Path) -> AccountLedger:
    return AccountLedger(path)
