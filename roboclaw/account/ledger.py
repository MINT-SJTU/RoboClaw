"""Small persistent credit ledger for account billing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

LedgerKind = Literal["admin_recharge", "payment_recharge", "dataset_reward", "freeze", "settle", "release"]
PaymentOrderStatus = Literal["pending", "paid", "cancelled"]


@dataclass(frozen=True)
class Wallet:
    username: str
    balance_cents: int = 0
    frozen_cents: int = 0
    updated_at: str = ""

    @property
    def available_cents(self) -> int:
        return self.balance_cents - self.frozen_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "balanceCents": self.balance_cents,
            "frozenCents": self.frozen_cents,
            "availableCents": self.available_cents,
            "updatedAt": self.updated_at,
        }


@dataclass(frozen=True)
class BillingRecord:
    record_id: str
    username: str
    kind: LedgerKind
    amount_cents: int
    balance_after_cents: int
    frozen_after_cents: int
    reason: str = ""
    task_name: str = ""
    job_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordId": self.record_id,
            "username": self.username,
            "kind": self.kind,
            "amountCents": self.amount_cents,
            "balanceAfterCents": self.balance_after_cents,
            "frozenAfterCents": self.frozen_after_cents,
            "reason": self.reason,
            "taskName": self.task_name,
            "jobId": self.job_id,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class PaymentOrder:
    order_id: str
    username: str
    amount_cents: int
    provider: str = "mock"
    status: PaymentOrderStatus = "pending"
    provider_order_id: str = ""
    pay_url: str = ""
    reason: str = "credit topup"
    created_at: str = ""
    paid_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "orderId": self.order_id,
            "username": self.username,
            "amountCents": self.amount_cents,
            "provider": self.provider,
            "status": self.status,
            "providerOrderId": self.provider_order_id,
            "payUrl": self.pay_url,
            "reason": self.reason,
            "createdAt": self.created_at,
            "paidAt": self.paid_at,
        }


class AccountLedger:
    """File-backed wallet ledger.

    This is intentionally small and swappable: production can replace it with
    MySQL/RDS while preserving route-level semantics.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".roboclaw" / "account_ledger.json"
        self._lock = threading.Lock()

    def wallet(self, username: str) -> Wallet:
        username = _clean_username(username)
        with self._lock:
            state = self._load()
            return self._wallet_from_state(state, username)

    def records(self, username: str = "", *, limit: int = 50) -> list[BillingRecord]:
        with self._lock:
            state = self._load()
            records = [_record_from_payload(item) for item in state.get("records", [])]
        if username:
            records = [record for record in records if record.username == username]
        return records[-max(limit, 0) :][::-1]

    def orders(self, username: str = "", *, limit: int = 50) -> list[PaymentOrder]:
        with self._lock:
            state = self._load()
            orders = [_order_from_payload(item) for item in state.get("paymentOrders", [])]
        if username:
            orders = [order for order in orders if order.username == username]
        return orders[-max(limit, 0) :][::-1]

    def create_topup_order(
        self,
        username: str,
        amount_cents: int,
        *,
        provider: str = "mock",
        reason: str = "credit topup",
    ) -> PaymentOrder:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        provider = (provider or "mock").strip()
        if not provider:
            raise ValueError("provider is required")
        with self._lock:
            state = self._load()
            order_id = uuid4().hex
            order = PaymentOrder(
                order_id=order_id,
                username=username,
                amount_cents=amount_cents,
                provider=provider,
                status="pending",
                provider_order_id=f"{provider}_{order_id}",
                pay_url=f"roboclaw://pay/{provider}/{order_id}",
                reason=reason,
                created_at=_now(),
            )
            state.setdefault("paymentOrders", []).append(order.to_dict())
            state.setdefault("wallets", {}).setdefault(username, self._wallet_from_state(state, username).to_dict())
            self._save(state)
            return order

    def complete_topup_order(
        self,
        order_id: str,
        *,
        provider_order_id: str = "",
    ) -> tuple[PaymentOrder, Wallet, BillingRecord | None]:
        order_id = order_id.strip()
        if not order_id:
            raise ValueError("order_id is required")
        with self._lock:
            state = self._load()
            orders = state.setdefault("paymentOrders", [])
            for index, payload in enumerate(orders):
                order = _order_from_payload(payload)
                if order.order_id != order_id:
                    continue
                if order.status == "paid":
                    wallet = self._wallet_from_state(state, order.username)
                    return order, wallet, None
                if order.status != "pending":
                    raise ValueError(f"cannot complete {order.status} order")
                paid_order = PaymentOrder(
                    order_id=order.order_id,
                    username=order.username,
                    amount_cents=order.amount_cents,
                    provider=order.provider,
                    status="paid",
                    provider_order_id=provider_order_id or order.provider_order_id,
                    pay_url=order.pay_url,
                    reason=order.reason,
                    created_at=order.created_at,
                    paid_at=_now(),
                )
                wallet = self._wallet_from_state(state, order.username)
                wallet = Wallet(
                    username=wallet.username,
                    balance_cents=wallet.balance_cents + order.amount_cents,
                    frozen_cents=wallet.frozen_cents,
                    updated_at=_now(),
                )
                record = self._append_record(
                    state,
                    wallet,
                    "payment_recharge",
                    order.amount_cents,
                    reason=f"{order.provider} payment recharge",
                    job_id=order.order_id,
                )
                orders[index] = paid_order.to_dict()
                self._save_wallet(state, wallet)
                self._save(state)
                return paid_order, wallet, record
        raise ValueError("payment order not found")

    def grant_dataset_reward(
        self,
        username: str,
        dataset_id: str,
        amount_cents: int,
        *,
        reason: str = "dataset upload reward",
    ) -> tuple[Wallet, BillingRecord, bool]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError("dataset_id is required")
        with self._lock:
            state = self._load()
            for payload in state.get("records", []):
                record = _record_from_payload(payload)
                if record.kind == "dataset_reward" and record.username == username and record.job_id == dataset_id:
                    return self._wallet_from_state(state, username), record, False
            wallet = self._wallet_from_state(state, username)
            wallet = Wallet(
                username=username,
                balance_cents=wallet.balance_cents + amount_cents,
                frozen_cents=wallet.frozen_cents,
                updated_at=_now(),
            )
            record = self._append_record(
                state,
                wallet,
                "dataset_reward",
                amount_cents,
                reason=reason,
                job_id=dataset_id,
            )
            self._save_wallet(state, wallet)
            self._save(state)
            return wallet, record, True

    def admin_recharge(self, username: str, amount_cents: int, *, reason: str = "admin recharge") -> tuple[Wallet, BillingRecord]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        with self._lock:
            state = self._load()
            wallet = self._wallet_from_state(state, username)
            wallet = Wallet(
                username=username,
                balance_cents=wallet.balance_cents + amount_cents,
                frozen_cents=wallet.frozen_cents,
                updated_at=_now(),
            )
            record = self._append_record(state, wallet, "admin_recharge", amount_cents, reason=reason)
            self._save_wallet(state, wallet)
            self._save(state)
            return wallet, record

    def freeze(
        self,
        username: str,
        amount_cents: int,
        *,
        reason: str = "freeze credits",
        task_name: str = "",
        job_id: str = "",
    ) -> tuple[Wallet, BillingRecord]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        with self._lock:
            state = self._load()
            wallet = self._wallet_from_state(state, username)
            if wallet.available_cents < amount_cents:
                raise ValueError("insufficient available balance")
            wallet = Wallet(
                username=username,
                balance_cents=wallet.balance_cents,
                frozen_cents=wallet.frozen_cents + amount_cents,
                updated_at=_now(),
            )
            record = self._append_record(
                state,
                wallet,
                "freeze",
                amount_cents,
                reason=reason,
                task_name=task_name,
                job_id=job_id,
            )
            self._save_wallet(state, wallet)
            self._save(state)
            return wallet, record

    def settle(
        self,
        username: str,
        amount_cents: int,
        *,
        reason: str = "settle credits",
        task_name: str = "",
        job_id: str = "",
    ) -> tuple[Wallet, BillingRecord]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        with self._lock:
            state = self._load()
            wallet = self._wallet_from_state(state, username)
            if wallet.frozen_cents < amount_cents:
                raise ValueError("settle amount exceeds frozen balance")
            wallet = Wallet(
                username=username,
                balance_cents=wallet.balance_cents - amount_cents,
                frozen_cents=wallet.frozen_cents - amount_cents,
                updated_at=_now(),
            )
            record = self._append_record(
                state,
                wallet,
                "settle",
                -amount_cents,
                reason=reason,
                task_name=task_name,
                job_id=job_id,
            )
            self._save_wallet(state, wallet)
            self._save(state)
            return wallet, record

    def release(
        self,
        username: str,
        amount_cents: int,
        *,
        reason: str = "release frozen credits",
        task_name: str = "",
        job_id: str = "",
    ) -> tuple[Wallet, BillingRecord]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        username = _clean_username(username)
        with self._lock:
            state = self._load()
            wallet = self._wallet_from_state(state, username)
            if wallet.frozen_cents < amount_cents:
                raise ValueError("release amount exceeds frozen balance")
            wallet = Wallet(
                username=username,
                balance_cents=wallet.balance_cents,
                frozen_cents=wallet.frozen_cents - amount_cents,
                updated_at=_now(),
            )
            record = self._append_record(
                state,
                wallet,
                "release",
                amount_cents,
                reason=reason,
                task_name=task_name,
                job_id=job_id,
            )
            self._save_wallet(state, wallet)
            self._save(state)
            return wallet, record

    def _append_record(
        self,
        state: dict[str, Any],
        wallet: Wallet,
        kind: LedgerKind,
        amount_cents: int,
        *,
        reason: str = "",
        task_name: str = "",
        job_id: str = "",
    ) -> BillingRecord:
        record = BillingRecord(
            record_id=uuid4().hex,
            username=wallet.username,
            kind=kind,
            amount_cents=amount_cents,
            balance_after_cents=wallet.balance_cents,
            frozen_after_cents=wallet.frozen_cents,
            reason=reason,
            task_name=task_name,
            job_id=job_id,
            created_at=_now(),
        )
        state.setdefault("records", []).append(record.to_dict())
        return record

    def _wallet_from_state(self, state: dict[str, Any], username: str) -> Wallet:
        payload = state.setdefault("wallets", {}).get(username) or {}
        return Wallet(
            username=username,
            balance_cents=int(payload.get("balanceCents", 0) or 0),
            frozen_cents=int(payload.get("frozenCents", 0) or 0),
            updated_at=str(payload.get("updatedAt") or ""),
        )

    def _save_wallet(self, state: dict[str, Any], wallet: Wallet) -> None:
        state.setdefault("wallets", {})[wallet.username] = wallet.to_dict()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"wallets": {}, "records": [], "paymentOrders": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_username(username: str) -> str:
    value = username.strip()
    if not value:
        raise ValueError("username is required")
    return value


def _record_from_payload(payload: dict[str, Any]) -> BillingRecord:
    return BillingRecord(
        record_id=str(payload.get("recordId") or ""),
        username=str(payload.get("username") or ""),
        kind=str(payload.get("kind") or "freeze"),  # type: ignore[arg-type]
        amount_cents=int(payload.get("amountCents", 0) or 0),
        balance_after_cents=int(payload.get("balanceAfterCents", 0) or 0),
        frozen_after_cents=int(payload.get("frozenAfterCents", 0) or 0),
        reason=str(payload.get("reason") or ""),
        task_name=str(payload.get("taskName") or ""),
        job_id=str(payload.get("jobId") or ""),
        created_at=str(payload.get("createdAt") or ""),
    )


def _order_from_payload(payload: dict[str, Any]) -> PaymentOrder:
    return PaymentOrder(
        order_id=str(payload.get("orderId") or ""),
        username=str(payload.get("username") or ""),
        amount_cents=int(payload.get("amountCents", 0) or 0),
        provider=str(payload.get("provider") or "mock"),
        status=str(payload.get("status") or "pending"),  # type: ignore[arg-type]
        provider_order_id=str(payload.get("providerOrderId") or ""),
        pay_url=str(payload.get("payUrl") or ""),
        reason=str(payload.get("reason") or ""),
        created_at=str(payload.get("createdAt") or ""),
        paid_at=str(payload.get("paidAt") or ""),
    )


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
