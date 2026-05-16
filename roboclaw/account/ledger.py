"""Small persistent credit ledger for account billing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

LedgerKind = Literal["admin_recharge", "freeze", "settle", "release"]


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
            return {"wallets": {}, "records": []}
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


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
