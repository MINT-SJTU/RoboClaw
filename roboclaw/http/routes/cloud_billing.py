"""Cloud training ledger and wallet helpers."""

from __future__ import annotations

import os
from typing import Any

from roboclaw.account import AccountLedger

_ledger: AccountLedger | None = None

def get_ledger() -> AccountLedger:
    global _ledger
    if _ledger is None:
        _ledger = AccountLedger()
    return _ledger
def set_ledger_for_tests(ledger: AccountLedger | None) -> None:
    global _ledger
    _ledger = ledger

def _cloud_job_root(job_id: str) -> str:
    root = str(job_id or "").strip()
    for marker in ("-intervention-", "-repair-", "-restart-"):
        if marker in root:
            root = root.rsplit(marker, 1)[0]
    return root

def _attach_user_wallet(payload: dict[str, Any], username: str) -> dict[str, Any]:
    result = dict(payload)
    username = username.strip()
    executor_wallet = result.get("wallet")
    if executor_wallet is not None:
        result["executorWallet"] = executor_wallet
    if username:
        result["wallet"] = get_ledger().wallet(username).to_dict()
        result["billingMode"] = "external"
    return result
def _release_failed_cloud_hold(payload: dict[str, Any], username: str, *, reason: str) -> dict[str, Any]:
    username = username.strip()
    if not username:
        return payload
    status = str(payload.get("status") or "").strip().lower()
    running = bool(payload.get("running"))
    external_billing = str(
        payload.get("billingMode")
        or payload.get("billing_mode")
        or os.environ.get("ROBOCLAW_EVO_TRAIN_BILLING_MODE", "")
    ).strip().lower() == "external"
    releasable_statuses = {"failed", "stopped", "deleted", "missing"}
    if external_billing:
        releasable_statuses.update({"succeeded", "success", "completed", "complete"})
    if running or status not in releasable_statuses:
        return payload
    job_id = str(payload.get("job_id") or "").strip()
    task_name = str(payload.get("task_name") or "").strip()
    billing = payload.get("billing") if isinstance(payload.get("billing"), dict) else {}
    record = billing.get("record") if isinstance(billing.get("record"), dict) else {}
    billing_job_id = str(record.get("jobId") or record.get("job_id") or "").strip()
    if not any((job_id, task_name, billing_job_id)):
        return payload
    result = dict(payload)
    release_candidates = [
        candidate.strip()
        for candidate in (job_id, billing_job_id, task_name)
        if str(candidate or "").strip()
    ]
    prefix = f"task:{username}:"
    release_candidates.extend(
        candidate[len(prefix):]
        for candidate in list(release_candidates)
        if candidate.startswith(prefix)
    )
    if external_billing and status in {"succeeded", "success", "completed", "complete"}:
        roots = {_cloud_job_root(candidate) for candidate in release_candidates if _cloud_job_root(candidate)}
        for record in get_ledger().records(username, limit=500):
            if record.kind != "freeze":
                continue
            record_roots = {
                _cloud_job_root(record.job_id),
                _cloud_job_root(record.task_name),
            }
            if roots.intersection(root for root in record_roots if root):
                release_candidates.append(record.job_id)
                if record.task_name:
                    release_candidates.append(record.task_name)
    seen: set[str] = set()
    released: list[dict[str, Any]] = []
    for candidate in release_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            _wallet, release_record = get_ledger().release_job_hold(
                username,
                candidate,
                reason=reason,
                task_name=task_name,
            )
            released.append(release_record.to_dict())
            if not (external_billing and status in {"succeeded", "success", "completed", "complete"}):
                result["billingRelease"] = release_record.to_dict()
                return result
        except ValueError:
            continue
    if released:
        result["billingRelease"] = released[-1]
        result["billingReleases"] = released
    return result
