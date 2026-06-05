"""Cloud training automation policy helpers."""

from __future__ import annotations

from typing import Any

_DEFAULT_SAFE_AUTOMATION_POLICY: dict[str, Any] = {
    "mode": "safe_auto",
    "autoRetrySameRuntime": True,
    "allowAgentRepairSameRuntime": True,
    "paidStartRequiresConfirmation": True,
}


def _resolve_automation_policy(
    automation_policy: dict[str, Any] | None,
    automation_mode: str = "",
) -> dict[str, Any]:
    policy = dict(_DEFAULT_SAFE_AUTOMATION_POLICY)
    supplied = dict(automation_policy or {})
    policy.update(supplied)
    requested_mode = str(automation_mode or "").strip()
    if requested_mode and "mode" not in supplied:
        policy["mode"] = requested_mode
    if policy.get("mode") == "full_auto":
        policy.setdefault("autoRetrySameRuntime", True)
        policy.setdefault("allowAgentRepairSameRuntime", True)
        if "paidStartRequiresConfirmation" not in supplied:
            policy["paidStartRequiresConfirmation"] = False
        else:
            policy["paidStartRequiresConfirmation"] = bool(policy.get("paidStartRequiresConfirmation", False))
    return policy
