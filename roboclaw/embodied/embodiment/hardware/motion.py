"""Motion detection helpers."""
from __future__ import annotations

from typing import Any

MOTION_THRESHOLD = 50


def detect_motion(baseline: dict[int, int], current: dict[int, int]) -> int:
    """Compute total absolute delta between baseline and current positions."""
    total = 0
    for mid, base_val in baseline.items():
        cur_val = current.get(mid)
        if cur_val is None:
            continue
        total += abs(cur_val - base_val)
    return total


def _result_id(result: dict[str, Any]) -> str:
    """Return a stable identifier for a motion result row."""
    return str(result.get("stable_id") or result.get("port_id") or result.get("dev") or "")


def resolve_active_motion(
    results: list[dict[str, Any]],
    active_id: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Resolve the single active arm for this poll.

    Fresh motion always wins. If nothing new moved in this poll, keep the
    previous active arm latched so the UI can stay focused on that device while
    the user lets go and types a name. The latch is cleared automatically when
    the active arm disappears from the current candidate set.
    """
    strongest = None
    for result in results:
        if not result.get("moved"):
            continue
        if strongest is None or int(result.get("delta", 0)) > int(strongest.get("delta", 0)):
            strongest = result

    next_active_id = _result_id(strongest) if strongest else ""
    if not next_active_id and active_id:
        for result in results:
            if _result_id(result) == active_id:
                next_active_id = active_id
                break

    normalized: list[dict[str, Any]] = []
    for result in results:
        normalized.append({**result, "moved": _result_id(result) == next_active_id})
    return normalized, next_active_id


def keep_strongest_motion(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the strongest active motion result in a poll batch.

    Setup identification is a single-choice interaction: the user is pointing at
    exactly one arm by moving it. If multiple ports cross the threshold in the
    same poll, we keep the one with the largest delta and clear the rest.
    """
    normalized, _ = resolve_active_motion(results)
    return normalized
