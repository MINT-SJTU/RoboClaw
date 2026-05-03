"""Recovery helpers for the dashboard."""

from __future__ import annotations

from typing import Any


def get_recovery_guides_json() -> dict[str, Any]:
    """Return recovery-guide payload for dashboard clients.

    The detailed guide catalog can grow later without changing the route shape.
    """
    return {"guides": []}
