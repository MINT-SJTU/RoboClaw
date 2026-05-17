"""Training billing helpers for cloud compute jobs."""

from __future__ import annotations

from math import ceil
from typing import Any, Mapping

DEFAULT_SERVICE_FEE_BPS = 1_000
DEFAULT_MIN_BILLABLE_MINUTES = 60


def estimate_training_hold_cents(
    *,
    hourly_cost_cents: int,
    service_fee_bps: int = DEFAULT_SERVICE_FEE_BPS,
    min_billable_minutes: int = DEFAULT_MIN_BILLABLE_MINUTES,
) -> int:
    """Estimate the upfront balance hold for a cloud training job."""
    if hourly_cost_cents <= 0:
        raise ValueError("hourly_cost_cents must be positive")
    if service_fee_bps < 0:
        raise ValueError("service_fee_bps must be non-negative")
    if min_billable_minutes <= 0:
        raise ValueError("min_billable_minutes must be positive")
    provider_cost = ceil(hourly_cost_cents * min_billable_minutes / 60)
    return apply_service_fee_cents(provider_cost, service_fee_bps=service_fee_bps)


def apply_service_fee_cents(provider_cost_cents: int, *, service_fee_bps: int = DEFAULT_SERVICE_FEE_BPS) -> int:
    """Convert provider cost to user-facing charge with service fee."""
    if provider_cost_cents <= 0:
        raise ValueError("provider_cost_cents must be positive")
    if service_fee_bps < 0:
        raise ValueError("service_fee_bps must be non-negative")
    return ceil(provider_cost_cents * (10_000 + service_fee_bps) / 10_000)


def hourly_cost_from_params(params: Mapping[str, Any]) -> int:
    """Read hourly provider cost from a training payload.

    The value is provider cost in cents before service fee. Accepted aliases keep
    the route compatible with EVO_Train and older AutoDL experiments.
    """
    for key in (
        "hourlyCostCents",
        "costHourlyCents",
        "estimatedHourlyCostCents",
        "hourlyPriceCents",
        "firstHourCostCents",
    ):
        value = params.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0
