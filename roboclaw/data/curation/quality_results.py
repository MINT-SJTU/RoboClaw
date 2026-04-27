from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .validators import run_quality_validators


def aggregate_quality_results(
    per_episode: list[dict[str, Any]],
    selected_validators: list[str],
    passed_count: int,
    failed_count: int,
    total: int,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    scores = [ep["score"] for ep in per_episode]
    overall_score = (sum(scores) / len(scores)) if scores else 0.0
    return {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "overall_score": round(overall_score, 1),
        "selected_validators": selected_validators,
        "threshold_overrides": threshold_overrides or {},
        "episodes": per_episode,
    }


def run_base_quality_validators(
    dataset_path: Path,
    episode_index: int,
    *,
    selected_validators: list[str],
    threshold_overrides: dict[str, float] | None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not selected_validators:
        return {
            "passed": True,
            "score": 100.0,
            "validators": {},
            "issues": [],
        }
    quality_runner = runner or run_quality_validators
    return quality_runner(
        dataset_path,
        episode_index,
        selected_validators=selected_validators,
        threshold_overrides=threshold_overrides,
    )
