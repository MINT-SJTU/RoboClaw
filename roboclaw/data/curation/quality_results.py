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
    episodes = [_with_decision_fields(ep) for ep in per_episode]
    scores = [ep["score"] for ep in episodes]
    overall_score = (sum(scores) / len(scores)) if scores else 0.0
    decision_counts = _decision_counts(episodes)
    return {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "overall_score": round(overall_score, 1),
        "decision_counts": decision_counts,
        "training_weight_sum": round(
            sum(float(ep.get("training_weight", 0.0) or 0.0) for ep in episodes),
            3,
        ),
        "selected_validators": selected_validators,
        "threshold_overrides": threshold_overrides or {},
        "episodes": episodes,
    }


def _with_decision_fields(episode: dict[str, Any]) -> dict[str, Any]:
    score = float(episode.get("score", 0.0) or 0.0)
    passed = bool(episode.get("passed", False))
    task_confidence = _task_confidence(episode)
    if not passed:
        label = "reject"
        weight = 0.0
        reason = "blocking_quality_issue"
    elif score >= 85.0 and task_confidence is not None and task_confidence >= 0.75:
        label = "accept"
        weight = 1.0
        reason = "passed_score_and_task_confidence"
    elif score >= 60.0:
        label = "review"
        weight = 0.5
        reason = "task_confidence_unavailable" if task_confidence is None else "needs_review"
    else:
        label = "low_weight"
        weight = 0.2
        reason = "low_quality_score"
    return {
        **episode,
        "decision_label": label,
        "decision_reason": reason,
        "task_confidence": task_confidence,
        "training_weight": weight,
    }


def _decision_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"accept": 0, "review": 0, "low_weight": 0, "reject": 0}
    for episode in episodes:
        label = str(episode.get("decision_label", "reject"))
        if label in counts:
            counts[label] += 1
    return counts


def _task_confidence(episode: dict[str, Any]) -> float | None:
    for key in ("task_confidence", "C_task"):
        value = _safe_confidence(episode.get(key))
        if value is not None:
            return value
    semantic = episode.get("semantic")
    if isinstance(semantic, dict):
        value = _safe_confidence(semantic.get("task_confidence"))
        if value is not None:
            return value
    return None


def _safe_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


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
