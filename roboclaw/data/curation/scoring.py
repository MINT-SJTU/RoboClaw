"""Quality scoring and contribution reward helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.account import AccountLedger

from .state import load_dataset_info, load_quality_results, load_workflow_state

_QUALITY_REWARD_POINTS_PER_VALID_MINUTE = 1.0
_QUALITY_REWARD_MAX_POINTS_PER_DATASET = 500


def _load_episode_duration(dataset_path: Path, episode_index: int) -> float:
    from .validators import load_episode_data

    data = load_episode_data(dataset_path, episode_index)
    rows = data["rows"]
    if len(rows) < 2:
        return 0.0
    from .features import resolve_timestamp
    timestamps = [resolve_timestamp(r) for r in rows]
    valid = [t for t in timestamps if t is not None]
    if len(valid) < 2:
        return 0.0
    return max(valid[-1] - valid[0], 0.0)


def _grant_quality_reward(
    *,
    dataset_path: Path,
    dataset_id: str,
    username: str,
) -> None:
    """Grant contribution points for quality-passed dataset content.

    Rewarding is deliberately best-effort: quality validation should remain
    the source of truth even when the account ledger is temporarily unavailable.
    """
    reward_username = _resolve_quality_reward_username(dataset_path, username)
    if not reward_username:
        return

    state = load_workflow_state(dataset_path)
    quality_stage = state.get("stages", {}).get("quality_validation", {})
    if quality_stage.get("status") != "completed":
        return

    results = load_quality_results(dataset_path) or {}
    reward_points = _calculate_quality_reward_points(dataset_path, results)
    if reward_points <= 0:
        return

    try:
        service_module = sys.modules.get("roboclaw.data.curation.service")
        ledger_factory = getattr(service_module, "AccountLedger", AccountLedger)
        _wallet, _record, granted = ledger_factory().grant_dataset_reward(
            reward_username,
            dataset_id,
            reward_points,
            reason="quality validation reward",
        )
    except Exception as exc:  # pragma: no cover - defensive integration guard
        logger.warning(
            "Failed to grant quality reward for dataset '{}': {}",
            dataset_id,
            exc,
        )
        return

    if granted:
        logger.info(
            "Granted {} quality reward points to '{}' for dataset '{}'",
            reward_points,
            reward_username,
            dataset_id,
        )


def _resolve_quality_reward_username(dataset_path: Path, fallback_username: str = "") -> str:
    """Resolve reward owner from dataset metadata, falling back to caller username."""
    info = load_dataset_info(dataset_path)
    owner = (
        info.get("ownerUsername")
        or info.get("owner_username")
        or info.get("uploaderUsername")
        or info.get("uploader_username")
        or info.get("uploadedBy")
        or info.get("uploaded_by")
    )
    if isinstance(owner, dict):
        owner = owner.get("username") or owner.get("name")
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    return fallback_username.strip()


def _calculate_quality_reward_points(dataset_path: Path, results: dict[str, Any]) -> int:
    passed_episode_indices = [
        episode.get("episode_index")
        for episode in results.get("episodes", [])
        if episode.get("passed") and episode.get("episode_index") is not None
    ]
    valid_seconds = 0.0
    service_module = sys.modules.get("roboclaw.data.curation.service")
    duration_loader = getattr(service_module, "_load_episode_duration", _load_episode_duration)
    for value in passed_episode_indices:
        try:
            valid_seconds += duration_loader(dataset_path, int(value))
        except Exception:
            logger.debug(
                "Failed to load episode duration for reward calculation: dataset={}, episode={}",
                dataset_path,
                value,
                exc_info=True,
            )
    if valid_seconds <= 0:
        return 0

    valid_minutes = valid_seconds / 60.0
    quality_multiplier = _quality_reward_multiplier(results.get("overall_score"))
    source_multiplier = _source_reward_multiplier(load_dataset_info(dataset_path))
    raw_points = valid_minutes * _QUALITY_REWARD_POINTS_PER_VALID_MINUTE * quality_multiplier * source_multiplier
    if raw_points <= 0:
        return 0
    return min(max(math.ceil(raw_points), 1), _QUALITY_REWARD_MAX_POINTS_PER_DATASET)


def _quality_reward_multiplier(score: Any) -> float:
    try:
        normalized = float(score)
    except (TypeError, ValueError):
        normalized = 0.0
    if normalized <= 1.0:
        normalized *= 100.0
    if normalized >= 95.0:
        return 1.2
    if normalized >= 85.0:
        return 1.0
    if normalized >= 70.0:
        return 0.5
    return 0.0


def _source_reward_multiplier(info: dict[str, Any]) -> float:
    visibility = str(
        info.get("visibility")
        or info.get("accessLevel")
        or info.get("access_level")
        or "private",
    ).strip().lower()
    if visibility not in {"public", "shared", "open"}:
        return 0.0

    source_type = str(
        info.get("contributionSource")
        or info.get("contribution_source")
        or info.get("sourceType")
        or info.get("source_type")
        or "self_collected",
    ).strip().lower()
    if source_type in {"public", "public_dataset", "cleaned_public_dataset", "imported_public"}:
        return 0.3
    if source_type in {"synthetic", "simulation", "sim"}:
        return 0.2
    if source_type in {"duplicate", "imported_only"}:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# CurationService
# ---------------------------------------------------------------------------


def _aggregate_quality_results(
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


# ---------------------------------------------------------------------------
# Prototype helpers
# ---------------------------------------------------------------------------


def _collect_passed_episodes(dataset_path: Path) -> list[int]:
    quality = load_quality_results(dataset_path)
    if quality is None:
        return []
    return [
        ep["episode_index"]
        for ep in quality.get("episodes", [])
        if ep.get("passed")
    ]


def _episode_quality_summary(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    quality = load_quality_results(dataset_path)
    if quality is None:
        return {}
    for ep in quality.get("episodes", []):
        if ep.get("episode_index") == episode_index:
            return {"score": ep.get("score", 0), "passed": ep.get("passed", False)}
    return {}
