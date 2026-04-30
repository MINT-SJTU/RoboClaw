from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .reference_tube import (
    TRAJECTORY_DTW_VALIDATOR,
    ReferenceTubeBuilder,
    ReferenceTubeEvaluator,
    inconclusive_result,
    unavailable_result,
)
from .trajectory_entries import build_prototype_entry, propagation_dtw_config
from .validators import _merge_threshold_overrides, weighted_validator_score


def append_trajectory_dtw_results(
    dataset_path: Path,
    per_episode: list[dict[str, Any]],
    *,
    threshold_overrides: dict[str, float] | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    if not per_episode:
        return

    thresholds = _merge_threshold_overrides(threshold_overrides)
    entry_lookup = _build_trajectory_entry_lookup(dataset_path, per_episode, progress_callback)
    groups = _group_trajectory_entries(entry_lookup.values())
    tube_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    alignment_cache: dict[str, tuple[str, list[list[float]], list[list[float]]]] = {}

    for position, episode in enumerate(per_episode):
        ep_idx = int(episode.get("episode_index", -1))
        entry = entry_lookup.get(ep_idx)
        result = _evaluate_trajectory_entry(
            entry,
            groups,
            thresholds,
            tube_cache=tube_cache,
            alignment_cache=alignment_cache,
        )
        attach_validator_result(episode, result)
        if progress_callback is not None:
            progress_callback({
                "phase": "trajectory_dtw",
                "episode_index": ep_idx,
                "completed": position + 1,
                "total": len(per_episode),
                "progress_percent": round(((position + 1) / max(len(per_episode), 1)) * 100, 1),
            })


def attach_validator_result(episode: dict[str, Any], result: dict[str, Any]) -> None:
    validators = dict(episode.get("validators", {}) or {})
    validators[TRAJECTORY_DTW_VALIDATOR] = {
        "passed": result["passed"],
        "score": result["score"],
    }
    issues = [
        issue for issue in episode.get("issues", []) or []
        if issue.get("operator_name") != TRAJECTORY_DTW_VALIDATOR
    ]
    issues.extend(result.get("issues", []) or [])

    episode["validators"] = validators
    episode["issues"] = issues
    blocking_levels = {"critical", "major"}
    episode["passed"] = all(
        issue.get("passed")
        for issue in issues
        if issue.get("level") in blocking_levels
    )
    validator_scores = [
        {"name": name, "score": value.get("score", 0.0)}
        for name, value in validators.items()
        if isinstance(value, dict)
    ]
    episode["score"] = weighted_validator_score(validator_scores) if validator_scores else 0.0


def _build_trajectory_entry_lookup(
    dataset_path: Path,
    per_episode: list[dict[str, Any]],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    total = len(per_episode)
    for position, episode in enumerate(per_episode):
        ep_idx = int(episode.get("episode_index", -1))
        if ep_idx < 0:
            continue
        entry = build_prototype_entry(
            dataset_path,
            ep_idx,
            quality={"score": episode.get("score", 0), "passed": episode.get("passed", False)},
        )
        if entry.get("sequence"):
            entry["base_quality_passed"] = bool(episode.get("passed", False))
            entries[ep_idx] = entry
        if progress_callback is not None:
            progress_callback({
                "phase": "building_trajectory_dtw",
                "episode_index": ep_idx,
                "completed": position + 1,
                "total": total,
                "progress_percent": round(((position + 1) / max(total, 1)) * 100, 1),
            })
    return entries


def _group_trajectory_entries(entries: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        group_key = _trajectory_group_key(entry)
        grouped.setdefault(group_key, []).append(entry)
    return grouped


def _trajectory_group_key(entry: dict[str, Any]) -> str:
    return "::".join([
        str(entry.get("task_key") or "unknown-task"),
        str(entry.get("robot_type") or "unknown-robot"),
        str(entry.get("canonical_mode") or "unknown-mode"),
    ])


def _evaluate_trajectory_entry(
    entry: dict[str, Any] | None,
    groups: dict[str, list[dict[str, Any]]],
    thresholds: dict[str, float],
    *,
    tube_cache: dict[str, tuple[Any, dict[str, Any]]] | None = None,
    alignment_cache: dict[str, tuple[str, list[list[float]], list[list[float]]]] | None = None,
) -> dict[str, Any]:
    if entry is None:
        return unavailable_result(
            "Trajectory sequence unavailable for DTW quality validation",
            {"reason": "entry_unavailable"},
        )

    group_entries = groups.get(_trajectory_group_key(entry), [])
    record_key = str(entry.get("record_key"))
    comparable_entries = [
        item for item in group_entries
        if str(item.get("record_key")) != record_key
    ]
    references = [item for item in comparable_entries if item.get("base_quality_passed")]
    if not references:
        references = comparable_entries[:]
    if not references:
        return inconclusive_result(
            "no_reference",
            "No reference trajectory available for trajectory_dtw",
            {"reason": "no_reference"},
        )
    minimum_reference_count = int(thresholds["trajectory_dtw_min_reference_count"])
    if len(references) < minimum_reference_count:
        return inconclusive_result(
            "insufficient_references",
            "Insufficient external references for trajectory_dtw",
            {
                "reason": "insufficient_references",
                "reference_count": len(references),
                "minimum_reference_count": minimum_reference_count,
            },
        )

    tube, dtw_config = _build_or_get_reference_tube(
        references,
        entry,
        thresholds,
        tube_cache,
        alignment_cache,
    )
    if tube is None:
        return inconclusive_result(
            "no_reference",
            "No comparable reference trajectory available for trajectory_dtw",
            {"reason": "no_comparable_reference"},
        )

    return ReferenceTubeEvaluator(
        tube,
        thresholds=thresholds,
        dtw_config=dtw_config,
    ).evaluate(entry)


def _build_or_get_reference_tube(
    references: list[dict[str, Any]],
    entry: dict[str, Any],
    thresholds: dict[str, float],
    tube_cache: dict[str, tuple[Any, dict[str, Any]]] | None,
    alignment_cache: dict[str, tuple[str, list[list[float]], list[list[float]]]] | None,
) -> tuple[Any, dict[str, Any]]:
    dtw_config = propagation_dtw_config(references[0], entry)
    cache_key = _reference_cache_key(references)
    if tube_cache is not None and cache_key in tube_cache:
        return tube_cache[cache_key]
    tube = ReferenceTubeBuilder(
        thresholds=thresholds,
        dtw_config=dtw_config,
        alignment_cache=alignment_cache,
    ).build(references)
    cached = (tube, dtw_config)
    if tube_cache is not None:
        tube_cache[cache_key] = cached
    return cached


def _reference_cache_key(references: list[dict[str, Any]]) -> str:
    return "\x1f".join(sorted(str(item.get("record_key", "")) for item in references))
