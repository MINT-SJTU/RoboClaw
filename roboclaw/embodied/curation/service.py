"""Curation service — orchestrates the 3-stage quality/prototype/annotation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from loguru import logger


class _CancelToken(Protocol):
    @property
    def is_cancelled(self) -> bool: ...

from .canonical import build_canonical_trajectory
from .clustering import discover_prototype_clusters, refine_clusters_with_dba
from .dtw import resolve_dtw_configuration
from .exports import save_working_quality_parquet
from .features import (
    build_episode_sequence,
    build_joint_trajectory_payload,
    extract_action_names,
    extract_state_names,
    normalize_joint_names,
    resolve_action_vector,
    resolve_state_vector,
)
from .propagation import (
    build_confidence_payload,
    derive_quality_tags,
    detect_grasp_place_events,
    propagate_annotation_spans,
)
from .state import (
    is_stage_pause_requested,
    load_annotations,
    load_dataset_info,
    load_propagation_results,
    load_prototype_results,
    load_quality_results,
    load_workflow_state,
    save_annotations,
    save_propagation_results,
    save_prototype_results,
    save_quality_results,
    save_workflow_state,
    set_stage_pause_requested,
)
from .validators import VALIDATOR_REGISTRY, load_episode_data, run_quality_validators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_load_info = load_dataset_info


def _episode_range(info: dict[str, Any]) -> list[int]:
    total = info.get("total_episodes", 0)
    return list(range(total))


def _set_stage_status(
    dataset_path: Path,
    stage_key: str,
    status: str,
) -> dict[str, Any]:
    state = load_workflow_state(dataset_path)
    state["stages"][stage_key]["status"] = status
    save_workflow_state(dataset_path, state)
    return state


def _update_stage_summary(
    dataset_path: Path,
    stage_key: str,
    summary: dict[str, Any],
    *,
    status: str = "completed",
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"][stage_key]
    stage["status"] = status
    stage["summary"] = summary
    save_workflow_state(dataset_path, state)


def _configure_quality_stage(
    dataset_path: Path,
    *,
    status: str,
    selected_validators: list[str],
) -> None:
    state = load_workflow_state(dataset_path)
    stage = state["stages"]["quality_validation"]
    stage["status"] = status
    stage["selected_validators"] = list(selected_validators)
    stage["pause_requested"] = False
    if status == "running":
        stage["summary"] = None
    save_workflow_state(dataset_path, state)


def _load_episode_duration(dataset_path: Path, episode_index: int) -> float:
    """Return episode duration in seconds from parquet timestamps."""
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


# ---------------------------------------------------------------------------
# CurationService
# ---------------------------------------------------------------------------


class CurationService:
    """Orchestrates the 3-stage curation pipeline for a single dataset."""

    def __init__(self, dataset_path: Path, dataset_name: str | None = None):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name or dataset_path.name

    # ------------------------------------------------------------------
    # Stage 1: Quality validation
    # ------------------------------------------------------------------

    def run_quality_batch(
        self,
        selected_validators: list[str],
        episode_indices: list[int] | None = None,
        threshold_overrides: dict[str, float] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        resume_existing: bool = False,
        cancel_token: _CancelToken | None = None,
    ) -> dict[str, Any]:
        """Run quality validation across episodes.

        Updates workflow state to running/completed and persists results.
        """
        _configure_quality_stage(
            self.dataset_path,
            status="running",
            selected_validators=selected_validators,
        )
        logger.info("Quality batch started for {}", self.dataset_path.name)

        info = _load_info(self.dataset_path)
        indices = episode_indices or _episode_range(info)
        per_episode: list[dict[str, Any]] = []
        passed_count = 0
        failed_count = 0
        total = len(indices)

        initial_completed = 0

        if resume_existing:
            existing = load_quality_results(self.dataset_path) or {}
            existing_episodes = existing.get("episodes", [])
            if isinstance(existing_episodes, list):
                per_episode = list(existing_episodes)
            initial_completed = len(per_episode)
            passed_count = sum(1 for episode in per_episode if episode.get("passed"))
            failed_count = max(len(per_episode) - passed_count, 0)
            existing_total = existing.get("total")
            try:
                total = int(existing_total)
            except (TypeError, ValueError):
                total = len(per_episode) + len(indices)

        def finalize_quality_run(stage_status: str) -> dict[str, Any]:
            aggregated = _aggregate_quality_results(
                per_episode,
                selected_validators,
                passed_count,
                failed_count,
                total,
                threshold_overrides,
            )
            save_quality_results(self.dataset_path, aggregated)

            parquet_path = None
            try:
                parquet_info = save_working_quality_parquet(self.dataset_name, self.dataset_path)
                parquet_path = parquet_info["path"]
            except Exception:
                logger.exception(
                    "Failed to write working quality parquet for {}",
                    self.dataset_path.name,
                )

            summary = {
                "total": total,
                "completed": len(per_episode),
                "remaining": max(total - len(per_episode), 0),
                "passed": passed_count,
                "failed": failed_count,
                "overall_score": aggregated["overall_score"],
                "progress_percent": round((len(per_episode) / max(total, 1)) * 100, 1),
                "quality_parquet_path": parquet_path,
            }
            _update_stage_summary(
                self.dataset_path,
                "quality_validation",
                summary,
                status=stage_status,
            )
            set_stage_pause_requested(self.dataset_path, "quality_validation", False)
            if stage_status == "paused":
                logger.info(
                    "Quality batch paused after {}/{} episodes",
                    len(per_episode),
                    total,
                )
            else:
                logger.info(
                    "Quality batch completed: {}/{} passed (mean score {:.1f})",
                    passed_count,
                    total,
                    aggregated["overall_score"],
                )
            return aggregated

        for position, ep_idx in enumerate(indices):
            if cancel_token is not None and cancel_token.is_cancelled:
                return finalize_quality_run("paused")
            if is_stage_pause_requested(self.dataset_path, "quality_validation"):
                return finalize_quality_run("paused")
            logger.info("Validating episode {}/{}", initial_completed + position + 1, total)
            result = run_quality_validators(
                self.dataset_path,
                ep_idx,
                selected_validators=selected_validators,
                threshold_overrides=threshold_overrides,
            )
            entry = {
                "episode_index": ep_idx,
                "passed": result["passed"],
                "score": result["score"],
                "validators": result["validators"],
                "issues": result["issues"],
            }
            per_episode.append(entry)
            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1

            save_quality_results(
                self.dataset_path,
                _aggregate_quality_results(
                    per_episode, selected_validators, passed_count, failed_count, total, threshold_overrides,
                ),
            )

            if is_stage_pause_requested(self.dataset_path, "quality_validation"):
                return finalize_quality_run("paused")

            if progress_callback is not None:
                progress_callback({
                    "phase": "quality_validation",
                    "episode_index": ep_idx,
                    "completed": initial_completed + position + 1,
                    "total": total,
                    "progress_percent": round(
                        ((initial_completed + position + 1) / max(total, 1)) * 100,
                        1,
                    ),
                })

        return finalize_quality_run("completed")

    # ------------------------------------------------------------------
    # Stage 2: Prototype discovery
    # ------------------------------------------------------------------

    def run_prototype_discovery(
        self,
        cluster_count: int | None = None,
        candidate_limit: int = 50,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: _CancelToken | None = None,
    ) -> dict[str, Any]:
        """Run DTW + k-medoids prototype discovery on quality-passed episodes."""
        _set_stage_status(self.dataset_path, "prototype_discovery", "running")
        logger.info("Prototype discovery started for {}", self.dataset_path.name)

        passed_episodes = _collect_passed_episodes(self.dataset_path)
        if not passed_episodes:
            return _finish_prototype_empty(self.dataset_path)

        candidates = passed_episodes[:candidate_limit]
        entries = _build_canonical_entries(
            self.dataset_path, candidates, progress_callback, cancel_token
        )
        if not entries:
            return _finish_prototype_empty(self.dataset_path)

        clustering = discover_prototype_clusters(
            entries,
            cluster_count=cluster_count,
            progress_callback=progress_callback,
        )
        refined = refine_clusters_with_dba(
            entries,
            clusters=clustering.get("clusters", []),
            progress_callback=progress_callback,
        )

        results = {
            "clustering": clustering,
            "refinement": refined,
            "candidate_count": len(candidates),
            "entry_count": len(entries),
            "cluster_count": refined.get("cluster_count", clustering.get("cluster_count", 0)),
        }
        save_prototype_results(self.dataset_path, results)
        _update_stage_summary(
            self.dataset_path,
            "prototype_discovery",
            {
                "candidate_count": len(candidates),
                "entry_count": len(entries),
                "cluster_count": results["cluster_count"],
            },
        )
        logger.info(
            "Prototype discovery completed: {} entries, {} clusters",
            len(entries), results["cluster_count"],
        )
        return results

    # ------------------------------------------------------------------
    # Stage 3: Semantic propagation
    # ------------------------------------------------------------------

    def run_semantic_propagation(
        self,
        source_episode_index: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: _CancelToken | None = None,
    ) -> dict[str, Any]:
        """Propagate annotations from source episode to cluster members."""
        _set_stage_status(self.dataset_path, "annotation", "running")
        logger.info(
            "Semantic propagation started from episode {} for {}",
            source_episode_index, self.dataset_path.name,
        )

        source_annotations = load_annotations(self.dataset_path, source_episode_index)
        if source_annotations is None:
            return _finish_propagation_empty(self.dataset_path, source_episode_index)

        spans = source_annotations.get("annotations", [])
        if not spans:
            return _finish_propagation_empty(self.dataset_path, source_episode_index)

        source_duration = _load_episode_duration(self.dataset_path, source_episode_index)
        prototype_results = load_prototype_results(self.dataset_path)
        targets = _collect_propagation_targets(
            prototype_results, source_episode_index,
        )

        propagated: list[dict[str, Any]] = []
        persisted_annotation_targets: set[int] = {source_episode_index}
        total = len(targets)
        for position, target in enumerate(targets):
            if cancel_token is not None and cancel_token.is_cancelled:
                break
            result, persisted = _propagate_single_target(
                self.dataset_path,
                target,
                spans,
                source_duration,
                source_annotations,
                source_episode_index,
            )
            propagated.append(result)
            if persisted:
                persisted_annotation_targets.add(target["episode_index"])
            if progress_callback is not None:
                progress_callback({
                    "phase": "semantic_propagation",
                    "completed": position + 1,
                    "total": total,
                    "progress_percent": round(((position + 1) / max(total, 1)) * 100, 1),
                })

        results = {
            "source_episode_index": source_episode_index,
            "target_count": len(propagated),
            "propagated": propagated,
        }
        save_propagation_results(self.dataset_path, results)
        state = load_workflow_state(self.dataset_path)
        annotation_stage = state["stages"]["annotation"]
        existing_targets = {
            int(value)
            for value in annotation_stage.get("annotated_episodes", [])
            if isinstance(value, int) or str(value).isdigit()
        }
        annotation_stage["annotated_episodes"] = sorted(existing_targets | persisted_annotation_targets)
        save_workflow_state(self.dataset_path, state)
        _update_stage_summary(
            self.dataset_path,
            "annotation",
            {
                "source_episode_index": source_episode_index,
                "target_count": len(propagated),
                "annotated_count": len(annotation_stage["annotated_episodes"]),
            },
        )
        logger.info(
            "Semantic propagation completed: {} targets from episode {}",
            len(propagated), source_episode_index,
        )
        return results


# ---------------------------------------------------------------------------
# Quality aggregation
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


def _build_canonical_entries(
    dataset_path: Path,
    episode_indices: list[int],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_token: _CancelToken | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total = len(episode_indices)
    for position, ep_idx in enumerate(episode_indices):
        if cancel_token is not None and cancel_token.is_cancelled:
            break
        logger.info("Building canonical trajectory for episode {}/{}", position + 1, total)
        data = load_episode_data(dataset_path, ep_idx)
        rows = data["rows"]
        if not rows:
            continue

        joint_traj = build_joint_trajectory_payload(
            rows,
            _extract_action_names(data["info"]),
            _extract_state_names(data["info"]),
        )
        canonical = build_canonical_trajectory(rows, joint_traj)
        entries.append({
            "record_key": str(ep_idx),
            "episode_index": ep_idx,
            "sequence": canonical.sequence,
            "feature_vector": canonical.feature_vector,
            "canonical_mode": canonical.mode,
            "canonical_groups": canonical.groups,
            "quality": _episode_quality_summary(dataset_path, ep_idx),
        })

        if progress_callback is not None:
            progress_callback({
                "phase": "building_canonical",
                "completed": position + 1,
                "total": total,
                "progress_percent": round(((position + 1) / max(total, 1)) * 100, 1),
            })

    return entries


_extract_action_names = extract_action_names
_extract_state_names = extract_state_names


def _episode_quality_summary(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    quality = load_quality_results(dataset_path)
    if quality is None:
        return {}
    for ep in quality.get("episodes", []):
        if ep.get("episode_index") == episode_index:
            return {"score": ep.get("score", 0), "passed": ep.get("passed", False)}
    return {}


def _finish_prototype_empty(dataset_path: Path) -> dict[str, Any]:
    results: dict[str, Any] = {
        "clustering": {},
        "refinement": {},
        "candidate_count": 0,
        "entry_count": 0,
        "cluster_count": 0,
    }
    save_prototype_results(dataset_path, results)
    _update_stage_summary(
        dataset_path,
        "prototype_discovery",
        {"candidate_count": 0, "entry_count": 0, "cluster_count": 0},
    )
    logger.warning("Prototype discovery: no passed episodes found")
    return results


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------


def _propagate_single_target(
    dataset_path: Path,
    target: dict[str, Any],
    spans: list[dict[str, Any]],
    source_duration: float,
    source_annotations: dict[str, Any],
    source_episode_index: int,
) -> tuple[dict[str, Any], bool]:
    target_idx = target["episode_index"]
    target_duration = _load_episode_duration(dataset_path, target_idx)
    target_spans = propagate_annotation_spans(
        spans,
        source_duration=source_duration,
        target_duration=target_duration,
        target_record_key=str(target_idx),
        prototype_score=target.get("prototype_score", 0.0),
    )
    result = {
        "episode_index": target_idx,
        "spans": target_spans,
        "prototype_score": target.get("prototype_score", 0.0),
    }
    existing = load_annotations(dataset_path, target_idx) or {}
    existing_annotations = existing.get("annotations", []) or []
    has_manual = any(
        isinstance(span, dict) and span.get("source") == "user"
        for span in existing_annotations
    )
    if has_manual:
        return result, False
    save_annotations(
        dataset_path,
        target_idx,
        {
            "episode_index": target_idx,
            "task_context": {
                **(source_annotations.get("task_context", {}) or {}),
                "source_episode_index": source_episode_index,
                "source": "propagation",
            },
            "annotations": target_spans,
        },
    )
    return result, True


def _collect_propagation_targets(
    prototype_results: dict[str, Any] | None,
    source_episode_index: int,
) -> list[dict[str, Any]]:
    """Find cluster members sharing a cluster with the source episode."""
    if prototype_results is None:
        return []

    refinement = prototype_results.get("refinement", {})
    clusters = refinement.get("clusters", [])
    if not clusters:
        clusters = prototype_results.get("clustering", {}).get("clusters", [])

    targets: list[dict[str, Any]] = []
    source_key = str(source_episode_index)
    for cluster in clusters:
        member_keys = [str(m.get("record_key", "")) for m in cluster.get("members", [])]
        if source_key not in member_keys:
            continue
        for member in cluster.get("members", []):
            member_key = str(member.get("record_key", ""))
            if member_key == source_key:
                continue
            targets.append({
                "episode_index": int(member_key),
                "prototype_score": 1.0 - float(member.get("distance_to_barycenter", member.get("distance_to_prototype", 0.0))),
            })
    return targets


def _finish_propagation_empty(
    dataset_path: Path,
    source_episode_index: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "source_episode_index": source_episode_index,
        "target_count": 0,
        "propagated": [],
    }
    save_propagation_results(dataset_path, results)
    _update_stage_summary(
        dataset_path,
        "annotation",
        {"source_episode_index": source_episode_index, "target_count": 0},
    )
    logger.warning("Semantic propagation: no annotations found for episode {}", source_episode_index)
    return results
