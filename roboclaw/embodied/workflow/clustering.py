from __future__ import annotations

import math
from typing import Any, Callable

from .dtw import (
    average_vectors,
    build_distance_matrix_with_progress,
    dtw_alignment,
    resolve_dtw_configuration,
    vector_distance,
)
from .features import mean

# ---------------------------------------------------------------------------
# K-medoids clustering with automatic cluster count selection
# ---------------------------------------------------------------------------


def discover_prototype_clusters(
    entries: list[dict[str, Any]],
    *,
    cluster_count: int | None,
    max_iterations: int = 8,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not entries:
        return _empty_clustering_result(max_iterations)

    distance_matrix, total_pairs = _build_matrix_with_reporting(
        entries, progress_callback,
    )
    runner = _KMedoidsRunner(entries, distance_matrix, max_iterations, progress_callback)

    if cluster_count is not None:
        clustering = runner.run(cluster_count, emit_progress=True)
        return {
            **clustering,
            "distance_matrix": distance_matrix,
            "distance_pair_count": total_pairs,
            "selection_mode": "fixed",
        }

    best_clustering = _auto_select_cluster_count(entries, runner)
    return {
        **best_clustering,
        "distance_matrix": distance_matrix,
        "distance_pair_count": total_pairs,
        "selection_mode": "auto",
    }


def _empty_clustering_result(max_iterations: int) -> dict[str, Any]:
    return {
        "cluster_count": 0,
        "distance_matrix": {},
        "clusters": [],
        "prototype_record_keys": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "converged": True,
        "medoid_history": [],
    }


def _build_matrix_with_reporting(
    entries: list[dict[str, Any]],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[dict[str, dict[str, float]], int]:
    def report(completed_pairs: int, total_pairs: int) -> None:
        if progress_callback is None:
            return
        pct = 100.0 if total_pairs <= 0 else round((completed_pairs / total_pairs) * 100, 1)
        progress_callback({
            "phase": "building_dtw_graph",
            "pairs_completed": completed_pairs,
            "pairs_total": total_pairs,
            "progress_percent": pct,
        })

    return build_distance_matrix_with_progress(entries, progress_callback=report)


# ---------------------------------------------------------------------------
# K-medoids runner (encapsulates distance_matrix lookup)
# ---------------------------------------------------------------------------


class _KMedoidsRunner:
    def __init__(
        self,
        entries: list[dict[str, Any]],
        distance_matrix: dict[str, dict[str, float]],
        max_iterations: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._entries = entries
        self._dm = distance_matrix
        self._max_iter = max_iterations
        self._cb = progress_callback

    # -- public interface --

    def run(self, requested_k: int, *, emit_progress: bool) -> dict[str, Any]:
        k = min(max(requested_k, 1), len(self._entries))
        medoids = self._seed_medoids(k)
        medoid_history = [list(medoids)]
        converged = False
        iteration_count = 0

        for iteration_index in range(self._max_iter):
            iteration_count = iteration_index + 1
            cluster_map = self._assign(medoids)
            next_medoids = self._update_medoids(medoids, cluster_map)

            if emit_progress and self._cb is not None:
                self._cb({
                    "phase": "k_medoids",
                    "iteration_count": iteration_count,
                    "max_iterations": self._max_iter,
                    "current_medoids": next_medoids[:],
                    "converged": next_medoids == medoids,
                    "progress_percent": round((iteration_count / max(self._max_iter, 1)) * 100, 1),
                })

            if next_medoids == medoids:
                converged = True
                break
            medoids = next_medoids
            medoid_history.append(list(medoids))

        cluster_map = self._assign(medoids)
        clusters = self._format_clusters(medoids, cluster_map)
        return {
            "cluster_count": k,
            "clusters": clusters,
            "prototype_record_keys": medoids,
            "iteration_count": iteration_count,
            "max_iterations": self._max_iter,
            "converged": converged,
            "medoid_history": medoid_history,
        }

    def average_silhouette(self, clusters: list[dict[str, Any]]) -> float:
        if len(clusters) <= 1:
            return 0.0
        cluster_members = [
            [str(m["record_key"]) for m in c.get("members", [])]
            for c in clusters
        ]
        scores: list[float] = []
        for ci, member_keys in enumerate(cluster_members):
            for mk in member_keys:
                scores.append(self._silhouette_sample(mk, ci, cluster_members))
        return mean(scores)

    # -- internals --

    def _seed_medoids(self, k: int) -> list[str]:
        medoids = [
            min(self._entries, key=lambda e: self._median_distance(e["record_key"]))["record_key"]
        ]
        while len(medoids) < k:
            candidates = [e for e in self._entries if e["record_key"] not in medoids]
            farthest = max(
                candidates,
                key=lambda e: min(self._dm[e["record_key"]][m] for m in medoids),
            )
            medoids.append(farthest["record_key"])
        return medoids

    def _median_distance(self, key: str) -> float:
        distances = [d for d in self._dm[key].values() if d > 0]
        if not distances:
            return 0.0
        s = sorted(distances)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    def _assign(self, medoids: list[str]) -> dict[str, list[dict[str, Any]]]:
        cluster_map: dict[str, list[dict[str, Any]]] = {m: [] for m in medoids}
        for entry in self._entries:
            nearest = min(medoids, key=lambda m: self._dm[entry["record_key"]][m])
            cluster_map[nearest].append({
                **entry,
                "distance_to_prototype": self._dm[entry["record_key"]][nearest],
            })
        return cluster_map

    def _update_medoids(
        self,
        medoids: list[str],
        cluster_map: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        next_medoids: list[str] = []
        for medoid in medoids:
            members = cluster_map[medoid]
            best = min(
                members,
                key=lambda m: sum(self._dm[m["record_key"]][o["record_key"]] for o in members),
            )
            next_medoids.append(best["record_key"])
        return next_medoids

    def _format_clusters(
        self,
        medoids: list[str],
        cluster_map: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for medoid in medoids:
            members = sorted(cluster_map[medoid], key=lambda m: m["distance_to_prototype"])
            clusters.append({
                "prototype_record_key": medoid,
                "member_count": len(members),
                "average_distance": mean([m["distance_to_prototype"] for m in members]),
                "members": members,
            })
        return clusters

    def _silhouette_sample(
        self,
        member_key: str,
        cluster_index: int,
        cluster_members: list[list[str]],
    ) -> float:
        same = [k for k in cluster_members[cluster_index] if k != member_key]
        intra = mean([float(self._dm[member_key][k]) for k in same]) if same else 0.0
        inter_dists: list[float] = []
        for oi, other_keys in enumerate(cluster_members):
            if oi == cluster_index or not other_keys:
                continue
            inter_dists.append(mean([float(self._dm[member_key][k]) for k in other_keys]))
        nearest_other = min(inter_dists) if inter_dists else 0.0
        denom = max(intra, nearest_other)
        if denom <= 1e-8:
            return 0.0
        return (nearest_other - intra) / denom


def _auto_select_cluster_count(
    entries: list[dict[str, Any]],
    runner: _KMedoidsRunner,
) -> dict[str, Any]:
    best_clustering: dict[str, Any] | None = None
    best_score = float("-inf")
    max_candidate = min(len(entries), max(len(entries) // 5, 15))

    for k in range(1, max_candidate + 1):
        candidate = runner.run(k, emit_progress=False)
        smallest = min(
            (c.get("member_count", 0) for c in candidate["clusters"]),
            default=0,
        )
        if k > 1 and smallest < 1:
            continue
        score = runner.average_silhouette(candidate["clusters"])
        if score > best_score:
            best_score = score
            best_clustering = candidate

    if best_clustering is None:
        return runner.run(1, emit_progress=True)
    return runner.run(best_clustering["cluster_count"], emit_progress=True)


# ---------------------------------------------------------------------------
# Rotation constraint helpers (for DBA barycenter)
# ---------------------------------------------------------------------------


def _normalize_vector(values: list[float], fallback: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-8:
        return fallback[:]
    return [v / norm for v in values]


def _orthogonalize_rot6d(values: list[float]) -> list[float]:
    if len(values) < 6:
        return values[:]

    e1 = _normalize_vector(values[:3], [1.0, 0.0, 0.0])
    raw_second = values[3:6]
    projection = sum(raw_second[i] * e1[i] for i in range(3))
    orthogonal = [raw_second[i] - projection * e1[i] for i in range(3)]

    if math.sqrt(sum(v * v for v in orthogonal)) <= 1e-8:
        orthogonal = _fallback_orthogonal(e1)

    e2 = _normalize_vector(orthogonal, [0.0, 1.0, 0.0])
    return [*e1, *e2]


def _fallback_orthogonal(e1: list[float]) -> list[float]:
    basis_candidates = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    best = min(
        basis_candidates,
        key=lambda c: abs(sum(c[i] * e1[i] for i in range(3))),
    )
    proj = sum(best[i] * e1[i] for i in range(3))
    return [best[i] - proj * e1[i] for i in range(3)]


def _restore_cartesian_rotation_constraints(
    frame: list[float],
    groups: dict[str, list[int]] | None,
) -> list[float]:
    if not groups or "eef_rot6d" not in groups:
        return frame
    rot_indices = groups.get("eef_rot6d", [])
    if len(rot_indices) < 6:
        return frame
    restored = frame[:]
    orthogonalized = _orthogonalize_rot6d([frame[i] for i in rot_indices[:6]])
    for offset, index in enumerate(rot_indices[:6]):
        restored[index] = orthogonalized[offset]
    return restored


# ---------------------------------------------------------------------------
# DBA barycenter computation
# ---------------------------------------------------------------------------


def compute_dba_barycenter(
    sequences: list[list[list[float]]],
    *,
    reference_sequence: list[list[float]] | None = None,
    max_iterations: int = 4,
    groups: dict[str, list[int]] | None = None,
    dtw_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not sequences:
        return {"sequence": [], "iteration_count": 0, "converged": True}

    barycenter = [
        list(step)
        for step in (reference_sequence or sequences[0] or [[0.0]])
    ]
    if not barycenter:
        barycenter = [[0.0]]

    resolved_cfg = dict(dtw_configuration or {})
    converged = False
    iteration_count = 0

    for iteration_index in range(max_iterations):
        iteration_count = iteration_index + 1
        assignments: list[list[list[float]]] = [[] for _ in barycenter]
        _collect_assignments(sequences, barycenter, resolved_cfg, assignments)
        next_barycenter, max_delta = _update_barycenter(barycenter, assignments, groups)
        barycenter = next_barycenter
        if max_delta < 1e-3:
            converged = True
            break

    return {
        "sequence": barycenter,
        "iteration_count": iteration_count,
        "converged": converged,
    }


def _collect_assignments(
    sequences: list[list[list[float]]],
    barycenter: list[list[float]],
    dtw_cfg: dict[str, Any],
    assignments: list[list[list[float]]],
) -> None:
    for sequence in sequences:
        if not sequence:
            continue
        _distance, alignment = dtw_alignment(barycenter, sequence, **dtw_cfg)
        for bc_idx, seq_idx in alignment:
            if bc_idx >= len(assignments) or seq_idx >= len(sequence):
                continue
            assignments[bc_idx].append(sequence[seq_idx])


def _update_barycenter(
    barycenter: list[list[float]],
    assignments: list[list[list[float]]],
    groups: dict[str, list[int]] | None,
) -> tuple[list[list[float]], float]:
    next_barycenter: list[list[float]] = []
    max_step_delta = 0.0

    for index, current_step in enumerate(barycenter):
        averaged_step = _compute_averaged_step(
            index, current_step, barycenter, assignments,
        )
        averaged_step = _restore_cartesian_rotation_constraints(averaged_step, groups)
        max_step_delta = max(max_step_delta, vector_distance(current_step, averaged_step))
        next_barycenter.append(averaged_step)

    return next_barycenter, max_step_delta


def _compute_averaged_step(
    index: int,
    current_step: list[float],
    barycenter: list[list[float]],
    assignments: list[list[list[float]]],
) -> list[float]:
    if assignments[index]:
        return average_vectors(assignments[index])
    if 0 < index < len(barycenter) - 1:
        return average_vectors([barycenter[index - 1], barycenter[index + 1]])
    return current_step[:]


# ---------------------------------------------------------------------------
# Cluster refinement with DBA
# ---------------------------------------------------------------------------


def refine_clusters_with_dba(
    entries: list[dict[str, Any]],
    *,
    clusters: list[dict[str, Any]],
    max_iterations: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not clusters:
        return {
            "clusters": [],
            "cluster_count": 0,
            "anchor_record_keys": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
        }

    entry_lookup = {
        str(entry["record_key"]): entry
        for entry in entries
        if entry.get("record_key") is not None
    }
    refined_clusters: list[dict[str, Any]] = []
    anchor_record_keys: list[str] = []
    observed_iteration_count = 0

    for cluster_index, cluster in enumerate(clusters):
        result = _refine_single_cluster(
            cluster, cluster_index, entry_lookup, max_iterations,
        )
        if result is None:
            continue
        refined, anchor_key, iter_count = result
        observed_iteration_count = max(observed_iteration_count, iter_count)
        anchor_record_keys.append(anchor_key)
        refined_clusters.append(refined)

        if progress_callback is not None:
            progress_callback({
                "phase": "dba_refinement",
                "cluster_index": cluster_index,
                "processed_count": len(refined_clusters),
                "total_count": len(clusters),
                "iteration_count": iter_count,
                "max_iterations": max_iterations,
                "anchor_record_key": anchor_key,
                "progress_percent": round((len(refined_clusters) / max(len(clusters), 1)) * 100.0, 1),
            })

    return {
        "clusters": refined_clusters,
        "cluster_count": len(refined_clusters),
        "anchor_record_keys": anchor_record_keys,
        "iteration_count": observed_iteration_count,
        "max_iterations": max_iterations,
    }


def _refine_single_cluster(
    cluster: dict[str, Any],
    cluster_index: int,
    entry_lookup: dict[str, dict[str, Any]],
    max_iterations: int,
) -> tuple[dict[str, Any], str, int] | None:
    member_records = cluster.get("members", [])
    member_entries = [
        entry_lookup[str(m.get("record_key"))]
        for m in member_records
        if str(m.get("record_key")) in entry_lookup
    ]
    if not member_entries:
        return None

    prototype_key = str(cluster.get("prototype_record_key") or member_entries[0]["record_key"])
    reference = entry_lookup.get(prototype_key, member_entries[0])
    ref_groups = reference.get("canonical_groups") or {}
    dtw_cfg = resolve_dtw_configuration(
        left_mode=reference.get("canonical_mode"),
        right_mode=reference.get("canonical_mode"),
        left_groups=ref_groups,
        right_groups=ref_groups,
    )

    bary = compute_dba_barycenter(
        [e.get("sequence") or [[0.0]] for e in member_entries],
        reference_sequence=reference.get("sequence") or [[0.0]],
        max_iterations=max_iterations,
        groups=ref_groups,
        dtw_configuration=dtw_cfg,
    )
    bary_seq = bary["sequence"]

    summaries = _compute_member_summaries(
        member_entries, member_records, bary_seq, dtw_cfg,
    )
    summaries.sort(key=lambda m: m["distance_to_barycenter"])
    for rank, member in enumerate(summaries, start=1):
        member["assignment_rank"] = rank

    anchor_key = str(summaries[0]["record_key"])
    refined = {
        "cluster_index": int(cluster.get("cluster_index", cluster_index)),
        "prototype_record_key": prototype_key,
        "anchor_record_key": anchor_key,
        "anchor_distance_to_barycenter": summaries[0]["distance_to_barycenter"],
        "member_count": len(summaries),
        "barycenter_length": len(bary_seq),
        "barycenter_iteration_count": bary["iteration_count"],
        "barycenter_converged": bary["converged"],
        "barycenter_sequence": bary_seq,
        "members": summaries,
    }
    return refined, anchor_key, int(bary["iteration_count"])


def _compute_member_summaries(
    member_entries: list[dict[str, Any]],
    member_records: list[dict[str, Any]],
    barycenter_sequence: list[list[float]],
    dtw_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for entry in member_entries:
        rk = str(entry["record_key"])
        dist, _align = dtw_alignment(
            barycenter_sequence,
            entry.get("sequence") or [[0.0]],
            **dtw_cfg,
        )
        proto_member = next(
            (m for m in member_records if str(m.get("record_key")) == rk),
            {},
        )
        summaries.append({
            "record_key": rk,
            "distance_to_prototype": float(proto_member.get("distance_to_prototype", 0.0)),
            "distance_to_barycenter": round(float(dist), 4),
            "quality": proto_member.get("quality", entry.get("quality", {})),
        })
    return summaries
