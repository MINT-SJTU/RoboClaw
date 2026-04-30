from __future__ import annotations

from roboclaw.data.curation.clustering import _auto_select_cluster_count


class _FakeRunner:
    def __init__(self, scores: dict[int, float], member_counts: dict[int, list[int]]) -> None:
        self._scores = scores
        self._member_counts = member_counts
        self.final_k: int | None = None

    def run(self, requested_k: int, *, emit_progress: bool) -> dict[str, object]:
        if emit_progress:
            self.final_k = requested_k
        return {
            "cluster_count": requested_k,
            "clusters": [
                {"member_count": count, "members": [{"record_key": f"{requested_k}-{index}"}]}
                for index, count in enumerate(self._member_counts[requested_k])
            ],
        }

    def average_silhouette(self, clusters: list[dict[str, object]]) -> float:
        cluster_count = len(clusters)
        return self._scores[cluster_count]


def test_auto_select_cluster_count_prefers_fewer_stable_clusters() -> None:
    runner = _FakeRunner(
        scores={
            1: 0.0,
            2: 0.31,
            3: 0.34,
            4: 0.36,
        },
        member_counts={
            1: [4],
            2: [2, 2],
            3: [1, 1, 2],
            4: [1, 1, 1, 1],
        },
    )

    result = _auto_select_cluster_count([{"record_key": str(index)} for index in range(4)], runner)

    assert result["cluster_count"] == 2
    assert runner.final_k == 2
    diagnostics = result["selection_diagnostics"]
    assert diagnostics["selected_k"] == 2
    assert diagnostics["best_k"] == 2
    assert diagnostics["rejected_singleton_heavy_count"] == 2
    assert diagnostics["evaluated"][1]["selected"] is True


def test_auto_select_cluster_count_rejects_singleton_heavy_high_k() -> None:
    runner = _FakeRunner(
        scores={
            1: 0.0,
            2: 0.3160,
            3: 0.3204,
            4: 0.3332,
            5: 0.3274,
            6: 0.2905,
            7: 0.3501,
            8: 0.2790,
            9: 0.3111,
            10: 0.3675,
            11: 0.4133,
            12: 0.4380,
            13: 0.5059,
            14: 0.5875,
            15: 0.6382,
        },
        member_counts={
            1: [21],
            2: [15, 6],
            3: [15, 5, 1],
            4: [15, 1, 1, 4],
            5: [9, 1, 1, 4, 6],
            6: [9, 1, 1, 2, 5, 3],
            7: [9, 1, 1, 2, 4, 3, 1],
            8: [4, 1, 1, 2, 4, 3, 1, 5],
            9: [1, 1, 1, 2, 4, 3, 1, 4, 4],
            10: [1, 1, 1, 2, 4, 2, 1, 4, 4, 1],
            11: [1, 1, 1, 2, 4, 2, 1, 4, 3, 1, 1],
            12: [2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 5],
            13: [2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 5, 1],
            14: [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 5, 1, 1],
            15: [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1],
        },
    )

    result = _auto_select_cluster_count([{"record_key": str(index)} for index in range(21)], runner)

    assert result["cluster_count"] == 2
    assert runner.final_k == 2
    diagnostics = result["selection_diagnostics"]
    assert diagnostics["selected_k"] == 2
    assert diagnostics["best_k"] == 2
    assert diagnostics["candidate_pool_count"] == 2
    assert diagnostics["rejected_singleton_heavy_count"] == 13
