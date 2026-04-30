from __future__ import annotations

from roboclaw.data.curation.quality_results import aggregate_quality_results


def test_quality_aggregation_adds_decision_labels_and_training_weights() -> None:
    result = aggregate_quality_results(
        [
            {"episode_index": 0, "passed": True, "score": 90.0, "task_confidence": 0.8},
            {"episode_index": 1, "passed": True, "score": 90.0},
            {"episode_index": 2, "passed": True, "score": 70.0},
            {"episode_index": 3, "passed": True, "score": 40.0},
            {"episode_index": 4, "passed": False, "score": 99.0},
        ],
        ["metadata"],
        passed_count=4,
        failed_count=1,
        total=5,
    )

    labels = [episode["decision_label"] for episode in result["episodes"]]
    weights = [episode["training_weight"] for episode in result["episodes"]]

    assert labels == ["accept", "review", "review", "low_weight", "reject"]
    assert weights == [1.0, 0.5, 0.5, 0.2, 0.0]
    assert result["decision_counts"] == {
        "accept": 1,
        "review": 2,
        "low_weight": 1,
        "reject": 1,
    }
    assert result["training_weight_sum"] == 2.2
