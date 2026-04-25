from __future__ import annotations

from roboclaw.data.curation.clustering import discover_prototype_clusters
from roboclaw.data.curation.propagation import propagate_annotation_spans
from roboclaw.data.curation.prototypes import discover_grouped_prototypes


def test_auto_cluster_selection_does_not_prefer_all_singletons() -> None:
    entries = [
        {"record_key": str(index), "sequence": [[float(index)]], "canonical_groups": {}}
        for index in range(4)
    ]

    result = discover_prototype_clusters(entries, cluster_count=None)

    assert result["cluster_count"] < len(entries)


def test_grouped_prototypes_keep_different_tasks_apart() -> None:
    entries = [
        {
            "record_key": "pick-0",
            "sequence": [[0.0], [0.1]],
            "task_key": "pick",
            "robot_type": "arm",
            "canonical_mode": "joint_canonical",
            "canonical_groups": {},
        },
        {
            "record_key": "pick-1",
            "sequence": [[0.0], [0.2]],
            "task_key": "pick",
            "robot_type": "arm",
            "canonical_mode": "joint_canonical",
            "canonical_groups": {},
        },
        {
            "record_key": "place-0",
            "sequence": [[10.0], [10.1]],
            "task_key": "place",
            "robot_type": "arm",
            "canonical_mode": "joint_canonical",
            "canonical_groups": {},
        },
        {
            "record_key": "place-1",
            "sequence": [[10.0], [10.2]],
            "task_key": "place",
            "robot_type": "arm",
            "canonical_mode": "joint_canonical",
            "canonical_groups": {},
        },
    ]

    result = discover_grouped_prototypes(entries, cluster_count=1)

    assert result["group_count"] == 2
    assert result["refinement"]["cluster_count"] == 2
    for cluster in result["refinement"]["clusters"]:
        task_keys = {member["record_key"].split("-")[0] for member in cluster["members"]}
        assert len(task_keys) == 1


def test_grouped_prototypes_treat_fixed_cluster_count_as_global_budget() -> None:
    entries = []
    for task in ("pick", "place"):
        for index in range(3):
            base = 0.0 if task == "pick" else 10.0
            entries.append({
                "record_key": f"{task}-{index}",
                "sequence": [[base + index * 0.1], [base + index * 0.2]],
                "task_key": task,
                "robot_type": "arm",
                "canonical_mode": "joint_canonical",
                "canonical_groups": {},
            })

    result = discover_grouped_prototypes(entries, cluster_count=3)

    assert result["group_count"] == 2
    assert result["refinement"]["cluster_count"] == 3


def test_propagation_uses_dtw_time_mapping_instead_of_duration_scaling() -> None:
    spans = [{"label": "grasp", "startTime": 1.0, "endTime": 1.0}]
    source_sequence = [[0.0], [1.0], [2.0]]
    target_sequence = [[0.0], [1.0], [1.0], [1.0], [2.0]]
    source_time_axis = [0.0, 1.0, 2.0]
    target_time_axis = [0.0, 0.5, 1.0, 1.5, 2.0]

    propagated = propagate_annotation_spans(
        spans,
        source_duration=2.0,
        target_duration=2.0,
        target_record_key="target",
        prototype_score=1.2,
        source_sequence=source_sequence,
        target_sequence=target_sequence,
        source_time_axis=source_time_axis,
        target_time_axis=target_time_axis,
    )

    assert propagated[0]["source"] == "dtw_propagated"
    assert propagated[0]["startTime"] == 1.0
    assert propagated[0]["prototype_score"] == 1.0
