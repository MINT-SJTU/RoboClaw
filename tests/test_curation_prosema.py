from __future__ import annotations

from roboclaw.data.curation.clustering import discover_prototype_clusters, refine_clusters_with_dba
from roboclaw.data.curation.propagation import propagate_annotation_spans
from roboclaw.data.curation.propagation_history import collect_propagation_targets
from roboclaw.data.curation.prototypes import discover_grouped_prototypes


def test_auto_cluster_selection_does_not_prefer_all_singletons() -> None:
    entries = [
        {"record_key": str(index), "sequence": [[float(index)]], "canonical_groups": {}}
        for index in range(4)
    ]

    result = discover_prototype_clusters(entries, cluster_count=None)

    assert result["cluster_count"] < len(entries)


def test_prototype_discovery_skips_empty_sequences() -> None:
    entries = [
        {"record_key": "empty", "sequence": [], "canonical_groups": {}},
        {"record_key": "valid-0", "sequence": [[0.0], [0.1]], "canonical_groups": {}},
        {"record_key": "valid-1", "sequence": [[0.0], [0.2]], "canonical_groups": {}},
    ]

    result = discover_prototype_clusters(entries, cluster_count=1)
    cluster_members = {
        member["record_key"]
        for cluster in result["clusters"]
        for member in cluster["members"]
    }

    assert "empty" not in cluster_members
    assert result["distance_matrix"].keys() == {"valid-0", "valid-1"}


def test_dba_refinement_skips_empty_sequences_without_zero_fallback() -> None:
    entries = [
        {"record_key": "empty", "sequence": [], "canonical_groups": {}},
        {"record_key": "valid", "sequence": [[0.0], [0.1]], "canonical_groups": {}},
    ]
    clusters = [
        {
            "prototype_record_key": "empty",
            "members": [
                {"record_key": "empty", "distance_to_prototype": 0.0},
                {"record_key": "valid", "distance_to_prototype": 0.0},
            ],
        }
    ]

    result = refine_clusters_with_dba(entries, clusters=clusters)

    assert result["cluster_count"] == 1
    assert result["anchor_record_keys"] == ["valid"]
    assert [member["record_key"] for member in result["clusters"][0]["members"]] == ["valid"]


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


def test_propagation_targets_normalize_prototype_score_within_cluster() -> None:
    prototype_results = {
        "refinement": {
            "clusters": [
                {
                    "members": [
                        {"record_key": "0", "distance_to_barycenter": 0.0},
                        {"record_key": "1", "distance_to_barycenter": 2.0},
                        {"record_key": "2", "distance_to_barycenter": 4.0},
                    ],
                },
            ],
        },
    }

    targets = collect_propagation_targets(prototype_results, source_episode_index=0)
    scores = {target["episode_index"]: target["prototype_score"] for target in targets}

    assert scores == {1: 0.5, 2: 0.0}


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
