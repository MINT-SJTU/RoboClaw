from __future__ import annotations

import math
import os

import pytest

from roboclaw.data.curation.dtw import (
    CARTESIAN_20D_GROUP_WEIGHTS,
    dtw_alignment,
    dtw_distance,
)
from roboclaw.data.curation.propagation import propagate_annotation_spans
from roboclaw.data.curation.validators import safe_float, validate_action


def test_safe_float_rejects_non_finite_values() -> None:
    assert safe_float(float("nan")) is None
    assert safe_float(float("inf")) is None
    assert safe_float("-inf") is None


def test_empty_dtw_is_not_perfect_match() -> None:
    assert math.isinf(dtw_distance([], [[1.0]]))
    distance, path = dtw_alignment([], [[1.0]])
    assert math.isinf(distance)
    assert path == []


def test_cuda_dtw_alignment_matches_cpu_when_available() -> None:
    left = [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]
    right = [[0.0, 0.0], [1.0, 0.5], [1.0, 0.5], [2.0, 1.0]]

    os.environ["ROBOCLAW_DTW_BACKEND"] = "cpu"
    cpu_distance, cpu_path = dtw_alignment(left, right)
    os.environ.pop("ROBOCLAW_DTW_BACKEND", None)
    gpu_distance, gpu_path = dtw_alignment(left, right)

    assert gpu_distance == pytest.approx(cpu_distance, abs=1e-5)
    assert gpu_path[0] == cpu_path[0]
    assert gpu_path[-1] == cpu_path[-1]


def test_cartesian_20d_group_weights_match_documented_defaults() -> None:
    assert CARTESIAN_20D_GROUP_WEIGHTS["eef_rot6d"] == 0.7
    assert CARTESIAN_20D_GROUP_WEIGHTS["gripper"] == 1.2


def test_propagation_falls_back_to_duration_scaling_for_empty_dtw() -> None:
    propagated = propagate_annotation_spans(
        [{"label": "grasp", "startTime": 1.0, "endTime": 1.5}],
        source_duration=2.0,
        target_duration=4.0,
        target_record_key="target",
        prototype_score=0.8,
        source_sequence=[],
        target_sequence=[[1.0]],
        source_time_axis=[],
        target_time_axis=[0.0],
    )

    assert propagated[0]["source"] == "duration_scaled"
    assert propagated[0]["startTime"] == 2.0


def test_validate_action_accepts_vector_series_without_false_missing_ratio() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "action": [0.0, 0.1, 0.2, 0.0],
            "action.joint_position": [0.0, 0.1, 0.2],
            "observation.state": [0.0, 0.1, 0.2, 0.0],
            "observation.state.joint_position": [0.0, 0.1, 0.2],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 0.2,
            "action": [0.02, 0.14, 0.24, 0.0],
            "action.joint_position": [0.02, 0.14, 0.24],
            "observation.state": [0.01, 0.13, 0.23, 0.0],
            "observation.state.joint_position": [0.01, 0.13, 0.23],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 0.4,
            "action": [0.05, 0.18, 0.29, 0.0],
            "action.joint_position": [0.05, 0.18, 0.29],
            "observation.state": [0.04, 0.17, 0.28, 0.0],
            "observation.state.joint_position": [0.04, 0.17, 0.28],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 0.6,
            "action": [0.09, 0.23, 0.35, 0.0],
            "action.joint_position": [0.09, 0.23, 0.35],
            "observation.state": [0.08, 0.22, 0.34, 0.0],
            "observation.state.joint_position": [0.08, 0.22, 0.34],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 0.8,
            "action": [0.14, 0.29, 0.42, 0.0],
            "action.joint_position": [0.14, 0.29, 0.42],
            "observation.state": [0.13, 0.28, 0.41, 0.0],
            "observation.state.joint_position": [0.13, 0.28, 0.41],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 1.0,
            "action": [0.2, 0.36, 0.5, 0.0],
            "action.joint_position": [0.2, 0.36, 0.5],
            "observation.state": [0.19, 0.35, 0.49, 0.0],
            "observation.state.joint_position": [0.19, 0.35, 0.49],
            "action.gripper_position": 0.0,
        },
        {
            "timestamp": 1.2,
            "action": [0.27, 0.44, 0.59, 0.0],
            "action.joint_position": [0.27, 0.44, 0.59],
            "observation.state": [0.26, 0.43, 0.58, 0.0],
            "observation.state.joint_position": [0.26, 0.43, 0.58],
            "action.gripper_position": 0.0,
        },
    ]
    data = {
        "rows": rows,
        "info": {
            "features": {
                "action": {"names": {"axes": ["joint_0", "joint_1", "joint_2", "gripper"]}},
                "action.joint_position": {"names": {"axes": ["joint_0", "joint_1", "joint_2"]}},
                "observation.state": {"names": {"axes": ["joint_0", "joint_1", "joint_2", "gripper"]}},
                "observation.state.joint_position": {
                    "names": {"axes": ["joint_0", "joint_1", "joint_2"]},
                },
            },
        },
    }

    result = validate_action(data)
    issues_by_name = {issue["check_name"]: issue for issue in result["issues"]}
    nan_issue = issues_by_name["nan_ratio"]

    assert nan_issue["passed"] is True
    assert nan_issue["value"]["nan_ratio"] == 0.0
    assert issues_by_name["duration"]["passed"] is True
    assert result["passed"] is True


def test_validate_action_counts_nan_values_as_missing() -> None:
    rows = [
        {"timestamp": 0.0, "action": [0.0, 0.1], "observation.state": [0.0, 0.1]},
        {"timestamp": 0.5, "action": [float("nan"), 0.2], "observation.state": [0.1, 0.2]},
        {"timestamp": 1.0, "action": [0.2, 0.3], "observation.state": [0.2, 0.3]},
    ]
    result = validate_action(
        {"rows": rows, "info": {"features": {"action": {"names": ["j0", "j1"]}}}},
        threshold_overrides={"action_max_nan_ratio": 0.01},
    )

    issues_by_name = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues_by_name["nan_ratio"]["passed"] is False
    assert issues_by_name["nan_ratio"]["value"]["nan_ratio"] > 0.0
    assert result["passed"] is False


def test_validate_action_uses_named_key_joints_instead_of_first_six_columns() -> None:
    rows = []
    for index in range(8):
        rows.append({
            "timestamp": float(index),
            "action": [
                float(index),
                float(index),
                float(index),
                float(index),
                float(index),
                float(index),
                0.0,
            ],
        })

    result = validate_action(
        {
            "rows": rows,
            "info": {
                "features": {
                    "action": {
                        "names": {
                            "axes": [
                                "aux_0",
                                "aux_1",
                                "aux_2",
                                "aux_3",
                                "aux_4",
                                "aux_5",
                                "wrist_joint",
                            ],
                        },
                    },
                },
            },
        },
        threshold_overrides={
            "action_max_all_static_s": 10.0,
            "action_max_key_static_s": 3.0,
            "action_max_velocity_rad_s": 10.0,
            "action_min_duration_s": 1.0,
        },
    )

    issues_by_name = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues_by_name["all_static_duration"]["passed"] is True
    assert issues_by_name["key_static_duration"]["passed"] is False
    assert issues_by_name["key_static_duration"]["value"]["key_static_duration_s"] > 3.0
