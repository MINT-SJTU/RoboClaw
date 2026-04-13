from __future__ import annotations

import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")

from roboclaw.embodied.curation.validators import validate_action


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
