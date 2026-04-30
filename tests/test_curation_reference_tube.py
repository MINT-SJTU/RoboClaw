from __future__ import annotations

from roboclaw.data.curation import service as curation_service
from roboclaw.data.curation.reference_tube import (
    ReferenceTubeBuilder,
    ReferenceTubeEvaluator,
)
from roboclaw.data.curation.service import _LegacyCurationService
from roboclaw.data.curation.trajectory_quality import _evaluate_trajectory_entry
from roboclaw.data.curation.validators import QUALITY_THRESHOLD_DEFAULTS


def _thresholds(**overrides: float) -> dict[str, float]:
    values = dict(QUALITY_THRESHOLD_DEFAULTS)
    values.update(overrides)
    return values


def _entry(record_key: str, sequence: list[list[float]]) -> dict[str, object]:
    return {
        "record_key": record_key,
        "sequence": sequence,
        "time_axis": [index * 0.1 for index in range(len(sequence))],
    }


def test_reference_tube_accepts_matching_candidate() -> None:
    thresholds = _thresholds(trajectory_dtw_min_segment_s=0.0)
    references = [
        _entry("ref-1", [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        _entry("ref-2", [[0.0, 0.0], [0.11, 0.1], [0.21, 0.2]]),
    ]
    tube = ReferenceTubeBuilder(thresholds=thresholds).build(references)
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
    )

    assert result["passed"] is True


def test_reference_tube_flags_deviation() -> None:
    thresholds = _thresholds(trajectory_dtw_min_segment_s=0.0)
    references = [
        _entry("ref-1", [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        _entry("ref-2", [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
    ]
    tube = ReferenceTubeBuilder(thresholds=thresholds).build(references)
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0, 0.0], [2.0, 2.0], [0.2, 0.2]]),
    )

    assert result["passed"] is False
    assert any(issue["check_name"] == "dtw_deviation" for issue in result["issues"])


def test_reference_tube_flags_hesitation() -> None:
    thresholds = _thresholds(
        trajectory_dtw_min_segment_s=0.0,
        trajectory_dtw_velocity_floor=0.001,
        trajectory_dtw_hesitation_multiplier=1.0,
        trajectory_dtw_deviation_multiplier=100.0,
    )
    references = [
        _entry("ref-1", [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
        _entry("ref-2", [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
    ]
    tube = ReferenceTubeBuilder(thresholds=thresholds).build(references)
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0, 0.0, 0.0], [0.7, -0.7, 0.7], [0.2, 0.2, 0.2]]),
    )

    assert result["passed"] is False
    assert any(issue["check_name"] == "dtw_hesitate" for issue in result["issues"])


def test_reference_tube_flags_synchronized_multijoint_hesitation() -> None:
    thresholds = _thresholds(
        trajectory_dtw_min_segment_s=0.0,
        trajectory_dtw_velocity_floor=0.001,
        trajectory_dtw_hesitation_multiplier=1.0,
        trajectory_dtw_deviation_multiplier=100.0,
    )
    references = [
        _entry("ref-1", [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
        _entry("ref-2", [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
    ]
    tube = ReferenceTubeBuilder(thresholds=thresholds).build(references)
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0, 0.0, 0.0], [0.7, 0.7, 0.7], [0.2, 0.2, 0.2]]),
    )

    assert result["passed"] is False
    evidence = next(
        issue["value"]["evidence"]
        for issue in result["issues"]
        if issue["check_name"] == "dtw_hesitate"
    )
    assert evidence["velocity_energy"] > evidence["threshold"]


def test_reference_tube_flags_stall() -> None:
    thresholds = _thresholds(
        trajectory_dtw_min_segment_s=0.0,
        trajectory_dtw_stall_frame_threshold=3.0,
    )
    references = [
        _entry("ref-1", [[0.0], [1.0], [2.0]]),
        _entry("ref-2", [[0.0], [1.0], [2.0]]),
    ]
    tube = ReferenceTubeBuilder(thresholds=thresholds).build(references)
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0], [1.0], [1.0], [1.0], [1.0], [2.0]]),
    )

    assert result["passed"] is False
    assert any(issue["check_name"] == "dtw_stall" for issue in result["issues"])


def test_reference_tube_single_reference_fallback_can_run() -> None:
    thresholds = _thresholds(trajectory_dtw_min_segment_s=0.0)
    tube = ReferenceTubeBuilder(thresholds=thresholds).build([
        _entry("ref-1", [[0.0], [1.0], [2.0]]),
    ])
    assert tube is not None

    result = ReferenceTubeEvaluator(tube, thresholds=thresholds).evaluate(
        _entry("candidate", [[0.0], [1.0], [2.0]]),
    )

    assert result["passed"] is True


def test_trajectory_dtw_excludes_candidate_from_references() -> None:
    thresholds = _thresholds(
        trajectory_dtw_min_segment_s=0.0,
        trajectory_dtw_min_reference_count=1.0,
    )
    reference = {
        **_entry("reference", [[0.0], [0.1], [0.2]]),
        "task_key": "pick",
        "robot_type": "arm",
        "canonical_mode": "joint_canonical",
        "base_quality_passed": True,
    }
    candidate = {
        **_entry("candidate", [[0.0], [0.8], [0.2]]),
        "task_key": "pick",
        "robot_type": "arm",
        "canonical_mode": "joint_canonical",
        "base_quality_passed": True,
    }
    groups = {"pick::arm::joint_canonical": [reference, candidate]}

    result = _evaluate_trajectory_entry(candidate, groups, thresholds)

    assert result["passed"] is False
    assert any(issue["check_name"] == "dtw_deviation" for issue in result["issues"])


def test_trajectory_dtw_requires_enough_external_references() -> None:
    thresholds = _thresholds(
        trajectory_dtw_min_segment_s=0.0,
        trajectory_dtw_min_reference_count=2.0,
    )
    reference = {
        **_entry("reference", [[0.0], [0.1], [0.2]]),
        "task_key": "pick",
        "robot_type": "arm",
        "canonical_mode": "joint_canonical",
        "base_quality_passed": True,
    }
    candidate = {
        **_entry("candidate", [[0.0], [0.8], [0.2]]),
        "task_key": "pick",
        "robot_type": "arm",
        "canonical_mode": "joint_canonical",
        "base_quality_passed": True,
    }
    groups = {"pick::arm::joint_canonical": [reference, candidate]}

    result = _evaluate_trajectory_entry(candidate, groups, thresholds)

    assert result["passed"] is False
    assert result["issues"][0]["check_name"] == "insufficient_references"
    assert result["issues"][0]["value"]["reference_count"] == 1


def test_quality_batch_marks_self_reference_as_unverified(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "demo"
    dataset_path.mkdir()
    info = {
        "total_episodes": 1,
        "fps": 30,
        "robot_type": "so101",
        "features": {
            "action": {"names": ["joint_0"]},
            "observation.state": {"names": ["joint_0"]},
        },
    }

    def fake_load_episode_data(_dataset_path, episode_index):
        return {
            "info": info,
            "episode_meta": {"episode_index": episode_index, "length": 3.0, "task": "pick"},
            "rows": [
                {"timestamp": 0.0, "action": [0.0], "observation.state": [0.0], "task": "pick"},
                {"timestamp": 0.5, "action": [0.1], "observation.state": [0.1], "task": "pick"},
                {"timestamp": 1.0, "action": [0.2], "observation.state": [0.2], "task": "pick"},
            ],
            "dataset_path": dataset_path,
            "parquet_path": dataset_path / "data.parquet",
            "video_files": [],
        }

    monkeypatch.setattr(curation_service, "_load_info", lambda _path: info)
    monkeypatch.setattr(curation_service, "load_episode_data", fake_load_episode_data)
    monkeypatch.setattr("roboclaw.data.curation.validators.load_episode_data", fake_load_episode_data)
    monkeypatch.setattr("roboclaw.data.curation.trajectory_entries.load_episode_data", fake_load_episode_data)
    monkeypatch.setattr(curation_service, "save_working_quality_parquet", lambda *_args: {"path": ""})

    result = _LegacyCurationService(dataset_path, "demo").run_quality_batch(
        ["metadata", "action", "trajectory_dtw"],
        threshold_overrides={
            "metadata_require_data_files": 0.0,
            "metadata_require_videos": 0.0,
            "metadata_require_task_description": 0.0,
            "metadata_min_duration_s": 0.1,
            "action_min_duration_s": 0.1,
            "action_max_all_static_s": 10.0,
            "action_max_key_static_s": 10.0,
            "action_max_velocity_rad_s": 10.0,
        },
    )

    episode = result["episodes"][0]
    assert "trajectory_dtw" in episode["validators"]
    assert episode["validators"]["trajectory_dtw"]["passed"] is False
    assert any(issue["check_name"] == "no_reference" for issue in episode["issues"])
    assert result["passed"] == 0
