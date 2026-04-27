from __future__ import annotations

from pathlib import Path

from roboclaw.data.curation import visual_validators
from roboclaw.data.curation.validators import validate_ee_trajectory


def test_visual_video_count_threshold_blocks_when_partially_missing(tmp_path: Path) -> None:
    video_path = tmp_path / "front.mp4"
    video_path.write_bytes(b"video")

    result = visual_validators.validate_visual_assets(
        {
            "video_files": [video_path],
            "rows": [],
            "info": {
                "features": {
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {"video.fps": 30, "video.width": 640, "video.height": 480},
                    }
                }
            },
        },
        threshold_overrides={"visual_min_video_count": 2},
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["video_count"]["passed"] is False
    assert issues["video_count"]["level"] == "major"
    assert result["passed"] is False


def test_visual_unavailable_resolution_and_fps_are_blocking(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "front.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(
        visual_validators,
        "_sample_video_frames",
        lambda _path: ([], 0.0, 0, 0, 0),
    )

    result = visual_validators.validate_visual_assets(
        {
            "video_files": [video_path],
            "rows": [],
            "info": {
                "features": {
                    "observation.images.front": {
                        "dtype": "video",
                    }
                }
            },
        },
        threshold_overrides={
            "visual_min_resolution_width": 640,
            "visual_min_resolution_height": 480,
            "visual_min_frame_rate": 24,
        },
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["video_resolution"]["passed"] is False
    assert issues["video_resolution"]["level"] == "major"
    assert issues["video_fps"]["passed"] is False
    assert issues["video_fps"]["level"] == "major"
    assert result["passed"] is False


def test_depth_unavailable_metrics_are_blocking(tmp_path: Path) -> None:
    depth_path = tmp_path / "camera_depth.mp4"
    depth_path.write_bytes(b"depth")

    result = visual_validators.validate_depth_assets(
        {
            "video_files": [depth_path],
            "rows": [],
            "info": {
                "features": {
                    "observation.images.depth": {
                        "dtype": "video",
                    }
                }
            },
        },
        threshold_overrides={
            "depth_min_stream_count": 1,
            "depth_invalid_pixel_max": 0.1,
            "depth_continuity_min": 0.9,
        },
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["depth_accessibility"]["passed"] is True
    assert issues["depth_invalid_ratio"]["passed"] is False
    assert issues["depth_invalid_ratio"]["level"] == "major"
    assert issues["depth_continuity"]["passed"] is False
    assert issues["depth_continuity"]["level"] == "major"
    assert result["passed"] is False


def test_ee_threshold_failures_are_blocking() -> None:
    rows = [
        {
            "timestamp": index * 0.1,
            "action": [0.0, 0.0, 0.0],
            "observation.state": [0.0, 0.0, 0.0],
        }
        for index in range(8)
    ]

    result = validate_ee_trajectory(
        {
            "rows": rows,
            "info": {
                "features": {
                    "action": {"names": ["joint_0", "joint_1", "gripper"]},
                    "observation.state": {"names": ["joint_0", "joint_1", "gripper"]},
                },
            },
        },
        threshold_overrides={
            "ee_min_event_count": 1,
            "ee_min_gripper_span": 0.05,
        },
    )

    issues = {issue["check_name"]: issue for issue in result["issues"]}
    assert issues["grasp_event_count"]["passed"] is False
    assert issues["grasp_event_count"]["level"] == "major"
    assert issues["gripper_motion_span"]["passed"] is False
    assert issues["gripper_motion_span"]["level"] == "major"
    assert result["passed"] is False
