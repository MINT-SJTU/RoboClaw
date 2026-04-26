from __future__ import annotations

from pathlib import Path

from roboclaw.data.curation import visual_validators


def test_visual_validator_uses_metadata_when_video_decoder_unavailable(
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
                        "shape": [480, 640, 3],
                        "info": {
                            "video.fps": 30,
                            "video.width": 640,
                            "video.height": 480,
                        },
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
    assert issues["video_count"]["passed"] is True
    assert issues["video_accessibility"]["passed"] is True
    assert issues["video_resolution"]["passed"] is True
    assert issues["video_fps"]["passed"] is True
    assert result["passed"] is True
