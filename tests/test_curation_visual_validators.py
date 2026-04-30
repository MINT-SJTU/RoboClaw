from __future__ import annotations

from pathlib import Path

import numpy as np

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
        lambda _path, **_kwargs: ([], 0.0, 0, 0, 0),
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


def test_visual_validator_checks_each_declared_video_stream(
    monkeypatch,
    tmp_path: Path,
) -> None:
    front_path = tmp_path / "observation.images.front.mp4"
    wrist_path = tmp_path / "observation.images.wrist.mp4"
    front_path.write_bytes(b"video")
    wrist_path.write_bytes(b"video")
    monkeypatch.setattr(
        visual_validators,
        "_sample_video_frames",
        lambda _path, **_kwargs: ([], 0.0, 0, 0, 0),
    )

    result = visual_validators.validate_visual_assets(
        {
            "video_files": [front_path, wrist_path],
            "rows": [],
            "info": {
                "features": {
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "info": {"video.fps": 30, "video.width": 640, "video.height": 480},
                    },
                    "observation.images.wrist": {
                        "dtype": "video",
                        "shape": [240, 320, 3],
                        "info": {"video.fps": 15, "video.width": 320, "video.height": 240},
                    },
                }
            },
        },
        threshold_overrides={
            "visual_min_resolution_width": 640,
            "visual_min_resolution_height": 480,
            "visual_min_frame_rate": 24,
        },
    )

    failed = [issue for issue in result["issues"] if not issue["passed"]]
    assert result["passed"] is False
    assert any(issue["check_name"] == "video_resolution" and issue["value"]["stream"] == "wrist" for issue in failed)
    assert any(issue["check_name"] == "video_fps" and issue["value"]["stream"] == "wrist" for issue in failed)


def test_visual_sampling_is_bounded_by_max_samples() -> None:
    assert visual_validators._video_sample_count(10, fps=30.0, frame_count=1800) == 10

    rows = [
        {
            "timestamp": float(index),
            "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        for index in range(61)
    ]

    sampled = visual_validators._iter_visual_parquet_frames(rows)

    assert len(sampled) == 10


def test_visual_sampling_uses_episode_clip_bounds() -> None:
    indexes = visual_validators._sample_frame_indexes(
        5,
        fps=30.0,
        frame_count=3000,
        clip_start_s=10.0,
        clip_end_s=20.0,
    )

    assert indexes == [300, 374, 449, 524, 599]


def test_visual_validator_passes_clip_bounds_to_sampler(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    calls: list[dict[str, float | None]] = []

    def fake_sample(_path: Path, **kwargs: float | None):
        calls.append(kwargs)
        return ([np.zeros((2, 2, 3), dtype=np.uint8)], 30.0, 640, 480, 3000)

    monkeypatch.setattr(visual_validators, "_sample_video_frames", fake_sample)

    result = visual_validators.validate_visual_assets(
        {
            "video_files": [video_path],
            "rows": [],
            "episode_meta": {
                "videos/observation.images.front/from_timestamp": 10.0,
                "videos/observation.images.front/to_timestamp": 20.0,
            },
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
    )

    assert calls == [{"clip_start_s": 10.0, "clip_end_s": 20.0}]
