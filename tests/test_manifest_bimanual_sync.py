"""Tests for bimanual calibration directory sync from per-serial sources.

Covers the regression where manifests written by older code paths omit the
`side` field on arms, causing `_pair_arms_by_side` to raise and
`refresh_bimanual_cal_dirs` to silently swallow the failure (so
`bimanual_left.json` / `bimanual_right.json` stop refreshing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.embodied.embodiment.manifest.helpers import (
    _pair_arms_by_side,
    get_calibration_root,
    refresh_bimanual_cal_dirs,
)


def test_pair_arms_with_explicit_side() -> None:
    right_arm = {
        "alias": "right_follower",
        "type": "so101_follower",
        "port": "/dev/serial/by-id/right",
        "calibration_dir": "/tmp/right",
        "side": "right",
    }
    left_arm = {
        "alias": "left_follower",
        "type": "so101_follower",
        "port": "/dev/serial/by-id/left",
        "calibration_dir": "/tmp/left",
        "side": "left",
    }
    left, right = _pair_arms_by_side([right_arm, left_arm], "followers")
    assert left is left_arm
    assert right is right_arm


def test_pair_arms_falls_back_to_alias() -> None:
    right_arm = {
        "alias": "right_follower",
        "type": "so101_follower",
        "port": "/dev/serial/by-id/right",
        "calibration_dir": "/tmp/right",
    }
    left_arm = {
        "alias": "left_follower",
        "type": "so101_follower",
        "port": "/dev/serial/by-id/left",
        "calibration_dir": "/tmp/left",
    }
    left, right = _pair_arms_by_side([right_arm, left_arm], "followers")
    assert left is left_arm
    assert right is right_arm


def test_pair_arms_raises_when_ambiguous() -> None:
    arm_a = {"alias": "arm_a", "type": "so101_follower", "calibration_dir": "/tmp/a"}
    arm_b = {"alias": "arm_b", "type": "so101_follower", "calibration_dir": "/tmp/b"}
    with pytest.raises(ValueError):
        _pair_arms_by_side([arm_a, arm_b], "followers")


def test_pair_arms_rejects_malformed_explicit_sides() -> None:
    # When `side` is non-empty on at least one arm, treat the manifest as
    # authoritative — refuse to silently fall back to alias inference.
    a = {"alias": "left_follower", "side": "left", "calibration_dir": "/tmp/a"}
    b = {"alias": "right_follower", "side": "Left", "calibration_dir": "/tmp/b"}
    with pytest.raises(ValueError):
        _pair_arms_by_side([a, b], "followers")

    a = {"alias": "left_follower", "side": "left", "calibration_dir": "/tmp/a"}
    b = {"alias": "right_follower", "side": "left", "calibration_dir": "/tmp/b"}
    with pytest.raises(ValueError):
        _pair_arms_by_side([a, b], "followers")


def test_pair_arms_handles_none_alias() -> None:
    a = {"alias": None, "calibration_dir": "/tmp/a"}
    b = {"alias": None, "calibration_dir": "/tmp/b"}
    with pytest.raises(ValueError):
        _pair_arms_by_side([a, b], "followers")


def test_refresh_bimanual_cal_dirs_no_side_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOCLAW_HOME", str(tmp_path))
    cal_root = get_calibration_root()
    cal_root.mkdir(parents=True, exist_ok=True)

    left_serial = "5B14030892"
    right_serial = "5B14031512"
    left_dir = cal_root / left_serial
    right_dir = cal_root / right_serial
    left_dir.mkdir()
    right_dir.mkdir()
    left_content = '{"source": "left", "value": 111}\n'
    right_content = '{"source": "right", "value": 222}\n'
    (left_dir / f"{left_serial}.json").write_text(left_content, encoding="utf-8")
    (right_dir / f"{right_serial}.json").write_text(right_content, encoding="utf-8")

    # Pre-seed stale alias files so the test proves they get OVERWRITTEN —
    # not just that they get created from scratch.
    bimanual_dir = cal_root / "bimanual_followers"
    bimanual_dir.mkdir(parents=True)
    stale_content = '{"source": "stale", "value": -1}\n'
    (bimanual_dir / "bimanual_left.json").write_text(stale_content, encoding="utf-8")
    (bimanual_dir / "bimanual_right.json").write_text(stale_content, encoding="utf-8")

    manifest = {
        "version": 2,
        "arms": [
            {
                "alias": "right_follower",
                "type": "so101_follower",
                "port": "/dev/serial/by-id/usb-right",
                "calibration_dir": str(right_dir),
                "calibrated": True,
            },
            {
                "alias": "left_follower",
                "type": "so101_follower",
                "port": "/dev/serial/by-id/usb-left",
                "calibration_dir": str(left_dir),
                "calibrated": True,
            },
        ],
    }

    refresh_bimanual_cal_dirs(manifest)

    assert (bimanual_dir / "bimanual_left.json").read_text(encoding="utf-8") == left_content
    assert (bimanual_dir / "bimanual_right.json").read_text(encoding="utf-8") == right_content
