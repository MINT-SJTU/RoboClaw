from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from roboclaw.embodied.command.helpers import group_arms
from roboclaw.embodied.embodiment.hardware.monitor import check_arm_status
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.binding import (
    ArmBinding,
    ArmRole,
    CameraBinding,
    HandBinding,
    load_binding,
)
from roboclaw.embodied.embodiment.manifest.helpers import load_manifest, save_manifest


def test_load_binding_returns_typed_objects() -> None:
    arm = load_binding(
        {
            "alias": "left_follower",
            "type": "so101_follower",
            "port": "/dev/ttyACM0",
            "calibration_dir": "/cal/ttyACM0",
            "calibrated": True,
        },
        "arm",
        {},
    )
    camera = load_binding(
        {
            "alias": "left_wrist",
            "side": "left",
            "port": "/dev/video0",
            "width": 640,
            "height": 480,
            "fps": 30,
        },
        "camera",
        {},
    )
    hand = load_binding(
        {
            "alias": "left_hand",
            "type": "inspire_rh56",
            "port": "/dev/ttyUSB0",
            "slave_id": 1,
        },
        "hand",
        {},
    )

    assert isinstance(arm, ArmBinding)
    assert arm.role is ArmRole.FOLLOWER
    assert arm.to_dict()["type"] == "so101_follower"
    assert not hasattr(arm, "is_follower")
    assert not hasattr(arm, "kind")

    assert isinstance(camera, CameraBinding)
    assert camera.side == "left"
    assert camera.to_dict()["side"] == "left"

    assert isinstance(hand, HandBinding)
    assert hand.hand_type == "inspire_rh56"
    assert hand.to_dict()["type"] == "inspire_rh56"


def test_manifest_returns_typed_bindings_and_stable_snapshot(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    save_manifest(
        {
            "version": 2,
            "arms": [
                {
                    "alias": "right_follower",
                    "type": "so101_follower",
                    "port": "/dev/ttyACM0",
                    "calibration_dir": "/cal/right",
                    "calibrated": False,
                }
            ],
            "hands": [
                {
                    "alias": "left_hand",
                    "type": "inspire_rh56",
                    "port": "/dev/ttyUSB0",
                    "slave_id": 1,
                }
            ],
            "cameras": [
                {
                    "alias": "front",
                    "port": "/dev/video0",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                }
            ],
            "datasets": {"root": "/data"},
            "policies": {"root": "/policies"},
        },
        manifest_path,
    )

    with patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=[]):
        manifest = Manifest(path=manifest_path)

    arm = manifest.arms[0]
    camera = manifest.cameras[0]
    hand = manifest.hands[0]

    assert isinstance(arm, ArmBinding)
    assert isinstance(camera, CameraBinding)
    assert isinstance(hand, HandBinding)
    assert isinstance(manifest.find_arm("right_follower"), ArmBinding)
    assert isinstance(manifest.find_camera("front"), CameraBinding)
    assert isinstance(manifest.find_hand("left_hand"), HandBinding)

    persisted = load_manifest(manifest_path)
    assert manifest.snapshot == persisted


def test_manifest_setters_return_typed_bindings(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = Manifest(path=manifest_path)
    serial = SerialInterface(by_id="/dev/serial/by-id/test-arm", dev="/dev/ttyACM0")
    camera_interface = VideoInterface(dev="/dev/video0", width=1280, height=720, fps=60)

    calibration_root = tmp_path / "calibration"
    with patch("roboclaw.embodied.embodiment.manifest.state.get_calibration_root", return_value=calibration_root):
        arm = manifest.set_arm("left_leader", "so101_leader", serial)
    camera = manifest.set_camera("front", camera_interface)

    assert isinstance(arm, ArmBinding)
    assert arm.role is ArmRole.LEADER
    assert arm.calibration_dir == str(calibration_root / "test-arm")
    assert isinstance(camera, CameraBinding)
    assert manifest.snapshot["arms"][0]["type"] == "so101_leader"
    assert manifest.snapshot["cameras"][0]["width"] == 1280


def test_manifest_camera_persists_stable_id_and_reload_recovers_runtime_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = Manifest(path=manifest_path)
    camera_interface = VideoInterface(
        by_id="usb-camera",
        by_path="USB相机",
        dev="2",
        width=1280,
        height=720,
        fps=60,
    )

    manifest.set_camera("front", camera_interface)

    assert manifest.snapshot["cameras"][0]["port"] == "usb-camera"

    with patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=[camera_interface]):
        reloaded = Manifest(path=manifest_path)

    assert reloaded.cameras[0].interface.address == "usb-camera"
    assert reloaded.cameras[0].interface.runtime_address == "2"


def test_group_arms_and_monitor_status_use_explicit_arm_role() -> None:
    follower = load_binding(
        {
            "alias": "right_follower",
            "type": "so101_follower",
            "port": "/dev/ttyACM0",
            "calibration_dir": "/cal/right",
            "calibrated": True,
        },
        "arm",
        {},
    )
    leader = load_binding(
        {
            "alias": "left_leader",
            "type": "so101_leader",
            "port": "/dev/ttyACM1",
            "calibration_dir": "/cal/left",
            "calibrated": False,
        },
        "arm",
        {},
    )

    grouped = group_arms([follower, leader])
    leader_status = check_arm_status(leader)

    assert [arm.alias for arm in grouped["followers"]] == ["right_follower"]
    assert [arm.alias for arm in grouped["leaders"]] == ["left_leader"]
    assert leader_status.arm_type == "so101_leader"
    assert leader_status.role == "leader"
