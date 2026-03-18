from __future__ import annotations

import pytest

from roboclaw.embodied.execution.integration.bridges.ros2.scservo import ServoCalibration
from roboclaw.embodied.execution.integration.bridges.ros2.so101_feetech import (
    ADDR_HOMING_OFFSET,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    So101FeetechRuntime,
)
from roboclaw.embodied.execution.integration.bridges.ros2.stage1_server import Stage1Ros2BridgeServer


def test_stage1_server_rejects_unknown_robot_even_when_profile_matches() -> None:
    server = Stage1Ros2BridgeServer.__new__(Stage1Ros2BridgeServer)

    with pytest.raises(ValueError, match="Unknown stage-1 ROS2 robot"):
        server._build_runtime(
            profile_id="so101_ros2_standard",
            robot_id="custom_arm",
            device="/dev/ttyACM0",
            calibration_path=None,
            calibration_id="so101_real",
        )


def test_stage1_server_rejects_unknown_profile_even_when_robot_matches() -> None:
    server = Stage1Ros2BridgeServer.__new__(Stage1Ros2BridgeServer)

    with pytest.raises(ValueError, match="Unknown stage-1 ROS2 profile"):
        server._build_runtime(
            profile_id="custom_profile",
            robot_id="so101",
            device="/dev/ttyACM0",
            calibration_path=None,
            calibration_id="so101_real",
        )


def test_so101_runtime_applies_calibration_before_motion() -> None:
    runtime = So101FeetechRuntime.__new__(So101FeetechRuntime)
    runtime._calibration = {
        "gripper": ServoCalibration(
            id=6,
            drive_mode=1,
            homing_offset=-120,
            range_min=100,
            range_max=500,
        )
    }
    writes: list[tuple[int, int, int]] = []
    runtime._write2 = lambda servo_id, address, data: writes.append((servo_id, address, data))

    runtime._apply_calibration("gripper")

    assert (6, ADDR_HOMING_OFFSET, So101FeetechRuntime._encode_signed_16(-120)) in writes
    assert (6, ADDR_MIN_POSITION_LIMIT, 100) in writes
    assert (6, ADDR_MAX_POSITION_LIMIT, 500) in writes


def test_so101_runtime_uses_drive_mode_in_gripper_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = So101FeetechRuntime.__new__(So101FeetechRuntime)
    runtime._calibration = {
        "gripper": ServoCalibration(
            id=6,
            drive_mode=1,
            homing_offset=0,
            range_min=100,
            range_max=500,
        )
    }

    assert runtime._normalized_to_raw("gripper", 0.0) == 500
    assert runtime._normalized_to_raw("gripper", 100.0) == 100

    monkeypatch.setattr(runtime, "read_gripper_position", lambda: 500)
    assert runtime.gripper_percent() == 0.0
