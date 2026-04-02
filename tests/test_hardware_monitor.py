"""Tests for hardware monitor fault detection logic."""

from __future__ import annotations

import time

import pytest

from roboclaw.embodied.hardware_monitor import (
    FaultType,
    HardwareFault,
    _check_arms,
    _check_cameras,
    _fault_key,
)


# ---------------------------------------------------------------------------
# HardwareFault
# ---------------------------------------------------------------------------

class TestHardwareFault:
    def test_to_dict(self):
        fault = HardwareFault(
            fault_type=FaultType.ARM_DISCONNECTED,
            device_alias="left_follower",
            message="Arm 'left_follower' USB port not found",
            timestamp=1000.0,
        )
        d = fault.to_dict()
        assert d["fault_type"] == "arm_disconnected"
        assert d["device_alias"] == "left_follower"

    def test_fault_key_uniqueness(self):
        f1 = HardwareFault(FaultType.ARM_DISCONNECTED, "arm_a", "", 0)
        f2 = HardwareFault(FaultType.ARM_NOT_CALIBRATED, "arm_a", "", 0)
        f3 = HardwareFault(FaultType.ARM_DISCONNECTED, "arm_b", "", 0)
        assert _fault_key(f1) != _fault_key(f2)
        assert _fault_key(f1) != _fault_key(f3)

    def test_fault_key_same_for_same_type_and_device(self):
        f1 = HardwareFault(FaultType.ARM_DISCONNECTED, "arm_a", "msg1", 1.0)
        f2 = HardwareFault(FaultType.ARM_DISCONNECTED, "arm_a", "msg2", 2.0)
        assert _fault_key(f1) == _fault_key(f2)


# ---------------------------------------------------------------------------
# _check_arms
# ---------------------------------------------------------------------------

class TestCheckArms:
    def test_no_arms(self):
        faults: list[HardwareFault] = []
        _check_arms([], time.time(), faults)
        assert faults == []

    def test_missing_port(self, tmp_path):
        arms = [{"alias": "follower", "port": str(tmp_path / "nonexistent"), "calibrated": True}]
        faults: list[HardwareFault] = []
        _check_arms(arms, time.time(), faults)
        assert len(faults) == 1
        assert faults[0].fault_type == FaultType.ARM_DISCONNECTED

    def test_port_exists_but_not_calibrated(self, tmp_path):
        port_file = tmp_path / "ttyUSB0"
        port_file.touch()
        arms = [{"alias": "follower", "port": str(port_file), "calibrated": False}]
        faults: list[HardwareFault] = []
        _check_arms(arms, time.time(), faults)
        assert len(faults) == 1
        assert faults[0].fault_type == FaultType.ARM_NOT_CALIBRATED

    def test_port_exists_and_calibrated(self, tmp_path):
        port_file = tmp_path / "ttyUSB0"
        port_file.touch()
        arms = [{"alias": "follower", "port": str(port_file), "calibrated": True}]
        faults: list[HardwareFault] = []
        _check_arms(arms, time.time(), faults)
        assert faults == []

    def test_missing_port_skips_calibration_check(self, tmp_path):
        """If port is missing, only ARM_DISCONNECTED is reported (not also uncalibrated)."""
        arms = [{"alias": "follower", "port": str(tmp_path / "gone"), "calibrated": False}]
        faults: list[HardwareFault] = []
        _check_arms(arms, time.time(), faults)
        assert len(faults) == 1
        assert faults[0].fault_type == FaultType.ARM_DISCONNECTED

    def test_empty_port_no_fault(self):
        arms = [{"alias": "follower", "port": "", "calibrated": True}]
        faults: list[HardwareFault] = []
        _check_arms(arms, time.time(), faults)
        assert faults == []


# ---------------------------------------------------------------------------
# _check_cameras
# ---------------------------------------------------------------------------

class TestCheckCameras:
    def test_no_cameras(self):
        faults: list[HardwareFault] = []
        _check_cameras([], time.time(), faults, recording_active=False)
        assert faults == []

    def test_missing_camera_port(self, tmp_path):
        cams = [{"alias": "wrist_cam", "port": str(tmp_path / "video0")}]
        faults: list[HardwareFault] = []
        _check_cameras(cams, time.time(), faults, recording_active=False)
        assert len(faults) == 1
        assert faults[0].fault_type == FaultType.CAMERA_DISCONNECTED

    def test_camera_exists(self, tmp_path):
        port_file = tmp_path / "video0"
        port_file.touch()
        cams = [{"alias": "wrist_cam", "port": str(port_file)}]
        faults: list[HardwareFault] = []
        _check_cameras(cams, time.time(), faults, recording_active=False)
        assert faults == []

    def test_skip_during_recording(self, tmp_path):
        cams = [{"alias": "wrist_cam", "port": str(tmp_path / "gone")}]
        faults: list[HardwareFault] = []
        _check_cameras(cams, time.time(), faults, recording_active=True)
        assert faults == []

    def test_empty_port_no_fault(self):
        cams = [{"alias": "cam", "port": ""}]
        faults: list[HardwareFault] = []
        _check_cameras(cams, time.time(), faults, recording_active=False)
        assert faults == []
