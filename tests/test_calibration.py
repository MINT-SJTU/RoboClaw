"""Tests for CalibrationSession."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roboclaw.embodied.calibration import CalibrationSession
from roboclaw.embodied.embodiment.arm.registry import SO101


@pytest.fixture
def arm_config(tmp_path: Path) -> dict:
    cal_dir = tmp_path / "calibration" / "TEST_SERIAL"
    cal_dir.mkdir(parents=True)
    return {
        "alias": "test_follower",
        "type": "so101_follower",
        "port": "/dev/ttyACM0",
        "calibration_dir": str(cal_dir),
        "calibrated": False,
    }


@pytest.fixture
def mock_bus():
    """Create a mock motor bus with realistic behavior."""
    bus = MagicMock()
    bus.motors = {
        "shoulder_pan": MagicMock(id=1),
        "shoulder_lift": MagicMock(id=2),
        "elbow_flex": MagicMock(id=3),
        "wrist_flex": MagicMock(id=4),
        "wrist_roll": MagicMock(id=5),
        "gripper": MagicMock(id=6),
    }
    bus.set_half_turn_homings.return_value = {
        "shoulder_pan": -100,
        "shoulder_lift": -200,
        "elbow_flex": 300,
        "wrist_flex": -400,
        "wrist_roll": 500,
        "gripper": 600,
    }
    # sync_read returns positions — vary between calls for min/max coverage
    call_count = [0]
    def _varying_sync_read(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            return {"shoulder_pan": 2000, "shoulder_lift": 2100, "elbow_flex": 2200, "wrist_flex": 2300, "gripper": 2400}
        return {"shoulder_pan": 1800, "shoulder_lift": 2500, "elbow_flex": 1900, "wrist_flex": 2700, "gripper": 2600}
    bus.sync_read.side_effect = _varying_sync_read
    return bus


def test_session_initial_state(arm_config: dict) -> None:
    session = CalibrationSession(arm_config)
    assert session.state == "idle"
    assert session.family == SO101


def test_session_state_transitions(arm_config: dict, mock_bus: MagicMock) -> None:
    session = CalibrationSession(arm_config)

    with patch.object(session, "_import_bus_class", return_value=lambda **kw: mock_bus), \
         patch.object(session, "_import_motor_types") as mock_mt, \
         patch.object(session, "_import_operating_mode") as mock_om, \
         patch.object(session, "_import_motor_calibration") as mock_mc:

        # Mock Motor and MotorNormMode
        MockMotor = MagicMock()
        MockNormMode = MagicMock()
        MockNormMode.RANGE_M100_100 = "range_m100_100"
        mock_mt.return_value = (MockMotor, MockNormMode)

        MockOperatingMode = MagicMock()
        MockOperatingMode.POSITION.value = 3
        mock_om.return_value = MockOperatingMode

        from types import SimpleNamespace
        mock_mc.return_value = lambda **kw: SimpleNamespace(**kw)

        # Connect
        session.connect()
        assert session.state == "connected"
        mock_bus.connect.assert_called_once()
        mock_bus.disable_torque.assert_called_once()

        # Set homing
        offsets = session.set_homing()
        assert session.state == "recording"
        assert offsets["shoulder_pan"] == -100
        mock_bus.set_half_turn_homings.assert_called_once()

        # Read range positions
        snapshot = session.read_range_positions()
        assert "shoulder_pan" in snapshot.positions
        assert "shoulder_pan" in snapshot.mins

        # Finish
        session.finish()
        assert session.state == "done"
        mock_bus.write_calibration.assert_called_once()


def test_session_saves_calibration_json(arm_config: dict, mock_bus: MagicMock) -> None:
    session = CalibrationSession(arm_config)

    with patch.object(session, "_import_bus_class", return_value=lambda **kw: mock_bus), \
         patch.object(session, "_import_motor_types") as mock_mt, \
         patch.object(session, "_import_operating_mode") as mock_om, \
         patch.object(session, "_import_motor_calibration") as mock_mc:

        MockMotor = MagicMock()
        MockNormMode = MagicMock()
        MockNormMode.RANGE_M100_100 = "range_m100_100"
        mock_mt.return_value = (MockMotor, MockNormMode)
        mock_om.return_value = MagicMock(POSITION=MagicMock(value=3))

        # Make MotorCalibration a simple namespace
        from types import SimpleNamespace
        mock_mc.return_value = lambda **kw: SimpleNamespace(**kw)

        # Make sync_read return different values on successive calls to test min/max
        positions_call_count = [0]
        def varying_positions(*args, **kwargs):
            positions_call_count[0] += 1
            if positions_call_count[0] == 1:
                # Initial read
                return {"shoulder_pan": 2000, "shoulder_lift": 2100, "elbow_flex": 2200, "wrist_flex": 2300, "gripper": 2400}
            # Moved positions
            return {"shoulder_pan": 1800, "shoulder_lift": 2500, "elbow_flex": 1900, "wrist_flex": 2700, "gripper": 2600}

        mock_bus.sync_read.side_effect = varying_positions

        session.connect()
        session.set_homing()
        session.read_range_positions()  # records min/max
        cal = session.finish()

        # Check calibration file was written
        cal_dir = Path(arm_config["calibration_dir"])
        cal_file = cal_dir / f"{cal_dir.name}.json"
        assert cal_file.exists()
        saved = json.loads(cal_file.read_text())
        assert "shoulder_pan" in saved
        assert saved["shoulder_pan"]["homing_offset"] == -100


def test_cancel_resets_state(arm_config: dict, mock_bus: MagicMock) -> None:
    session = CalibrationSession(arm_config)

    with patch.object(session, "_import_bus_class", return_value=lambda **kw: mock_bus), \
         patch.object(session, "_import_motor_types") as mock_mt, \
         patch.object(session, "_import_operating_mode") as mock_om:

        MockMotor = MagicMock()
        MockNormMode = MagicMock()
        MockNormMode.RANGE_M100_100 = "range_m100_100"
        mock_mt.return_value = (MockMotor, MockNormMode)
        mock_om.return_value = MagicMock(POSITION=MagicMock(value=3))

        session.connect()
        assert session.state == "connected"

        session.cancel()
        assert session.state == "idle"
        mock_bus.disconnect.assert_called()


def test_wrong_state_raises(arm_config: dict) -> None:
    session = CalibrationSession(arm_config)
    with pytest.raises(RuntimeError, match="Cannot set homing"):
        session.set_homing()
    with pytest.raises(RuntimeError, match="Cannot read range"):
        session.read_range_positions()
    with pytest.raises(RuntimeError, match="Cannot finish"):
        session.finish()
