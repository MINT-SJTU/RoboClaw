"""Tests for InspireController with mocked Modbus + hardware integration tests."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from roboclaw.embodied.embodiment.hand.inspire_rh56 import (
    InspireController,
    _REG_ANGLE_ACT,
    _REG_ANGLE_SET,
    _REG_FORCE_ACT,
)
from roboclaw.embodied.embodiment.hand.registry import INSPIRE_RH56

PORT = "/dev/ttyUSB0"


# -- Unit tests (mocked Modbus) --


def _mock_bus():
    bus = MagicMock()
    bus.read_registers.side_effect = lambda addr, count: {
        _REG_ANGLE_ACT: [0, 100, 200, 300, 400, 500],
        _REG_FORCE_ACT: [10, 20, 30, 40, 50, 60],
    }[addr]
    return bus


def _patch_session(bus):
    """Return a patch that replaces _session with a context manager yielding *bus*."""
    @contextmanager
    def _fake_session(port, slave_id):
        yield bus

    return patch.object(InspireController, "_session", staticmethod(_fake_session))


@pytest.fixture()
def mock_bus():
    bus = _mock_bus()
    with _patch_session(bus):
        yield bus


def test_open_hand(mock_bus: MagicMock) -> None:
    result = InspireController().open_hand(PORT)
    assert result == "Hand opened."
    mock_bus.write_registers.assert_called_once_with(
        _REG_ANGLE_SET, [1000, 1000, 1000, 1000, 1000, 1000],
    )


def test_close_hand(mock_bus: MagicMock) -> None:
    result = InspireController().close_hand(PORT)
    assert result == "Hand closed."
    mock_bus.write_registers.assert_called_once_with(
        _REG_ANGLE_SET, [0, 0, 0, 0, 0, 0],
    )


def test_set_pose(mock_bus: MagicMock) -> None:
    positions = [100, 200, 300, 400, 500, 600]
    result = InspireController().set_pose(PORT, positions)
    assert "little=100" in result
    assert "thumb_rotation=600" in result
    mock_bus.write_registers.assert_called_once_with(_REG_ANGLE_SET, positions)


def test_get_status(mock_bus: MagicMock) -> None:
    result = InspireController().get_status(PORT)
    assert "angles=" in result
    assert "forces=" in result
    assert "'little': 0" in result
    assert "'thumb_rotation': 500" in result


def test_set_pose_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 6"):
        InspireController().set_pose(PORT, [0, 1, 2])


def test_set_pose_out_of_range() -> None:
    with pytest.raises(ValueError, match="0-1000"):
        InspireController().set_pose(PORT, [0, 0, 0, 0, 0, 1001])
    with pytest.raises(ValueError, match="0-1000"):
        InspireController().set_pose(PORT, [-1, 0, 0, 0, 0, 0])


def test_close_called_on_error(mock_bus: MagicMock) -> None:
    """Ensure bus.close() is called even when write_registers raises."""
    mock_bus.write_registers.side_effect = RuntimeError("serial error")
    with pytest.raises(RuntimeError, match="serial error"):
        InspireController().open_hand(PORT)


def test_finger_labels() -> None:
    assert len(INSPIRE_RH56.finger_labels) == 6
    assert INSPIRE_RH56.finger_labels == ("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation")


# -- Hardware integration tests --


@pytest.mark.hardware
def test_hw_get_status() -> None:
    result = InspireController().get_status(PORT, slave_id=2)
    assert "angles=" in result
    assert "forces=" in result


@pytest.mark.hardware
def test_hw_open_hand() -> None:
    result = InspireController().open_hand(PORT, slave_id=2)
    assert result == "Hand opened."


@pytest.mark.hardware
def test_hw_close_hand() -> None:
    result = InspireController().close_hand(PORT, slave_id=2)
    assert result == "Hand closed."
