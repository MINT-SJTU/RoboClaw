"""Tests for Manifest.rebind_arm() — change arm type/alias without re-running identify."""

from __future__ import annotations

import os
import json
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("ROBOCLAW_STUB", "1")

from roboclaw.embodied.manifest.state import Manifest
from roboclaw.embodied.interface.serial import SerialInterface


def _make_serial(dev="/dev/ttyACM0", by_id="usb-fake-001"):
    return SerialInterface(dev=dev, by_id=by_id, by_path="", bus_type="dynamixel", motor_ids=(1, 2, 3, 4, 5, 6))


@pytest.fixture
def manifest(tmp_path):
    home = tmp_path / ".roboclaw"
    ws = home / "workspace" / "embodied"
    ws.mkdir(parents=True)
    cal = ws / "calibration"
    cal.mkdir()
    manifest_path = ws / "manifest.json"
    with patch.dict(os.environ, {"ROBOCLAW_HOME": str(home)}):
        m = Manifest(manifest_path)
        yield m


class TestRebindArm:
    def test_rebind_changes_type_and_alias(self, manifest):
        iface = _make_serial()
        manifest.set_arm("left_leader", "koch_leader", iface)

        manifest.rebind_arm("left_leader", "left_follower", "koch_follower")

        assert manifest.find_arm("left_follower") is not None
        assert manifest.find_arm("left_leader") is None
        b = manifest.find_arm("left_follower")
        assert b.type_name == "koch_follower"
        assert b.interface.stable_id == iface.stable_id

    def test_rebind_preserves_calibration_same_model(self, manifest):
        iface = _make_serial()
        manifest.set_arm("left_leader", "koch_leader", iface)
        manifest.mark_arm_calibrated("left_leader")

        manifest.rebind_arm("left_leader", "left_follower", "koch_follower")

        b = manifest.find_arm("left_follower")
        assert b.calibrated is True

    def test_rebind_clears_calibration_different_model(self, manifest):
        iface = _make_serial()
        manifest.set_arm("left_leader", "koch_leader", iface)
        manifest.mark_arm_calibrated("left_leader")

        manifest.rebind_arm("left_leader", "left_follower", "so101_follower")

        b = manifest.find_arm("left_follower")
        assert b.calibrated is False

    def test_rebind_same_type_is_idempotent(self, manifest):
        iface = _make_serial()
        manifest.set_arm("left_leader", "koch_leader", iface)

        manifest.rebind_arm("left_leader", "left_leader", "koch_leader")

        b = manifest.find_arm("left_leader")
        assert b.type_name == "koch_leader"

    def test_rebind_alias_conflict_raises(self, manifest):
        iface1 = _make_serial(dev="/dev/ttyACM0", by_id="usb-fake-001")
        iface2 = _make_serial(dev="/dev/ttyACM1", by_id="usb-fake-002")
        manifest.set_arm("arm_a", "koch_leader", iface1)
        manifest.set_arm("arm_b", "koch_follower", iface2)

        with pytest.raises(ValueError, match="already exists"):
            manifest.rebind_arm("arm_a", "arm_b", "koch_follower")

    def test_rebind_not_found_raises(self, manifest):
        with pytest.raises(ValueError):
            manifest.rebind_arm("nonexistent", "new_name", "koch_follower")

    def test_rebind_invalid_type_raises(self, manifest):
        iface = _make_serial()
        manifest.set_arm("left_leader", "koch_leader", iface)

        with pytest.raises(ValueError, match="Invalid arm_type"):
            manifest.rebind_arm("left_leader", "left_follower", "nonexistent_leader")
