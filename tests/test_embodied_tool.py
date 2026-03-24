"""Tests for the EmbodiedTool integration with the agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from unittest.mock import patch as std_patch

from roboclaw.embodied.setup import (
    _CALIBRATION_ROOT,
    arm_display_name,
    find_arm,
    load_setup,
    remove_arm,
    remove_camera,
    save_setup,
    set_arm,
    set_camera,
)
from roboclaw.embodied.tool import EmbodiedTool, _resolve_operation_arms

_MOCK_SCANNED_PORTS = [
    {"by_path": "/dev/serial/by-path/pci-0:2.1", "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00", "dev": "/dev/ttyACM0"},
    {"by_path": "/dev/serial/by-path/pci-0:2.2", "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00", "dev": "/dev/ttyACM1"},
]


def test_tool_schema() -> None:
    tool = EmbodiedTool()
    assert tool.name == "embodied"
    assert "robot" in tool.description.lower()

    params = tool.parameters
    assert params["type"] == "object"
    assert "action" in params["properties"]
    assert params["required"] == ["action"]

    action_schema = params["properties"]["action"]
    assert action_schema["type"] == "string"
    expected_actions = [
        "doctor", "calibrate", "teleoperate", "record",
        "train", "run_policy", "job_status",
        "setup_show", "set_arm", "remove_arm", "set_camera", "remove_camera",
    ]
    assert action_schema["enum"] == expected_actions


_MOCK_SETUP = {
    "version": 2,
    "arms": [
        {
            "alias": "right_follower",
            "type": "so101_follower",
            "port": "/dev/ttyACM0",
            "calibration_dir": "/cal/f",
            "calibrated": False,
        },
        {
            "alias": "left_leader",
            "type": "so101_leader",
            "port": "/dev/ttyACM1",
            "calibration_dir": "/cal/l",
            "calibrated": False,
        },
    ],
    "cameras": {
        "front": {"by_path": "", "by_id": "", "dev": "/dev/video0"},
    },
    "datasets": {"root": "/data"},
    "policies": {"root": "/policies"},
    "scanned_ports": [],
    "scanned_cameras": [],
}


@pytest.mark.asyncio
async def test_doctor_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (0, "lerobot 0.5.0", "")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="doctor")

    assert "lerobot 0.5.0" in result
    assert "setup" in result.lower()


@pytest.mark.asyncio
async def test_calibrate_all_arms() -> None:
    mock_handoff = AsyncMock()
    tool = EmbodiedTool(tty_handoff=mock_handoff)
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.setup.mark_arm_calibrated"),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert "right_follower" in result
    assert "left_leader" in result
    assert mock_runner.run_interactive.call_count == 2
    assert mock_handoff.call_count == 4  # start+stop for each arm


@pytest.mark.asyncio
async def test_calibrate_no_arms() -> None:
    empty_setup = {**_MOCK_SETUP, "arms": []}
    tool = EmbodiedTool()
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=empty_setup):
        result = await tool.execute(action="calibrate")
    assert "no arms" in result.lower()


@pytest.mark.asyncio
async def test_calibrate_no_tty() -> None:
    tool = EmbodiedTool()  # no tty_handoff
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="calibrate")
    assert "local terminal" in result.lower()


@pytest.mark.asyncio
async def test_record_action() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="record", dataset_name="test", task="grasp", num_episodes=5,
            follower_names="right_follower", leader_names="left_leader",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args[0][0]
    assert "--robot.type=so101_follower" in argv
    assert "--teleop.type=so101_leader" in argv
    assert any("--robot.cameras=" in a for a in argv)


@pytest.mark.asyncio
async def test_train_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run_detached.return_value = "job-abc-123"

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="train", dataset_name="test", steps=5000)

    assert "job-abc-123" in result


@pytest.mark.asyncio
async def test_unknown_action() -> None:
    tool = EmbodiedTool()
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="fly_to_moon")
    assert "Unknown action" in result


# ── setup.py structured mutator tests ───────────────────────────────


@pytest.fixture()
def setup_file(tmp_path: Path) -> Path:
    """Create a minimal setup.json for testing."""
    p = tmp_path / "setup.json"
    base = {
        "version": 2,
        "arms": [],
        "cameras": {},
        "datasets": {"root": "/data"},
        "policies": {"root": "/policies"},
        "scanned_ports": [
            {"by_path": "/dev/serial/by-path/pci-0:2.1", "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00", "dev": "/dev/ttyACM0"},
            {"by_path": "/dev/serial/by-path/pci-0:2.2", "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00", "dev": "/dev/ttyACM1"},
        ],
        "scanned_cameras": [
            {"by_path": "/dev/v4l/by-path/cam0", "by_id": "usb-cam0", "dev": "/dev/video0"},
            {"by_path": "/dev/v4l/by-path/cam1", "by_id": "usb-cam1", "dev": "/dev/video2"},
        ],
    }
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_set_arm(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm is not None
    assert arm["type"] == "so101_follower"
    assert arm["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00"
    assert arm["calibration_dir"] == str(_CALIBRATION_ROOT / "5B14032630")
    assert arm["calibrated"] is False
    # Verify persisted
    persisted = load_setup(setup_file)
    assert find_arm(persisted["arms"], "my_follower") == arm


def test_set_arm_replaces_existing(setup_file: Path) -> None:
    """Setting an arm with the same alias should replace the existing entry."""
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("my_arm", "so101_follower", "/dev/ttyACM0", path=setup_file)
        result = set_arm("my_arm", "so101_leader", "/dev/ttyACM1", path=setup_file)
    assert len(result["arms"]) == 1
    arm = find_arm(result["arms"], "my_arm")
    assert arm["type"] == "so101_leader"
    assert arm["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00"


def test_set_arm_resolves_volatile_port(setup_file: Path) -> None:
    """Volatile /dev/ttyACMx should be resolved to stable /dev/serial/by-id/..."""
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_leader", "so101_leader", "/dev/ttyACM1", path=setup_file)
    arm = find_arm(result["arms"], "my_leader")
    assert arm["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00"


def test_set_arm_keeps_stable_port(setup_file: Path) -> None:
    """Already-stable by-id port should be kept as-is."""
    stable = "/dev/serial/by-id/usb-custom-device"
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", stable, path=setup_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm["port"] == stable


def test_set_arm_unmatched_volatile_port(setup_file: Path) -> None:
    """Volatile port not in scan results should be kept as-is."""
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyUSB99", path=setup_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm["port"] == "/dev/ttyUSB99"


def test_set_arm_invalid_type(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid arm_type"):
            set_arm("my_follower", "bogus_arm", "/dev/ttyACM0", path=setup_file)


def test_set_arm_empty_alias(setup_file: Path) -> None:
    with pytest.raises(ValueError, match="Arm alias is required"):
        set_arm("", "so101_follower", "/dev/ttyACM0", path=setup_file)


def test_remove_arm(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    result = remove_arm("my_follower", path=setup_file)
    assert find_arm(result["arms"], "my_follower") is None


def test_remove_arm_missing(setup_file: Path) -> None:
    with pytest.raises(ValueError, match="No arm with alias"):
        remove_arm("nonexistent", path=setup_file)


def test_set_camera(setup_file: Path) -> None:
    result = set_camera("front", 0, path=setup_file)
    cam = result["cameras"]["front"]
    assert cam["by_path"] == "/dev/v4l/by-path/cam0"
    assert cam["dev"] == "/dev/video0"
    assert cam["by_id"] == "usb-cam0"
    # Only _CAMERA_FIELDS should be copied
    assert set(cam.keys()) <= {"by_path", "by_id", "dev"}


def test_set_camera_bad_index(setup_file: Path) -> None:
    with pytest.raises(ValueError, match="out of range"):
        set_camera("front", 99, path=setup_file)


def test_remove_camera(setup_file: Path) -> None:
    set_camera("front", 0, path=setup_file)
    result = remove_camera("front", path=setup_file)
    assert "front" not in result["cameras"]


def test_remove_camera_missing(setup_file: Path) -> None:
    with pytest.raises(ValueError, match="No camera named"):
        remove_camera("nonexistent", path=setup_file)


def test_validation_rejects_unknown_arm_fields(setup_file: Path) -> None:
    """save_setup should reject arms with unexpected fields."""
    bad = load_setup(setup_file)
    bad["arms"] = [{"alias": "x", "type": "so101_follower", "port": "/dev/x", "junk": True}]
    with pytest.raises(ValueError, match="unknown fields"):
        save_setup(bad, setup_file)


def test_validation_rejects_unknown_camera_fields(setup_file: Path) -> None:
    bad = load_setup(setup_file)
    bad["cameras"]["front"] = {"dev": "/dev/video0", "fps": 30}
    with pytest.raises(ValueError, match="unknown fields"):
        save_setup(bad, setup_file)


def test_validation_rejects_bad_arm_type(setup_file: Path) -> None:
    bad = load_setup(setup_file)
    bad["arms"] = [{"alias": "x", "type": "garbage", "port": "/dev/x"}]
    with pytest.raises(ValueError, match="invalid type"):
        save_setup(bad, setup_file)


# ── arm_display_name / find_arm tests ────────────────────────────────


def test_arm_display_name() -> None:
    assert arm_display_name({"alias": "right"}) == "right"
    assert arm_display_name({}) == "unnamed"
    assert arm_display_name({"alias": ""}) == ""


def test_find_arm() -> None:
    arms = [
        {"alias": "a", "type": "so101_follower"},
        {"alias": "b", "type": "so101_leader"},
    ]
    assert find_arm(arms, "a") == arms[0]
    assert find_arm(arms, "b") == arms[1]
    assert find_arm(arms, "c") is None
    assert find_arm([], "a") is None


# ── _resolve_operation_arms tests ────────────────────────────────────


def test_resolve_operation_arms_single() -> None:
    result = _resolve_operation_arms(_MOCK_SETUP, "right_follower", "left_leader")
    assert isinstance(result, dict)
    assert result["mode"] == "single"
    assert len(result["followers"]) == 1
    assert len(result["leaders"]) == 1
    assert result["followers"][0]["alias"] == "right_follower"
    assert result["leaders"][0]["alias"] == "left_leader"


def test_resolve_operation_arms_bimanual() -> None:
    bimanual_setup = {
        **_MOCK_SETUP,
        "arms": [
            {"alias": "left_f", "type": "so101_follower", "port": "/dev/a", "calibration_dir": "/c/a", "calibrated": True},
            {"alias": "right_f", "type": "so101_follower", "port": "/dev/b", "calibration_dir": "/c/b", "calibrated": True},
            {"alias": "left_l", "type": "so101_leader", "port": "/dev/c", "calibration_dir": "/c/c", "calibrated": True},
            {"alias": "right_l", "type": "so101_leader", "port": "/dev/d", "calibration_dir": "/c/d", "calibrated": True},
        ],
    }
    result = _resolve_operation_arms(bimanual_setup, "left_f,right_f", "left_l,right_l")
    assert isinstance(result, dict)
    assert result["mode"] == "bimanual"
    assert len(result["followers"]) == 2
    assert len(result["leaders"]) == 2


def test_resolve_operation_arms_auto() -> None:
    """When names are omitted, auto-resolve from all arms in setup."""
    result = _resolve_operation_arms(_MOCK_SETUP, "", "")
    assert isinstance(result, dict)
    assert result["mode"] == "single"
    assert result["followers"][0]["alias"] == "right_follower"
    assert result["leaders"][0]["alias"] == "left_leader"


def test_resolve_operation_arms_missing_arm() -> None:
    result = _resolve_operation_arms(_MOCK_SETUP, "nonexistent", "left_leader")
    assert isinstance(result, str)
    assert "nonexistent" in result


def test_resolve_operation_arms_empty_setup() -> None:
    empty = {**_MOCK_SETUP, "arms": []}
    result = _resolve_operation_arms(empty, "", "")
    assert isinstance(result, str)
    assert "no arms" in result.lower()


def test_resolve_operation_arms_count_mismatch() -> None:
    setup_3 = {
        **_MOCK_SETUP,
        "arms": [
            {"alias": "f1", "type": "so101_follower", "port": "/dev/a", "calibration_dir": "/c/a", "calibrated": True},
            {"alias": "f2", "type": "so101_follower", "port": "/dev/b", "calibration_dir": "/c/b", "calibrated": True},
            {"alias": "l1", "type": "so101_leader", "port": "/dev/c", "calibration_dir": "/c/c", "calibrated": True},
        ],
    }
    result = _resolve_operation_arms(setup_3, "f1,f2", "l1")
    assert isinstance(result, str)
    assert "mismatch" in result.lower()


# ── Bimanual teleoperate test ────────────────────────────────────────


@pytest.mark.asyncio
async def test_teleoperate_bimanual() -> None:
    bimanual_setup = {
        **_MOCK_SETUP,
        "arms": [
            {"alias": "left_f", "type": "so101_follower", "port": "/dev/a", "calibration_dir": "/c/a", "calibrated": True},
            {"alias": "right_f", "type": "so101_follower", "port": "/dev/b", "calibration_dir": "/c/b", "calibrated": True},
            {"alias": "left_l", "type": "so101_leader", "port": "/dev/c", "calibration_dir": "/c/c", "calibrated": True},
            {"alias": "right_l", "type": "so101_leader", "port": "/dev/d", "calibration_dir": "/c/d", "calibrated": True},
        ],
    }
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=bimanual_setup),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="teleoperate",
            follower_names="left_f,right_f",
            leader_names="left_l,right_l",
        )

    assert "Teleoperation finished" in result
    argv = mock_runner.run_interactive.call_args[0][0]
    assert "lerobot-teleoperate" == argv[0]
    assert "--robot.type=bi_so_follower" in argv
    assert "--teleop.type=bi_so_leader" in argv
    assert any("--robot.left_arm_config.port=" in a for a in argv)
    assert any("--robot.right_arm_config.port=" in a for a in argv)
    assert any("--teleop.left_arm_config.port=" in a for a in argv)
    assert any("--teleop.right_arm_config.port=" in a for a in argv)


# ── Serial number extraction test ────────────────────────────────────


def test_calibration_dir_uses_serial_number(setup_file: Path) -> None:
    """calibration_dir should be based on serial number extracted from by_id port."""
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm["calibration_dir"] == str(_CALIBRATION_ROOT / "5B14032630")
