"""Tests for the EmbodiedTool integration with the agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from unittest.mock import patch as std_patch

from roboclaw.embodied.setup import (
    arm_display_name,
    find_arm,
    load_setup,
    remove_arm,
    remove_camera,
    rename_arm,
    save_setup,
    set_arm,
    set_camera,
)
from roboclaw.embodied.tool import EmbodiedTool, _group_arms, _resolve_arms

_MOCK_SCANNED_PORTS = [
    {
        "by_path": "/dev/serial/by-path/pci-0:2.1",
        "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00",
        "dev": "/dev/ttyACM0",
    },
    {
        "by_path": "/dev/serial/by-path/pci-0:2.2",
        "by_id": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00",
        "dev": "/dev/ttyACM1",
    },
]

_FOLLOWER_PORT = _MOCK_SCANNED_PORTS[0]["by_id"]
_LEADER_PORT = _MOCK_SCANNED_PORTS[1]["by_id"]

_MOCK_SETUP = {
    "version": 2,
    "arms": [
        {
            "alias": "right_follower",
            "type": "so101_follower",
            "port": _FOLLOWER_PORT,
            "calibration_dir": "/cal/f",
            "calibrated": False,
        },
        {
            "alias": "left_leader",
            "type": "so101_leader",
            "port": _LEADER_PORT,
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


@pytest.fixture(autouse=True)
def calibration_root(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    with std_patch("roboclaw.embodied.setup._CALIBRATION_ROOT", root):
        yield root


def test_tool_schema() -> None:
    tool = EmbodiedTool()
    params = tool.parameters

    assert tool.name == "embodied"
    assert "replay" in tool.description.lower()
    assert params["type"] == "object"
    assert params["required"] == ["action"]
    assert params["properties"]["use_cameras"]["default"] is True
    assert "name" not in params["properties"]
    assert "follower_names" not in params["properties"]
    assert "leader_names" not in params["properties"]
    assert params["properties"]["arms"]["type"] == "string"
    assert params["properties"]["target_action"]["type"] == "string"
    assert params["properties"]["episode"]["type"] == "integer"
    assert params["properties"]["alias"]["type"] == "string"
    assert "port" not in params["properties"]
    assert params["properties"]["action"]["enum"] == [
        "doctor",
        "identify",
        "describe",
        "calibrate",
        "teleoperate",
        "record",
        "replay",
        "train",
        "run_policy",
        "job_status",
        "setup_show",
        "set_arm",
        "rename_arm",
        "remove_arm",
        "set_camera",
        "remove_camera",
    ]


@pytest.mark.asyncio
async def test_describe_action() -> None:
    tool = EmbodiedTool()
    result = await tool.execute(action="describe", target_action="record")
    assert "record" in result
    assert "dataset" in result.lower()


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
    assert "current setup" in result.lower()


@pytest.mark.asyncio
async def test_calibrate_all_arms() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("builtins.print") as mock_print,
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.setup.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert mock_runner.run_interactive.call_count == 2
    assert mock_mark.call_count == 2
    assert mock_print.call_args_list[0].args == ("\n=== Calibrating: right_follower ===",)


@pytest.mark.asyncio
async def test_calibrate_selected_arms_even_if_calibrated() -> None:
    setup = {
        **_MOCK_SETUP,
        "arms": [
            {**_MOCK_SETUP["arms"][0], "calibrated": True},
            _MOCK_SETUP["arms"][1],
        ],
    }
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("builtins.print"),
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.setup.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate", arms=_FOLLOWER_PORT)

    assert "1 succeeded, 0 failed." in result
    mock_mark.assert_called_once_with("right_follower")


@pytest.mark.asyncio
async def test_calibrate_no_arms() -> None:
    tool = EmbodiedTool()
    with patch("roboclaw.embodied.setup.ensure_setup", return_value={**_MOCK_SETUP, "arms": []}):
        result = await tool.execute(action="calibrate")
    assert result == "No arms configured."


@pytest.mark.asyncio
async def test_calibrate_missing_arm() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="calibrate", arms="missing_arm")
    assert result == "No arm with port 'missing_arm' found in setup."


@pytest.mark.asyncio
async def test_calibrate_interrupted_on_sigint() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.side_effect = [0, 130]

    with (
        patch("builtins.print"),
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.setup.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert result == "interrupted"
    mock_mark.assert_called_once_with("right_follower")


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
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "record"]
    assert "--robot.type=so101_follower" in argv
    assert "--teleop.type=so101_leader" in argv
    assert "--dataset.root=/data" in argv
    assert "--dataset.push_to_hub=false" in argv
    assert any("--robot.cameras=" in arg for arg in argv)


@pytest.mark.asyncio
async def test_record_action_without_cameras() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
            use_cameras=False,
        )

    argv = mock_runner.run_interactive.call_args.args[0]
    assert not any("--robot.cameras=" in arg for arg in argv)


@pytest.mark.asyncio
async def test_record_action_rejects_non_ascii_dataset_name() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(
            action="record",
            dataset_name="抓取任务",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )
    assert "dataset_name must be" in result


@pytest.mark.asyncio
async def test_record_bimanual() -> None:
    setup = {
        **_MOCK_SETUP,
        "arms": [
            {"alias": "left_f", "type": "so101_follower", "port": "/dev/a", "calibration_dir": "/c/5B14032630", "calibrated": True},
            {"alias": "right_f", "type": "so101_follower", "port": "/dev/b", "calibration_dir": "/c/5B14030892", "calibrated": True},
            {"alias": "left_l", "type": "so101_leader", "port": "/dev/c", "calibration_dir": "/c/5B14030001", "calibrated": True},
            {"alias": "right_l", "type": "so101_leader", "port": "/dev/d", "calibration_dir": "/c/5B14030002", "calibrated": True},
        ],
    }
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.tool.shutil.copy2") as mock_copy,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            arms="/dev/a,/dev/b,/dev/c,/dev/d",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--robot.id=bimanual" in argv
    assert "--teleop.id=bimanual" in argv
    assert "--dataset.root=/data" in argv
    assert len(mock_copy.call_args_list) == 4


@pytest.mark.asyncio
async def test_replay_single_uses_followers_only() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="replay", dataset_name="test", episode=2)

    assert "Replay finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "replay"]
    assert "--robot.type=so101_follower" in argv
    assert "--dataset.root=/data" in argv
    assert "--dataset.episode=2" in argv


@pytest.mark.asyncio
async def test_replay_bimanual_with_root_fallback() -> None:
    setup = {
        **_MOCK_SETUP,
        "datasets": {"root": ""},
        "arms": [
            {"alias": "left_f", "type": "so101_follower", "port": "/dev/a", "calibration_dir": "/c/5B14032630", "calibrated": True},
            {"alias": "right_f", "type": "so101_follower", "port": "/dev/b", "calibration_dir": "/c/5B14030892", "calibrated": True},
        ],
    }
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = 0

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.tool.shutil.copy2") as mock_copy,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="replay", dataset_name="test", arms="/dev/a,/dev/b")

    assert "Replay finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--robot.id=bimanual" in argv
    assert f"--dataset.root={Path('~/.cache/huggingface/lerobot').expanduser()}" in argv
    assert len(mock_copy.call_args_list) == 2


@pytest.mark.asyncio
async def test_replay_rejects_explicit_leaders() -> None:
    tool = EmbodiedTool(tty_handoff=AsyncMock())
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="replay", dataset_name="test", arms=_LEADER_PORT)
    assert "Replay only supports follower arms" in result


@pytest.mark.asyncio
async def test_teleoperate_bimanual() -> None:
    setup = {
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
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.tool.shutil.copy2") as mock_copy,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="teleoperate", arms="/dev/a,/dev/b,/dev/c,/dev/d")

    assert "Teleoperation finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--robot.id=bimanual" in argv
    assert "--teleop.id=bimanual" in argv
    assert len(mock_copy.call_args_list) == 4


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
async def test_run_policy_no_follower_arm() -> None:
    tool = EmbodiedTool()
    setup = {**_MOCK_SETUP, "arms": [{**_MOCK_SETUP["arms"][1]}]}

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="run_policy")

    assert result == "No follower arm configured."


@pytest.mark.asyncio
async def test_run_policy_requires_single_follower() -> None:
    tool = EmbodiedTool()
    setup = {
        **_MOCK_SETUP,
        "arms": [
            {**_MOCK_SETUP["arms"][0]},
            {**_MOCK_SETUP["arms"][0], "alias": "left_follower", "port": "/dev/ttyACM2", "calibration_dir": "/cal/f2"},
        ],
    }

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="run_policy")

    assert "exactly 1 follower arm" in result


@pytest.mark.asyncio
async def test_unknown_action() -> None:
    tool = EmbodiedTool()
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP):
        result = await tool.execute(action="fly_to_moon")
    assert "Unknown action" in result


@pytest.fixture()
def setup_file(tmp_path: Path) -> Path:
    p = tmp_path / "setup.json"
    base = {
        "version": 2,
        "arms": [],
        "cameras": {},
        "datasets": {"root": "/data"},
        "policies": {"root": "/policies"},
        "scanned_ports": [
            _MOCK_SCANNED_PORTS[0],
            _MOCK_SCANNED_PORTS[1],
        ],
        "scanned_cameras": [
            {"by_path": "/dev/v4l/by-path/cam0", "by_id": "usb-cam0", "dev": "/dev/video0"},
            {"by_path": "/dev/v4l/by-path/cam1", "by_id": "usb-cam1", "dev": "/dev/video2"},
        ],
    }
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_set_arm(setup_file: Path, calibration_root: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm is not None
    assert arm["type"] == "so101_follower"
    assert arm["port"] == _MOCK_SCANNED_PORTS[0]["by_id"]
    assert arm["calibration_dir"] == str(calibration_root / "5B14032630")
    assert arm["calibrated"] is False
    assert find_arm(load_setup(setup_file)["arms"], "my_follower") == arm


def test_set_arm_replaces_existing(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("my_arm", "so101_follower", "/dev/ttyACM0", path=setup_file)
        result = set_arm("my_arm", "so101_leader", "/dev/ttyACM1", path=setup_file)
    assert len(result["arms"]) == 1
    assert find_arm(result["arms"], "my_arm")["type"] == "so101_leader"


def test_set_arm_rejects_duplicate_port(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=setup_file)
        with pytest.raises(ValueError, match="already assigned"):
            set_arm("right_arm", "so101_leader", "/dev/ttyACM0", path=setup_file)


def test_set_arm_resolves_volatile_port(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_leader", "so101_leader", "/dev/ttyACM1", path=setup_file)
    assert find_arm(result["arms"], "my_leader")["port"] == _MOCK_SCANNED_PORTS[1]["by_id"]


def test_set_arm_keeps_stable_port(setup_file: Path) -> None:
    stable = "/dev/serial/by-id/usb-custom-device"
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", stable, path=setup_file)
    assert find_arm(result["arms"], "my_follower")["port"] == stable


def test_set_arm_unmatched_volatile_port(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyUSB99", path=setup_file)
    assert find_arm(result["arms"], "my_follower")["port"] == "/dev/ttyUSB99"


def test_set_arm_marks_existing_calibration(setup_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    (calibration_dir / f"{serial}.json").write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True


def test_set_arm_migrates_none_json(setup_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    legacy = calibration_dir / "None.json"
    target = calibration_dir / f"{serial}.json"
    legacy.write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=setup_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True
    assert not legacy.exists()
    assert target.exists()


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


def test_rename_arm_preserves_fields(setup_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    (calibration_dir / f"{serial}.json").write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("old_alias", "so101_follower", "/dev/ttyACM0", path=setup_file)
    result = rename_arm("old_alias", "new_alias", path=setup_file)
    arm = find_arm(result["arms"], "new_alias")
    assert arm["calibration_dir"] == str(calibration_dir)
    assert arm["calibrated"] is True
    assert find_arm(result["arms"], "old_alias") is None


def test_rename_arm_rejects_duplicate_alias(setup_file: Path) -> None:
    with std_patch("roboclaw.embodied.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=setup_file)
        set_arm("right_arm", "so101_leader", "/dev/ttyACM1", path=setup_file)
    with pytest.raises(ValueError, match="already exists"):
        rename_arm("left_arm", "right_arm", path=setup_file)


def test_set_camera(setup_file: Path) -> None:
    result = set_camera("front", 0, path=setup_file)
    cam = result["cameras"]["front"]
    assert cam["by_path"] == "/dev/v4l/by-path/cam0"
    assert cam["dev"] == "/dev/video0"
    assert cam["by_id"] == "usb-cam0"
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


def test_resolve_arms_single() -> None:
    result = _resolve_arms(_MOCK_SETUP, f"{_FOLLOWER_PORT},{_LEADER_PORT}")
    assert [arm["alias"] for arm in result] == ["right_follower", "left_leader"]


def test_resolve_arms_auto() -> None:
    result = _resolve_arms(_MOCK_SETUP, "")
    assert [arm["alias"] for arm in result] == ["right_follower", "left_leader"]


def test_resolve_arms_missing() -> None:
    with pytest.raises(ValueError, match="No arm with port 'missing'"):
        _resolve_arms(_MOCK_SETUP, "missing")


def test_resolve_arms_rejects_alias_lookup() -> None:
    with pytest.raises(ValueError, match="No arm with port 'right_follower'"):
        _resolve_arms(_MOCK_SETUP, "right_follower")


def test_group_arms() -> None:
    grouped = _group_arms(_resolve_arms(_MOCK_SETUP, f"{_FOLLOWER_PORT},{_LEADER_PORT}"))
    assert [arm["alias"] for arm in grouped["followers"]] == ["right_follower"]
    assert [arm["alias"] for arm in grouped["leaders"]] == ["left_leader"]
