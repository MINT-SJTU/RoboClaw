"""Tests for the EmbodiedToolGroup integration with the agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from unittest.mock import patch as std_patch

from roboclaw.embodied.manifest.helpers import (
    arm_display_name,
    find_arm,
    find_camera,
    find_hand,
    get_calibration_root,
    load_manifest,
    remove_arm,
    remove_camera,
    rename_arm,
    remove_hand,
    save_manifest,
    set_arm,
    set_camera,
    set_hand,
)
from roboclaw.embodied.engine.helpers import dataset_path, group_arms, _resolve_arms
from roboclaw.embodied.interface.serial import SerialInterface
from roboclaw.embodied.interface.video import VideoInterface
from roboclaw.embodied.sensor.camera import resolve_cameras as _resolve_cameras
from roboclaw.embodied.tool import EmbodiedToolGroup, create_embodied_tools

_MOCK_SCANNED_PORTS = [
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.1",
        by_id="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00",
        dev="/dev/ttyACM0",
    ),
    SerialInterface(
        by_path="/dev/serial/by-path/pci-0:2.2",
        by_id="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00",
        dev="/dev/ttyACM1",
    ),
]

_FOLLOWER_PORT = _MOCK_SCANNED_PORTS[0].by_id
_LEADER_PORT = _MOCK_SCANNED_PORTS[1].by_id

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
    "hands": [],
    "cameras": [
        {"alias": "front", "port": "/dev/video0", "width": 640, "height": 480, "fps": 30},
    ],
    "datasets": {"root": "/data"},
    "policies": {"root": "/policies"},
}


def _find_tool(tools: list[EmbodiedToolGroup], name: str) -> EmbodiedToolGroup:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"No tool named {name}")


@pytest.fixture(autouse=True)
def calibration_root(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    with (
        std_patch("roboclaw.embodied.manifest.helpers.get_calibration_root", return_value=root),
        std_patch("roboclaw.embodied.manifest.state.get_calibration_root", return_value=root),
    ):
        yield root


def test_create_embodied_tools_returns_six_groups() -> None:
    tools = create_embodied_tools()
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert names == {"embodied_setup", "embodied_hardware", "embodied_control", "embodied_replay", "embodied_train", "embodied_hand"}


def test_setup_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_setup")
    params = tool.parameters

    assert params["type"] == "object"
    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert "alias" in params["properties"]
    assert "port" in params["properties"]
    assert "arm_type" in params["properties"]
    assert "target_action" in params["properties"]
    assert "preview_cameras" in params["properties"]["action"]["enum"]
    # These params should NOT be in setup
    assert "dataset_name" not in params["properties"]
    assert "checkpoint_path" not in params["properties"]
    assert "arms" not in params["properties"]


def test_hardware_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_hardware")
    params = tool.parameters

    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert set(params["properties"].keys()) == {"action", "arms"}
    assert params["properties"]["action"]["enum"] == ["identify", "calibrate"]
    # Hardware should NOT have port parameter
    assert "port" not in params["properties"]


def test_control_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_control")
    params = tool.parameters

    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert "checkpoint_path" in params["properties"]
    assert "dataset_name" in params["properties"]
    assert params["properties"]["use_cameras"]["default"] is True
    assert params["properties"]["action"]["enum"] == ["teleoperate", "record"]


def test_replay_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_replay")
    params = tool.parameters

    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert "episode" in params["properties"]
    assert "dataset_name" in params["properties"]
    assert params["properties"]["action"]["enum"] == ["replay"]


def test_train_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_train")
    params = tool.parameters

    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert "steps" in params["properties"]
    assert "device" in params["properties"]
    assert "job_id" in params["properties"]
    assert params["properties"]["action"]["enum"] == ["train", "job_status", "list_datasets", "list_policies"]


def test_cross_group_isolation_hardware_rejects_port() -> None:
    """Hardware tool should not accept port parameter in its schema."""
    hw_tool = _find_tool(create_embodied_tools(), "embodied_hardware")
    assert "port" not in hw_tool.parameters["properties"]


@pytest.mark.asyncio
async def test_describe_action() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_setup")
    result = await tool.execute(action="describe", target_action="record")
    assert "record" in result
    assert "dataset" in result.lower()


@pytest.mark.asyncio
async def test_doctor_action() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_setup")
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (0, "lerobot 0.5.0", "")

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="doctor")

    assert "lerobot 0.5.0" in result
    assert "current setup" in result.lower()


@pytest.mark.asyncio
async def test_calibrate_all_arms() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_hardware")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.manifest.helpers.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert mock_runner.run_interactive.call_count == 2
    assert mock_mark.call_count == 2


@pytest.mark.asyncio
async def test_calibrate_selected_arms_even_if_calibrated() -> None:
    setup = {
        **_MOCK_SETUP,
        "arms": [
            {**_MOCK_SETUP["arms"][0], "calibrated": True},
            _MOCK_SETUP["arms"][1],
        ],
    }
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_hardware")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup),
        patch("roboclaw.embodied.manifest.helpers.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate", arms=_FOLLOWER_PORT)

    assert "1 succeeded, 0 failed." in result
    mock_mark.assert_called_once_with("right_follower")


@pytest.mark.asyncio
async def test_calibrate_no_arms() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_hardware")
    with patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value={**_MOCK_SETUP, "arms": []}):
        result = await tool.execute(action="calibrate")
    assert result == "No arms configured."


@pytest.mark.asyncio
async def test_calibrate_missing_arm() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_hardware")
    with patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP):
        result = await tool.execute(action="calibrate", arms="missing_arm")
    assert result == "No arm with port 'missing_arm' found in manifest."


@pytest.mark.asyncio
async def test_calibrate_interrupted_on_sigint() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_hardware")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.side_effect = [(0, ""), (130, "")]

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.manifest.helpers.mark_arm_calibrated") as mock_mark,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert result == "interrupted"
    mock_mark.assert_called_once_with("right_follower")


@pytest.mark.asyncio
async def test_record_action() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup, kwargs, tty_handoff):
        assert action == "record"
        assert kwargs.get("task") == "grasp"
        assert kwargs.get("num_episodes") == 5
        return "Recording finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result


@pytest.mark.asyncio
async def test_record_action_cli_reports_failure() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup, kwargs, tty_handoff):
        return "Record failed: Process exited with code 7\ncamera init failed"

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "failed" in result.lower()
    assert "camera init failed" in result


@pytest.mark.asyncio
async def test_record_action_without_cameras() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup, kwargs, tty_handoff):
        assert kwargs.get("use_cameras") is False
        return "Recording finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
            use_cameras=False,
        )

    assert "Recording finished" in result


@pytest.mark.asyncio
async def test_record_action_rejects_non_ascii_dataset_name() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup, kwargs, tty_handoff):
        return "Recording finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="\u6293\u53d6\u4efb\u52a1",
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
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup, kwargs, tty_handoff):
        assert action == "record"
        return "Recording finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            arms="/dev/a,/dev/b,/dev/c,/dev/d",
        )

    assert "Recording finished" in result


@pytest.mark.asyncio
async def test_replay_single_uses_followers_only() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_replay")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="replay", dataset_name="test", episode=2)

    assert "Replay finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "replay"]
    assert "--robot.type=so101_follower" in argv
    assert "--dataset.root=/data/local/test" in argv
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
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_replay")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup),
        patch("roboclaw.embodied.manifest.helpers.ensure_bimanual_cal_dir", return_value="/tmp/bimanual") as mock_cal,
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="replay", dataset_name="test", arms="/dev/a,/dev/b")

    assert "Replay finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--robot.id=bimanual" in argv
    fallback = Path("~/.cache/huggingface/lerobot").expanduser() / "local" / "test"
    assert f"--dataset.root={fallback}" in argv
    assert mock_cal.call_count == 1


@pytest.mark.asyncio
async def test_replay_rejects_explicit_leaders() -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_replay")
    with patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP):
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
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")

    async def fake_cli_session(service, action, setup_arg, kwargs, tty_handoff):
        assert action == "teleoperate"
        return "Teleoperation finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(action="teleoperate", arms="/dev/a,/dev/b,/dev/c,/dev/d")

    assert "Teleoperation finished" in result


@pytest.mark.asyncio
async def test_train_action() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_train")
    mock_runner = AsyncMock()
    mock_runner.run_detached.return_value = "job-abc-123"

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="train", dataset_name="test", steps=5000)

    assert "job-abc-123" in result
    argv = mock_runner.run_detached.call_args.kwargs["argv"]
    assert "--dataset.root=/data/local/test" in argv


@pytest.mark.asyncio
async def test_run_policy_no_follower_arm() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_control")
    setup = {**_MOCK_SETUP, "arms": [{**_MOCK_SETUP["arms"][1]}]}

    with patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup):
        result = await tool.execute(action="record", checkpoint_path="/models/act")

    assert result == "No follower arm configured."


@pytest.mark.asyncio
async def test_run_policy_bimanual() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_control")
    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=(0, "ok", ""))
    setup = {
        **_MOCK_SETUP,
        "arms": [
            {**_MOCK_SETUP["arms"][0]},
            {**_MOCK_SETUP["arms"][0], "alias": "left_follower", "port": "/dev/ttyACM2", "calibration_dir": "/cal/f2"},
        ],
    }

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=setup),
        patch("roboclaw.embodied.manifest.helpers.ensure_bimanual_cal_dir", return_value="/tmp/bimanual"),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="record", checkpoint_path="/models/act")

    argv = mock_runner.run.call_args[0][0]
    assert "--robot.type=bi_so_follower" in argv
    assert "--policy.path=/models/act" in argv


@pytest.mark.asyncio
async def test_unknown_action_in_group() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_setup")
    result = await tool.execute(action="fly_to_moon")
    assert "Unknown action" in result


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "setup.json"
    base = {
        "version": 2,
        "arms": [],
        "hands": [],
        "cameras": [],
        "datasets": {"root": "/data"},
        "policies": {"root": "/policies"},
    }
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_set_arm(manifest_file: Path, calibration_root: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm is not None
    assert arm["type"] == "so101_follower"
    assert arm["port"] == _MOCK_SCANNED_PORTS[0].by_id
    assert arm["calibration_dir"] == str(calibration_root / "5B14032630")
    assert arm["calibrated"] is False
    assert find_arm(load_manifest(manifest_file)["arms"], "my_follower") == arm


def test_set_arm_replaces_existing(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("my_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        result = set_arm("my_arm", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    assert len(result["arms"]) == 1
    assert find_arm(result["arms"], "my_arm")["type"] == "so101_leader"


def test_set_arm_rejects_duplicate_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        with pytest.raises(ValueError, match="already assigned"):
            set_arm("right_arm", "so101_leader", "/dev/ttyACM0", path=manifest_file)


def test_set_arm_resolves_volatile_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_leader", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    assert find_arm(result["arms"], "my_leader")["port"] == _MOCK_SCANNED_PORTS[1].by_id


def test_set_arm_keeps_stable_port(manifest_file: Path) -> None:
    stable = "/dev/serial/by-id/usb-custom-device"
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", stable, path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["port"] == stable


def test_set_arm_unmatched_volatile_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyUSB99", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["port"] == "/dev/ttyUSB99"


def test_set_arm_marks_existing_calibration(manifest_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    (calibration_dir / f"{serial}.json").write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True


def test_set_arm_migrates_none_json(manifest_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    legacy = calibration_dir / "None.json"
    target = calibration_dir / f"{serial}.json"
    legacy.write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True
    assert not legacy.exists()
    assert target.exists()


def test_set_arm_invalid_type(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid arm_type"):
            set_arm("my_follower", "bogus_arm", "/dev/ttyACM0", path=manifest_file)


def test_set_arm_empty_alias(manifest_file: Path) -> None:
    with pytest.raises(ValueError, match="Arm alias is required"):
        set_arm("", "so101_follower", "/dev/ttyACM0", path=manifest_file)


def test_remove_arm(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    result = remove_arm("my_follower", path=manifest_file)
    assert find_arm(result["arms"], "my_follower") is None


def test_remove_arm_missing(manifest_file: Path) -> None:
    with pytest.raises(ValueError, match="No arm with alias"):
        remove_arm("nonexistent", path=manifest_file)


def test_rename_arm_preserves_fields(manifest_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    (calibration_dir / f"{serial}.json").write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("old_alias", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    result = rename_arm("old_alias", "new_alias", path=manifest_file)
    arm = find_arm(result["arms"], "new_alias")
    assert arm["calibration_dir"] == str(calibration_dir)
    assert arm["calibrated"] is True
    assert find_arm(result["arms"], "old_alias") is None


def test_rename_arm_rejects_duplicate_alias(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        set_arm("right_arm", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    with pytest.raises(ValueError, match="already exists"):
        rename_arm("left_arm", "right_arm", path=manifest_file)


_MOCK_SCANNED_CAMERAS = [
    VideoInterface(by_path="/dev/v4l/by-path/cam0", by_id="usb-cam0", dev="/dev/video0", width=640, height=480),
    VideoInterface(by_path="/dev/v4l/by-path/cam1", by_id="usb-cam1", dev="/dev/video2", width=320, height=240),
]


def test_set_camera(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
        result = set_camera("front", 0, path=manifest_file)
    cam = find_camera(result["cameras"], "front")
    assert cam is not None
    assert cam["alias"] == "front"
    assert cam["port"] == "/dev/v4l/by-path/cam0"
    assert cam["width"] == 640
    assert cam["height"] == 480
    assert cam["fps"] == 30


@pytest.mark.asyncio
async def test_preview_cameras_action() -> None:
    previews = [{"camera": "/dev/v4l/by-path/cam0", "image_path": "/tmp/front.jpg"}]
    tool = _find_tool(create_embodied_tools(), "embodied_setup")

    with (
        patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=[VideoInterface(dev="/dev/video0", width=640, height=480, fps=30)]),
        patch("roboclaw.embodied.hardware.scan.capture_camera_frames", return_value=previews) as mock_capture,
        patch("pathlib.Path.is_file", return_value=False),
    ):
        result = await tool.execute(action="preview_cameras")

    # Returns multimodal content blocks (list) with text summary
    assert isinstance(result, list)
    text_blocks = [b for b in result if b.get("type") == "text"]
    assert any("Detected 1 camera" in b["text"] for b in text_blocks)
    output_dir = mock_capture.call_args.args[1]
    assert output_dir == Path("~/.roboclaw").expanduser() / "workspace" / "embodied" / "camera_previews"


def test_set_camera_bad_index(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
        with pytest.raises(ValueError, match="out of range"):
            set_camera("front", 99, path=manifest_file)


def test_remove_camera(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
        set_camera("front", 0, path=manifest_file)
    result = remove_camera("front", path=manifest_file)
    assert find_camera(result["cameras"], "front") is None


def test_remove_camera_missing(manifest_file: Path) -> None:
    with pytest.raises(ValueError, match="No camera with alias"):
        remove_camera("nonexistent", path=manifest_file)


def test_validation_rejects_unknown_arm_fields(manifest_file: Path) -> None:
    bad = load_manifest(manifest_file)
    bad["arms"] = [{"alias": "x", "type": "so101_follower", "port": "/dev/x", "junk": True}]
    with pytest.raises(ValueError, match="unknown fields"):
        save_manifest(bad, manifest_file)


def test_validation_rejects_unknown_camera_fields(manifest_file: Path) -> None:
    bad = load_manifest(manifest_file)
    bad["cameras"] = [{"alias": "front", "port": "/dev/video0", "junk": True}]
    with pytest.raises(ValueError, match="unknown fields"):
        save_manifest(bad, manifest_file)


def test_validation_rejects_bad_arm_type(manifest_file: Path) -> None:
    bad = load_manifest(manifest_file)
    bad["arms"] = [{"alias": "x", "type": "garbage", "port": "/dev/x"}]
    with pytest.raises(ValueError, match="invalid type"):
        save_manifest(bad, manifest_file)


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
    grouped = group_arms(_resolve_arms(_MOCK_SETUP, f"{_FOLLOWER_PORT},{_LEADER_PORT}"))
    assert [arm["alias"] for arm in grouped["followers"]] == ["right_follower"]
    assert [arm["alias"] for arm in grouped["leaders"]] == ["left_leader"]


# ── Serial number extraction test ────────────────────────────────────


def test_calibration_dir_uses_serial_number(manifest_file: Path, calibration_root: Path) -> None:
    """calibration_dir should be based on serial number extracted from by_id port."""
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm["calibration_dir"] == str(calibration_root / "5B14032630")


# ── set_hand / remove_hand / find_hand tests ─────────────────────────


def test_set_hand(manifest_file: Path) -> None:
    with (
        std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS),
        std_patch("roboclaw.embodied.manifest.helpers._probe_hand_slave_id", return_value=1),
    ):
        result = set_hand("left_hand", "inspire_rh56", "/dev/ttyACM0", path=manifest_file)
    hand = find_hand(result["hands"], "left_hand")
    assert hand is not None
    assert hand["type"] == "inspire_rh56"
    assert hand["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00"
    assert hand["slave_id"] == 1
    assert "calibration_dir" not in hand
    assert "calibrated" not in hand
    persisted = load_manifest(manifest_file)
    assert find_hand(persisted["hands"], "left_hand") == hand


def test_set_hand_replaces_existing(manifest_file: Path) -> None:
    with (
        std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS),
        std_patch("roboclaw.embodied.manifest.helpers._probe_hand_slave_id", return_value=1),
    ):
        set_hand("h", "inspire_rh56", "/dev/ttyACM0", path=manifest_file)
        result = set_hand("h", "revo2", "/dev/ttyACM1", path=manifest_file)
    assert len(result["hands"]) == 1
    assert find_hand(result["hands"], "h")["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00"


def test_set_hand_invalid_type(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid hand_type"):
            set_hand("h", "so101_follower", "/dev/ttyUSB0", path=manifest_file)


def test_remove_hand(manifest_file: Path) -> None:
    with (
        std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]),
        std_patch("roboclaw.embodied.manifest.helpers._probe_hand_slave_id", return_value=1),
    ):
        set_hand("left_hand", "inspire_rh56", "/dev/ttyUSB0", path=manifest_file)
    result = remove_hand("left_hand", path=manifest_file)
    assert find_hand(result["hands"], "left_hand") is None


def test_remove_hand_missing(manifest_file: Path) -> None:
    with pytest.raises(ValueError, match="No hand with alias"):
        remove_hand("nonexistent", path=manifest_file)


def test_find_hand() -> None:
    hands = [
        {"alias": "left", "type": "inspire_rh56", "port": "/dev/ttyUSB0", "slave_id": 1},
        {"alias": "right", "type": "revo2", "port": "/dev/ttyUSB1", "slave_id": 126},
    ]
    assert find_hand(hands, "left") == hands[0]
    assert find_hand(hands, "right") == hands[1]
    assert find_hand(hands, "none") is None
    assert find_hand([], "left") is None


def test_set_arm_rejects_revo2(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid arm_type"):
            set_arm("h", "revo2", "/dev/ttyUSB0", path=manifest_file)


# ── embodied_hand tool group tests ────────────────────────────────────


def test_hand_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodied_hand")
    params = tool.parameters

    assert params["type"] == "object"
    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert set(params["properties"]["action"]["enum"]) == {
        "hand_open", "hand_close", "hand_pose", "hand_status",
    }
    assert "hand_name" in params["properties"]
    assert "positions" in params["properties"]
    # Hand-specific params should NOT leak into other groups
    assert "alias" not in params["properties"]
    assert "arm_type" not in params["properties"]


def test_train_schema_has_no_hand_params() -> None:
    """hand_type, hand_name, positions should NOT appear in embodied_train."""
    tool = _find_tool(create_embodied_tools(), "embodied_train")
    props = tool.parameters["properties"]
    assert "hand_type" not in props
    assert "hand_name" not in props
    assert "positions" not in props


def test_setup_schema_has_no_hand_runtime_params() -> None:
    """hand_name and positions should NOT appear in embodied_setup."""
    tool = _find_tool(create_embodied_tools(), "embodied_setup")
    props = tool.parameters["properties"]
    assert "hand_name" not in props
    assert "positions" not in props
    # hand_type SHOULD be in setup (for set_hand)
    assert "hand_type" in props
    assert props["hand_type"]["enum"] == ["inspire_rh56", "revo2"]


def test_resolve_cameras_defaults_and_passthrough() -> None:
    setup = {
        "cameras": [
            {"alias": "front", "port": "/dev/video0", "width": 640, "height": 480},
            {"alias": "side", "port": "/dev/video1", "width": 320, "height": 240, "fps": 15},
            {"alias": "dv20", "port": "/dev/video2", "width": 640, "height": 480, "fourcc": "MJPG"},
        ],
    }
    cameras = _resolve_cameras(setup)
    assert cameras["front"]["fps"] == 30
    assert "fourcc" not in cameras["front"]
    assert cameras["side"]["fps"] == 15
    assert cameras["dv20"]["fourcc"] == "MJPG"
    assert cameras["dv20"]["fps"] == 30


def test_dataset_path_appends_local_and_dataset_name() -> None:
    assert dataset_path(_MOCK_SETUP, "demo") == Path("/data/local/demo")
