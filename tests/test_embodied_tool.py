"""Tests for the EmbodiedToolGroup integration with the agent."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from unittest.mock import patch as std_patch

from roboclaw.embodied.manifest import Manifest
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


def _manifest_from_data(tmp_path: Path, data: dict) -> Manifest:
    path = tmp_path / "manifest.json"
    save_manifest(copy.deepcopy(data), path)
    return Manifest(path=path)


@pytest.fixture(autouse=True)
def calibration_root(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    with (
        std_patch("roboclaw.embodied.manifest.helpers.get_calibration_root", return_value=root),
        std_patch("roboclaw.embodied.manifest.state.get_calibration_root", return_value=root),
    ):
        yield root


def test_create_embodied_tools_returns_ten_groups() -> None:
    tools = create_embodied_tools()
    assert len(tools) == 10
    names = {t.name for t in tools}
    assert names == {
        "manifest",
        "setup",
        "doctor",
        "calibration",
        "teleop",
        "record",
        "replay",
        "train",
        "infer",
        "embodiment_control",
    }


@pytest.mark.parametrize(
    ("tool_name", "expected_actions", "included", "excluded"),
    [
        ("manifest", _MANIFEST_ACTIONS := {
            "status", "bind_arm", "unbind_arm", "rename_arm", "bind_camera",
            "unbind_camera", "rename_camera", "bind_hand", "unbind_hand",
            "rename_hand", "describe",
        }, {"alias", "arm_type", "port", "camera_name", "camera_index", "hand_type", "target_action", "new_alias"}, {"arms", "dataset_name", "positions"}),
        ("setup", {"scan", "identify", "preview_cameras"}, {"model"}, {"alias", "arms", "dataset_name"}),
        ("doctor", {"check"}, set(), {"alias", "arms", "dataset_name"}),
        ("calibration", {"calibrate"}, {"arms"}, {"port", "dataset_name", "positions"}),
        ("teleop", {"teleoperate"}, {"arms", "fps"}, {"dataset_name", "checkpoint_path", "positions"}),
        ("record", {"record"}, {"arms", "dataset_name", "task", "num_episodes", "fps", "episode_time_s", "reset_time_s", "use_cameras"}, {"checkpoint_path", "positions"}),
        ("replay", {"replay"}, {"arms", "dataset_name", "episode", "fps"}, {"checkpoint_path", "positions"}),
        ("train", {"train", "job_status", "list_datasets", "list_policies"}, {"dataset_name", "steps", "device", "job_id"}, {"positions", "port"}),
        ("infer", {"run_policy"}, {"arms", "dataset_name", "source_dataset", "checkpoint_path", "task", "num_episodes", "use_cameras"}, {"positions", "port"}),
        ("embodiment_control", {"hand_open", "hand_close", "hand_pose", "hand_status"}, {"hand_name", "positions"}, {"dataset_name", "arms", "port"}),
    ],
)
def test_tool_group_schemas(
    tool_name: str,
    expected_actions: set[str],
    included: set[str],
    excluded: set[str],
) -> None:
    tool = _find_tool(create_embodied_tools(), tool_name)
    params = tool.parameters
    props = params["properties"]

    assert params["type"] == "object"
    assert params["required"] == ["action"]
    assert params["additionalProperties"] is False
    assert set(props["action"]["enum"]) == expected_actions
    for name in included:
        assert name in props
    for name in excluded:
        assert name not in props


@pytest.mark.asyncio
async def test_manifest_describe_action() -> None:
    tool = _find_tool(create_embodied_tools(), "manifest")
    result = await tool.execute(action="describe", target_action="record")
    assert "record" in result
    assert "dataset" in result.lower()


@pytest.mark.asyncio
async def test_manifest_status_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "manifest")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService

    tool.embodied_service = EmbodiedService(manifest=manifest)
    result = await tool.execute(action="status")

    payload = json.loads(result)
    assert "status" in payload
    assert payload["status"]["arms"][0]["alias"] == "right_follower"


@pytest.mark.asyncio
async def test_doctor_check_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "doctor")
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (0, "lerobot 0.5.0", "")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="check")

    assert "lerobot 0.5.0" in result
    assert "current setup" in result.lower()


@pytest.mark.asyncio
async def test_setup_scan_requires_model() -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    result = await tool.execute(action="scan")
    assert "requires model" in result


@pytest.mark.asyncio
async def test_setup_scan_action() -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    with patch(
        "roboclaw.embodied.service.setup_session.HardwareDiscovery.discover",
        return_value=_MOCK_SCANNED_PORTS,
    ), patch(
        "roboclaw.embodied.service.setup_session.HardwareDiscovery.discover_cameras",
        return_value=[VideoInterface(dev="/dev/video0", width=640, height=480, fps=30)],
    ):
        result = await tool.execute(action="scan", model="so101")

    assert "Found 2 serial port(s) and 1 camera(s)." in result


@pytest.mark.asyncio
async def test_setup_preview_cameras_action() -> None:
    previews = [{"camera": "/dev/v4l/by-path/cam0", "image_path": "/tmp/front.jpg"}]
    tool = _find_tool(create_embodied_tools(), "setup")

    with (
        patch("roboclaw.embodied.hardware.scan.scan_cameras", return_value=[VideoInterface(dev="/dev/video0", width=640, height=480, fps=30)]),
        patch("roboclaw.embodied.hardware.scan.capture_camera_frames", return_value=previews) as mock_capture,
        patch("pathlib.Path.is_file", return_value=False),
    ):
        result = await tool.execute(action="preview_cameras")

    assert isinstance(result, list)
    text_blocks = [block for block in result if block.get("type") == "text"]
    assert any("Detected 1 camera" in block["text"] for block in text_blocks)
    output_dir = mock_capture.call_args.args[1]
    assert output_dir == Path("~/.roboclaw").expanduser() / "workspace" / "embodied" / "camera_previews"


@pytest.mark.asyncio
async def test_calibration_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "calibration")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert mock_runner.run_interactive.call_count == 2


@pytest.mark.asyncio
async def test_teleop_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "teleop")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    async def fake_cli_session(service, action, setup_arg, kwargs, tty_handoff):
        assert action == "teleoperate"
        assert kwargs["fps"] == 20
        return "Teleoperation finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(action="teleoperate", fps=20, arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}")

    assert result == "Teleoperation finished."


@pytest.mark.asyncio
async def test_record_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "record")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    async def fake_cli_session(service, action, setup_arg, kwargs, tty_handoff):
        assert action == "record"
        assert kwargs["dataset_name"] == "test"
        return "Recording finished."

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.adapters.cli.run_cli_session", side_effect=fake_cli_session),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert result == "Recording finished."


@pytest.mark.asyncio
async def test_replay_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "replay")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="replay", dataset_name="test", episode=2)

    assert result == "Replay finished."
    argv = mock_runner.run_interactive.call_args.args[0]
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "replay"]
    assert "--dataset.episode=2" in argv


@pytest.mark.asyncio
async def test_train_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "train")
    mock_runner = AsyncMock()
    mock_runner.run_detached.return_value = "job-abc-123"
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="train", dataset_name="test", steps=5000)

    assert "job-abc-123" in result


@pytest.mark.asyncio
async def test_infer_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "infer")
    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=(0, "ok", ""))
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        await tool.execute(action="run_policy", checkpoint_path="/models/act")

    argv = mock_runner.run.call_args[0][0]
    assert "--policy.path=/models/act" in argv


@pytest.mark.asyncio
async def test_embodiment_control_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "embodiment_control")
    manifest = _manifest_from_data(
        tmp_path,
        {
            **_MOCK_SETUP,
            "hands": [{"alias": "left_hand", "type": "inspire_rh56", "port": _FOLLOWER_PORT, "slave_id": 1}],
        },
    )

    with (
        patch("roboclaw.embodied.manifest.helpers.ensure_manifest", return_value=manifest),
        patch("roboclaw.embodied.service.hand_session.HandSession._get_hand_controller") as mock_controller,
    ):
        mock_controller.return_value.open_hand = AsyncMock(return_value="opened")
        result = await tool.execute(action="hand_open", hand_name="left_hand")

    assert result == "opened"


@pytest.mark.asyncio
async def test_unknown_action_in_group() -> None:
    tool = _find_tool(create_embodied_tools(), "manifest")
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
    tool = _find_tool(create_embodied_tools(), "setup")

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


def test_resolve_arms_single(tmp_path: Path) -> None:
    result = _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), f"{_FOLLOWER_PORT},{_LEADER_PORT}")
    assert [arm.alias for arm in result] == ["right_follower", "left_leader"]


def test_resolve_arms_auto(tmp_path: Path) -> None:
    result = _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), "")
    assert [arm.alias for arm in result] == ["right_follower", "left_leader"]


def test_resolve_arms_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No arm with port 'missing'"):
        _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), "missing")


def test_resolve_arms_rejects_alias_lookup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No arm with port 'right_follower'"):
        _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), "right_follower")


def test_group_arms(tmp_path: Path) -> None:
    grouped = group_arms(
        _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), f"{_FOLLOWER_PORT},{_LEADER_PORT}")
    )
    assert [arm.alias for arm in grouped["followers"]] == ["right_follower"]
    assert [arm.alias for arm in grouped["leaders"]] == ["left_leader"]


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


# ── embodiment_control tool group tests ───────────────────────────────


def test_hand_tool_schema() -> None:
    tool = _find_tool(create_embodied_tools(), "embodiment_control")
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
    """hand_type, hand_name, positions should NOT appear in train."""
    tool = _find_tool(create_embodied_tools(), "train")
    props = tool.parameters["properties"]
    assert "hand_type" not in props
    assert "hand_name" not in props
    assert "positions" not in props


def test_setup_schema_has_no_hand_runtime_params() -> None:
    """hand_name and positions should NOT appear in manifest."""
    tool = _find_tool(create_embodied_tools(), "manifest")
    props = tool.parameters["properties"]
    assert "hand_name" not in props
    assert "positions" not in props
    # hand_type SHOULD be in manifest (for bind_hand)
    assert "hand_type" in props
    assert props["hand_type"]["enum"] == ["inspire_rh56", "revo2"]


def test_resolve_cameras_defaults_and_passthrough(tmp_path: Path) -> None:
    setup = {
        "cameras": [
            {"alias": "front", "port": "/dev/video0", "width": 640, "height": 480},
            {"alias": "side", "port": "/dev/video1", "width": 320, "height": 240, "fps": 15},
            {"alias": "dv20", "port": "/dev/video2", "width": 640, "height": 480, "fourcc": "MJPG"},
        ],
    }
    cameras = _resolve_cameras(_manifest_from_data(tmp_path, {**_MOCK_SETUP, **setup}).cameras)
    assert cameras["front"]["fps"] == 30
    assert "fourcc" not in cameras["front"]
    assert cameras["side"]["fps"] == 15
    assert cameras["dv20"]["fourcc"] == "MJPG"
    assert cameras["dv20"]["fps"] == 30


def test_dataset_path_appends_local_and_dataset_name(tmp_path: Path) -> None:
    assert dataset_path(_manifest_from_data(tmp_path, _MOCK_SETUP), "demo") == Path("/data/local/demo")
