"""Tests for the EmbodiedToolGroup integration with the agent."""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from unittest.mock import patch as std_patch

if "roboclaw.agent" not in sys.modules:
    agent_pkg = types.ModuleType("roboclaw.agent")
    agent_pkg.__path__ = []
    sys.modules["roboclaw.agent"] = agent_pkg

    tools_pkg = types.ModuleType("roboclaw.agent.tools")
    tools_pkg.__path__ = []
    sys.modules["roboclaw.agent.tools"] = tools_pkg

    base_mod = types.ModuleType("roboclaw.agent.tools.base")

    class Tool:
        pass

    base_mod.Tool = Tool
    sys.modules["roboclaw.agent.tools.base"] = base_mod

from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.helpers import (
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
from roboclaw.embodied.command.helpers import (
    ActionError,
    dataset_path,
    group_arms,
    resolve_action_arms as _resolve_arms,
    resolve_bimanual_pair,
)
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.command.helpers import resolve_cameras as _resolve_cameras
from roboclaw.embodied.toolkit.tools import EmbodiedToolGroup, create_embodied_tools

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


def _with_temp_calibration_dirs(tmp_path: Path, data: dict) -> dict:
    updated = copy.deepcopy(data)
    for idx, arm in enumerate(updated["arms"], start=1):
        cal_dir = tmp_path / "calibration" / f"SIM{idx:03d}"
        cal_dir.mkdir(parents=True, exist_ok=True)
        arm["calibration_dir"] = str(cal_dir)
        arm["calibrated"] = False
    return updated


@pytest.fixture(autouse=True)
def calibration_root(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    with (
        std_patch("roboclaw.embodied.embodiment.manifest.helpers.get_calibration_root", return_value=root),
        std_patch("roboclaw.embodied.embodiment.manifest.state.get_calibration_root", return_value=root),
        std_patch("roboclaw.embodied.embodiment.lock.get_roboclaw_home", return_value=tmp_path),
    ):
        yield root


def test_create_embodied_tools_returns_nine_groups() -> None:
    tools = create_embodied_tools()
    assert len(tools) == 9
    names = {t.name for t in tools}
    assert names == {
        "setup", "doctor", "calibration", "teleop",
        "record", "replay", "train", "infer", "hub",
    }


@pytest.mark.parametrize(
    ("tool_name", "expected_actions", "included", "excluded"),
    [
        ("setup", {"identify", "modify", "preview_cameras"}, {"model", "target", "operation", "alias", "new_alias", "arm_type", "port", "dev"}, {"arms", "dataset_name", "positions"}),
        ("doctor", {"check"}, set(), {"alias", "arms", "dataset_name"}),
        ("calibration", {"calibrate"}, {"arms"}, {"port", "dataset_name", "positions"}),
        ("teleop", {"teleoperate"}, {"arms", "fps"}, {"dataset_name", "checkpoint_path", "positions"}),
        ("record", {"record"}, {"arms", "dataset_name", "task", "num_episodes", "fps", "episode_time_s", "reset_time_s", "use_cameras"}, {"checkpoint_path", "positions"}),
        ("replay", {"replay"}, {"arms", "dataset_name", "episode", "fps"}, {"checkpoint_path", "positions"}),
        ("train", {"train", "job_status", "list_datasets", "list_policies"}, {"dataset_name", "steps", "device", "job_id"}, {"positions", "port"}),
        ("infer", {"run_policy"}, {"arms", "dataset_name", "source_dataset", "checkpoint_path", "task", "num_episodes", "use_cameras"}, {"positions", "port"}),
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
    assert set(props["action"]["enum"]) == expected_actions
    for name in included:
        assert name in props
    for name in excluded:
        assert name not in props


@pytest.mark.asyncio
async def test_doctor_check_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "doctor")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    svc = EmbodiedService(manifest=manifest)
    tool.embodied_service = svc

    result = await tool.execute(action="check")

    payload = json.loads(result)
    assert "environment" in payload
    assert "manifest" in payload
    assert "hardware_status" in payload


@pytest.mark.asyncio
async def test_calibration_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "calibration")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")
    manifest = _manifest_from_data(tmp_path, _with_temp_calibration_dirs(tmp_path, _MOCK_SETUP))
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    async def fake_run(argv):
        port_arg = next(arg for arg in argv if arg.startswith("--") and ".port=" in arg)
        port = port_arg.split("=", 1)[1]
        arm = next(arm for arm in manifest.arms if arm.port == port)
        cal_path = Path(arm.calibration_dir) / f"{Path(arm.calibration_dir).name}.json"
        cal_path.write_text("{}", encoding="utf-8")
        return (0, "")

    mock_runner.run_interactive.side_effect = fake_run

    with patch("roboclaw.embodied.executor.SubprocessExecutor", return_value=mock_runner):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert mock_runner.run_interactive.call_count == 2
    assert all(arm.calibrated for arm in manifest.arms)


@pytest.mark.asyncio
async def test_calibration_action_fails_without_saved_file(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "calibration")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")
    manifest = _manifest_from_data(tmp_path, _with_temp_calibration_dirs(tmp_path, _MOCK_SETUP))
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with patch("roboclaw.embodied.executor.SubprocessExecutor", return_value=mock_runner):
        result = await tool.execute(action="calibrate")

    assert "0 succeeded, 2 failed." in result
    assert "did not save" in result
    assert not any(arm.calibrated for arm in manifest.arms)


@pytest.mark.asyncio
async def test_teleop_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "teleop")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    async def fake_tty_run(self, session):
        assert session._kwargs["fps"] == 20
        return "Teleoperation finished."

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(action="teleoperate", fps=20, arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}")

    assert result == "Teleoperation finished."


@pytest.mark.asyncio
async def test_record_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "record")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    async def fake_tty_run(self, session):
        assert session._kwargs["dataset_name"] == "test"
        return "Recording finished."

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(
            action="record",
            dataset_name="test",
            task="grasp",
            num_episodes=5,
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert result == "Recording finished."


@pytest.mark.asyncio
async def test_replay_action_delegates_to_service_run_replay(tmp_path: Path) -> None:
    tty_handoff = AsyncMock()
    tool = _find_tool(create_embodied_tools(tty_handoff=tty_handoff), "replay")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    run_replay = AsyncMock(return_value="Replay finished.")
    tool.embodied_service = SimpleNamespace(
        manifest=manifest,
        run_replay=run_replay,
    )

    result = await tool.execute(
        action="replay",
        dataset_name="test",
        episode=2,
        fps=12,
        arms=_FOLLOWER_PORT,
    )

    assert result == "Replay finished."
    run_replay.assert_awaited_once_with(
        dataset_name="test",
        episode=2,
        fps=12,
        arms=_FOLLOWER_PORT,
        tty_handoff=tty_handoff,
    )


@pytest.mark.asyncio
async def test_train_action(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "train")
    mock_runner = AsyncMock()
    mock_runner.run_detached.return_value = "job-abc-123"
    data = copy.deepcopy(_MOCK_SETUP)
    data["datasets"]["root"] = str(tmp_path / "datasets")
    data["policies"]["root"] = str(tmp_path / "policies")
    dataset_meta = tmp_path / "datasets" / "local" / "test" / "meta"
    dataset_meta.mkdir(parents=True)
    (dataset_meta / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 2, "fps": 30}),
        encoding="utf-8",
    )
    manifest = _manifest_from_data(tmp_path, data)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with patch("roboclaw.embodied.executor.SubprocessExecutor", return_value=mock_runner):
        result = await tool.execute(action="train", dataset_name="test", steps=5000)

    assert "job-abc-123" in result


@pytest.mark.asyncio
async def test_infer_action_delegates_to_service_run_inference(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "infer")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    run_inference = AsyncMock(return_value="Inference finished.")
    tool.embodied_service = SimpleNamespace(
        manifest=manifest,
        run_inference=run_inference,
    )

    result = await tool.execute(
        action="run_policy",
        checkpoint_path="/models/act",
        source_dataset="pick_cube",
        dataset_name="eval_run",
        task="eval_pick",
        num_episodes=3,
        use_cameras=False,
        arms=_FOLLOWER_PORT,
    )

    assert result == "Inference finished."
    run_inference.assert_awaited_once_with(
        checkpoint_path="/models/act",
        source_dataset="pick_cube",
        dataset_name="eval_run",
        task="eval_pick",
        num_episodes=3,
        episode_time_s=60,
        arms=_FOLLOWER_PORT,
        use_cameras=False,
        tty_handoff=None,
    )


@pytest.mark.asyncio
async def test_unknown_action_in_group() -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    result = await tool.execute(action="fly_to_moon")
    assert "Unknown action" in result


@pytest.mark.asyncio
async def test_setup_modify_rename_arm(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    svc = EmbodiedService(manifest=manifest)
    tool.embodied_service = svc

    result = await tool.execute(
        action="modify", target="arm", operation="rename",
        alias="right_follower", new_alias="my_follower",
    )

    assert "renamed" in result.lower()
    assert "my_follower" in result


@pytest.mark.asyncio
async def test_setup_modify_unbind_arm(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    svc = EmbodiedService(manifest=manifest)
    tool.embodied_service = svc

    result = await tool.execute(
        action="modify", target="arm", operation="unbind", alias="right_follower",
    )

    assert "removed" in result.lower()


@pytest.mark.asyncio
async def test_setup_modify_requires_params(tmp_path: Path) -> None:
    tool = _find_tool(create_embodied_tools(), "setup")
    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    svc = EmbodiedService(manifest=manifest)
    tool.embodied_service = svc

    result = await tool.execute(action="modify")
    assert "requires" in result.lower()


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
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm is not None
    assert arm["type"] == "so101_follower"
    assert arm["port"] == _MOCK_SCANNED_PORTS[0].by_id
    assert arm["calibration_dir"] == str(calibration_root / "5B14032630")
    assert arm["calibrated"] is False
    assert find_arm(load_manifest(manifest_file)["arms"], "my_follower") == arm


def test_set_arm_stores_side(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("left_follower", "so101_follower", "/dev/ttyACM0", side="left", path=manifest_file)
    assert find_arm(result["arms"], "left_follower")["side"] == "left"


def test_set_arm_replaces_existing(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("my_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        result = set_arm("my_arm", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    assert len(result["arms"]) == 1
    assert find_arm(result["arms"], "my_arm")["type"] == "so101_leader"


def test_set_arm_rejects_duplicate_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        with pytest.raises(ValueError, match="already assigned"):
            set_arm("right_arm", "so101_leader", "/dev/ttyACM0", path=manifest_file)


def test_set_arm_rejects_ambiguous_bimanual_and_rolls_back(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("arm_a", "so101_follower", "/dev/ttyACM0", side="left", path=manifest_file)
        with pytest.raises(ValueError, match="one 'left' arm and one 'right' arm"):
            set_arm("arm_b", "so101_follower", "/dev/ttyACM1", path=manifest_file)
    arms = load_manifest(manifest_file)["arms"]
    assert [arm["alias"] for arm in arms] == ["arm_a"]


def test_set_arm_resolves_volatile_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_leader", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    assert find_arm(result["arms"], "my_leader")["port"] == _MOCK_SCANNED_PORTS[1].by_id


def test_set_arm_keeps_stable_port(manifest_file: Path) -> None:
    stable = "/dev/serial/by-id/usb-custom-device"
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", stable, path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["port"] == stable


def test_set_arm_unmatched_volatile_port(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyUSB99", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["port"] == "/dev/ttyUSB99"


def test_set_arm_marks_existing_calibration(manifest_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    (calibration_dir / f"{serial}.json").write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True


def test_set_arm_migrates_none_json(manifest_file: Path, calibration_root: Path) -> None:
    serial = "5B14032630"
    calibration_dir = calibration_root / serial
    calibration_dir.mkdir(parents=True)
    legacy = calibration_dir / "None.json"
    target = calibration_dir / f"{serial}.json"
    legacy.write_text("{}", encoding="utf-8")
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    assert find_arm(result["arms"], "my_follower")["calibrated"] is True
    assert not legacy.exists()
    assert target.exists()


def test_set_arm_invalid_type(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid arm_type"):
            set_arm("my_follower", "bogus_arm", "/dev/ttyACM0", path=manifest_file)


def test_set_arm_empty_alias(manifest_file: Path) -> None:
    with pytest.raises(ValueError, match="Arm alias is required"):
        set_arm("", "so101_follower", "/dev/ttyACM0", path=manifest_file)


def test_remove_arm(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
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
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("old_alias", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    result = rename_arm("old_alias", "new_alias", path=manifest_file)
    arm = find_arm(result["arms"], "new_alias")
    assert arm["calibration_dir"] == str(calibration_dir)
    assert arm["calibrated"] is True
    assert find_arm(result["arms"], "old_alias") is None


def test_rename_arm_rejects_duplicate_alias(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        set_arm("left_arm", "so101_follower", "/dev/ttyACM0", path=manifest_file)
        set_arm("right_arm", "so101_leader", "/dev/ttyACM1", path=manifest_file)
    with pytest.raises(ValueError, match="already exists"):
        rename_arm("left_arm", "right_arm", path=manifest_file)


_MOCK_SCANNED_CAMERAS = [
    VideoInterface(by_path="/dev/v4l/by-path/cam0", by_id="usb-cam0", dev="/dev/video0", width=640, height=480),
    VideoInterface(by_path="/dev/v4l/by-path/cam1", by_id="usb-cam1", dev="/dev/video2", width=320, height=240),
]


def test_set_camera(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
        result = set_camera("front", 0, path=manifest_file)
    cam = find_camera(result["cameras"], "front")
    assert cam is not None
    assert cam["alias"] == "front"
    assert cam["port"] == "usb-cam0"
    assert cam["width"] == 640
    assert cam["height"] == 480
    assert cam["fps"] == 30


def test_set_camera_bad_index(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
        with pytest.raises(ValueError, match="out of range"):
            set_camera("front", 99, path=manifest_file)


def test_remove_camera(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=_MOCK_SCANNED_CAMERAS):
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


def test_validation_rejects_bimanual_followers_without_distinct_sides(manifest_file: Path) -> None:
    bad = load_manifest(manifest_file)
    bad["arms"] = [
        {"alias": "arm_a", "type": "so101_follower", "port": "/dev/a", "side": "left"},
        {"alias": "arm_b", "type": "so101_follower", "port": "/dev/b"},
    ]
    with pytest.raises(ValueError, match="one 'left' arm and one 'right' arm"):
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
    with pytest.raises(ActionError, match="No arm with alias or port 'missing' in manifest."):
        _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), "missing")


def test_resolve_arms_accepts_alias_lookup(tmp_path: Path) -> None:
    result = _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), "right_follower")
    assert [arm.alias for arm in result] == ["right_follower"]


def test_group_arms(tmp_path: Path) -> None:
    grouped = group_arms(
        _resolve_arms(_manifest_from_data(tmp_path, _MOCK_SETUP), f"{_FOLLOWER_PORT},{_LEADER_PORT}")
    )
    assert [arm.alias for arm in grouped["followers"]] == ["right_follower"]
    assert [arm.alias for arm in grouped["leaders"]] == ["left_leader"]


def test_group_arms_uses_explicit_side_for_bimanual(tmp_path: Path) -> None:
    manifest = {
        **_MOCK_SETUP,
        "arms": [
            {
                "alias": "arm_b_follower",
                "type": "so101_follower",
                "port": _FOLLOWER_PORT,
                "calibration_dir": "/cal/right",
                "calibrated": False,
                "side": "right",
            },
            {
                "alias": "arm_a_follower",
                "type": "so101_follower",
                "port": _LEADER_PORT,
                "calibration_dir": "/cal/left",
                "calibrated": False,
                "side": "left",
            },
        ],
    }
    grouped = group_arms(_manifest_from_data(tmp_path, manifest).arms)
    assert [arm.side for arm in grouped["followers"]] == ["left", "right"]


def test_resolve_bimanual_pair_rejects_missing_side(tmp_path: Path) -> None:
    manifest = {
        **_MOCK_SETUP,
        "arms": [
            {
                "alias": "follower_a",
                "type": "so101_follower",
                "port": _FOLLOWER_PORT,
                "calibration_dir": "/cal/a",
                "calibrated": False,
            },
            {
                "alias": "follower_b",
                "type": "so101_follower",
                "port": _LEADER_PORT,
                "calibration_dir": "/cal/b",
                "calibrated": False,
            },
        ],
    }
    # Bypass save_manifest's bimanual validation to construct an ambiguous
    # manifest on disk, then verify runtime pairing rejects it.
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    followers = Manifest(path=path).arms
    with pytest.raises(ActionError, match="one 'left' arm and one 'right' arm"):
        resolve_bimanual_pair(followers, "followers")


# ── Serial number extraction test ────────────────────────────────────


def test_calibration_dir_uses_serial_number(manifest_file: Path, calibration_root: Path) -> None:
    """calibration_dir should be based on serial number extracted from by_id port."""
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS):
        result = set_arm("my_follower", "so101_follower", "/dev/ttyACM0", path=manifest_file)
    arm = find_arm(result["arms"], "my_follower")
    assert arm["calibration_dir"] == str(calibration_root / "5B14032630")


# ── set_hand / remove_hand / find_hand tests ─────────────────────────


def test_set_hand(manifest_file: Path) -> None:
    with (
        std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS),
        std_patch("roboclaw.embodied.embodiment.manifest.helpers._probe_hand_slave_id", return_value=1),
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
        std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=_MOCK_SCANNED_PORTS),
        std_patch("roboclaw.embodied.embodiment.manifest.helpers._probe_hand_slave_id", return_value=1),
    ):
        set_hand("h", "inspire_rh56", "/dev/ttyACM0", path=manifest_file)
        result = set_hand("h", "revo2", "/dev/ttyACM1", path=manifest_file)
    assert len(result["hands"]) == 1
    assert find_hand(result["hands"], "h")["port"] == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030892-if00"


def test_set_hand_invalid_type(manifest_file: Path) -> None:
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid hand_type"):
            set_hand("h", "so101_follower", "/dev/ttyUSB0", path=manifest_file)


def test_remove_hand(manifest_file: Path) -> None:
    with (
        std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]),
        std_patch("roboclaw.embodied.embodiment.manifest.helpers._probe_hand_slave_id", return_value=1),
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
    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_serial_ports", return_value=[]):
        with pytest.raises(ValueError, match="Invalid arm_type"):
            set_arm("h", "revo2", "/dev/ttyUSB0", path=manifest_file)


def test_train_schema_has_no_hand_params() -> None:
    """hand_type, hand_name, positions should NOT appear in train."""
    tool = _find_tool(create_embodied_tools(), "train")
    props = tool.parameters["properties"]
    assert "hand_type" not in props
    assert "hand_name" not in props
    assert "positions" not in props


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


def test_resolve_cameras_maps_stable_camera_port_to_runtime_index(tmp_path: Path) -> None:
    setup = {
        "cameras": [
            {"alias": "front", "port": "usb-camera", "width": 640, "height": 480, "fps": 30},
        ],
    }
    scanned = [
        VideoInterface(
            by_id="usb-camera",
            by_path="USB相机",
            dev="2",
            width=1920,
            height=1080,
            fps=30,
        ),
    ]

    with std_patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=scanned):
        cameras = _resolve_cameras(_manifest_from_data(tmp_path, {**_MOCK_SETUP, **setup}).cameras)

    assert cameras["front"]["index_or_path"] == 2


def test_dataset_path_appends_local_and_dataset_name(tmp_path: Path) -> None:
    assert dataset_path(_manifest_from_data(tmp_path, _MOCK_SETUP), "demo") == Path("/data/local/demo")
