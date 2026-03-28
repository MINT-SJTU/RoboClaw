"""Tests for dataset/policy listing and record auto-timestamp/resume logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.tool import EmbodiedToolGroup, create_embodied_tools


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
    with patch("roboclaw.embodied.setup.get_calibration_root", return_value=root):
        yield root


# ── Auto-timestamp and resume tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_record_auto_generates_timestamp_name() -> None:
    """When dataset_name is omitted, record should auto-generate a rec_YYYYMMDD_HHMMSS name."""
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="record",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    repo_arg = [a for a in argv if a.startswith("--dataset.repo_id=")][0]
    assert repo_arg.startswith("--dataset.repo_id=local/rec_")
    assert "--resume=true" not in argv


@pytest.mark.asyncio
async def test_record_resumes_existing_named_dataset(tmp_path: Path) -> None:
    """When user specifies dataset_name and it already exists, --resume=true should be set."""
    setup = {**_MOCK_SETUP, "datasets": {"root": str(tmp_path)}}
    existing = tmp_path / "local" / "my_dataset"
    existing.mkdir(parents=True)

    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="my_dataset",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--resume=true" in argv


@pytest.mark.asyncio
async def test_record_no_resume_for_new_named_dataset() -> None:
    """When user specifies dataset_name but dir does not exist, no --resume."""
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "embodied_control")
    mock_runner = AsyncMock()
    mock_runner.run_interactive.return_value = (0, "")

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=_MOCK_SETUP),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(
            action="record",
            dataset_name="brand_new",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result
    argv = mock_runner.run_interactive.call_args.args[0]
    assert "--resume=true" not in argv


# ── list_datasets / list_policies tests ──────────────────────────────


@pytest.mark.asyncio
async def test_list_datasets_empty() -> None:
    setup = {**_MOCK_SETUP, "datasets": {"root": "/nonexistent"}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_datasets")

    assert result == "No datasets found."


@pytest.mark.asyncio
async def test_list_datasets_with_entries(tmp_path: Path) -> None:
    ds_dir = tmp_path / "local" / "demo1" / "meta"
    ds_dir.mkdir(parents=True)
    (ds_dir / "info.json").write_text(
        json.dumps({"total_episodes": 3, "total_frames": 90, "fps": 30})
    )
    setup = {**_MOCK_SETUP, "datasets": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_datasets")

    datasets = json.loads(result)
    assert len(datasets) == 1
    assert datasets[0]["name"] == "demo1"
    assert datasets[0]["episodes"] == 3


@pytest.mark.asyncio
async def test_list_datasets_skips_corrupt_json(tmp_path: Path) -> None:
    ds_dir = tmp_path / "local" / "bad" / "meta"
    ds_dir.mkdir(parents=True)
    (ds_dir / "info.json").write_text("{corrupt")
    setup = {**_MOCK_SETUP, "datasets": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_datasets")

    assert result == "No datasets found."


@pytest.mark.asyncio
async def test_list_policies_empty() -> None:
    setup = {**_MOCK_SETUP, "policies": {"root": "/nonexistent"}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_policies")

    assert result == "No policies found."


@pytest.mark.asyncio
async def test_list_policies_with_entries(tmp_path: Path) -> None:
    p = tmp_path / "my_policy" / "checkpoints" / "last" / "pretrained_model"
    p.mkdir(parents=True)
    (p / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "local/demo"}, "steps": 5000})
    )
    setup = {**_MOCK_SETUP, "policies": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_policies")

    policies = json.loads(result)
    assert len(policies) == 1
    assert policies[0]["name"] == "my_policy"
    assert policies[0]["dataset"] == "local/demo"
    assert policies[0]["steps"] == 5000


@pytest.mark.asyncio
async def test_list_policies_skips_corrupt_config(tmp_path: Path) -> None:
    p = tmp_path / "bad_pol" / "checkpoints" / "last" / "pretrained_model"
    p.mkdir(parents=True)
    (p / "train_config.json").write_text("not json")
    setup = {**_MOCK_SETUP, "policies": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "embodied_train")

    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup):
        result = await tool.execute(action="list_policies")

    policies = json.loads(result)
    assert len(policies) == 1
    assert policies[0]["name"] == "bad_pol"
    assert policies[0]["dataset"] == ""
    assert policies[0]["steps"] == 0
