"""Tests for dataset/policy listing and record auto-timestamp/resume logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.helpers import save_manifest
from roboclaw.embodied.toolkit.tools import EmbodiedToolGroup, create_embodied_tools


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


def _manifest_from_data(tmp_path: Path, data: dict) -> Manifest:
    path = tmp_path / "manifest.json"
    save_manifest(data, path)
    return Manifest(path=path)


@pytest.fixture(autouse=True)
def calibration_root(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    with patch("roboclaw.embodied.embodiment.manifest.helpers.get_calibration_root", return_value=root):
        yield root


# ── Auto-timestamp and resume tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_record_auto_generates_timestamp_name(tmp_path: Path) -> None:
    """When dataset_name is omitted, record goes through CLI adapter without error."""
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "record")

    async def fake_tty_run(self, session):
        assert "dataset_name" not in session._kwargs  # auto-generated inside engine
        return "Recording finished."

    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(
            action="record",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result


@pytest.mark.asyncio
async def test_record_resumes_existing_named_dataset(tmp_path: Path) -> None:
    """When user specifies dataset_name, it is passed through kwargs to CLI adapter."""
    setup = {**_MOCK_SETUP, "datasets": {"root": str(tmp_path)}}
    existing = tmp_path / "local" / "my_dataset"
    existing.mkdir(parents=True)

    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "record")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    async def fake_tty_run(self, session):
        assert session._kwargs.get("dataset_name") == "my_dataset"
        return "Recording finished."

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(
            action="record",
            dataset_name="my_dataset",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result


@pytest.mark.asyncio
async def test_record_no_resume_for_new_named_dataset(tmp_path: Path) -> None:
    """When user specifies a new dataset_name, it is passed to CLI adapter."""
    tool = _find_tool(create_embodied_tools(tty_handoff=AsyncMock()), "record")

    async def fake_tty_run(self, session):
        assert session._kwargs.get("dataset_name") == "brand_new"
        return "Recording finished."

    manifest = _manifest_from_data(tmp_path, _MOCK_SETUP)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with patch("roboclaw.embodied.toolkit.tty.TtySession.run", fake_tty_run):
        result = await tool.execute(
            action="record",
            dataset_name="brand_new",
            task="grasp",
            arms=f"{_FOLLOWER_PORT},{_LEADER_PORT}",
        )

    assert "Recording finished" in result


# ── list_datasets / list_policies tests ──────────────────────────────


@pytest.mark.asyncio
async def test_list_datasets_empty(tmp_path: Path) -> None:
    setup = {**_MOCK_SETUP, "datasets": {"root": "/nonexistent"}}
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

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
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    result = await tool.execute(action="list_datasets")

    datasets = json.loads(result)
    assert len(datasets) == 1
    assert datasets[0]["id"] == "local/demo1"
    assert datasets[0]["label"] == "demo1"
    assert datasets[0]["stats"]["total_episodes"] == 3
    assert datasets[0]["runtime"]["name"] == "demo1"


@pytest.mark.asyncio
async def test_list_datasets_raises_on_corrupt_json(tmp_path: Path) -> None:
    ds_dir = tmp_path / "local" / "bad" / "meta"
    ds_dir.mkdir(parents=True)
    (ds_dir / "info.json").write_text("{corrupt")
    setup = {**_MOCK_SETUP, "datasets": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with pytest.raises(json.JSONDecodeError):
        await tool.execute(action="list_datasets")


@pytest.mark.asyncio
async def test_list_policies_empty(tmp_path: Path) -> None:
    setup = {**_MOCK_SETUP, "policies": {"root": "/nonexistent"}}
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

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
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    result = await tool.execute(action="list_policies")

    policies = json.loads(result)
    assert len(policies) == 1
    assert policies[0]["name"] == "my_policy"
    assert policies[0]["dataset"] == "local/demo"
    assert policies[0]["steps"] == 5000


@pytest.mark.asyncio
async def test_list_policies_raises_on_corrupt_config(tmp_path: Path) -> None:
    p = tmp_path / "bad_pol" / "checkpoints" / "last" / "pretrained_model"
    p.mkdir(parents=True)
    (p / "train_config.json").write_text("not json")
    setup = {**_MOCK_SETUP, "policies": {"root": str(tmp_path)}}
    tool = _find_tool(create_embodied_tools(), "train")
    manifest = _manifest_from_data(tmp_path, setup)
    from roboclaw.embodied.service import EmbodiedService
    tool.embodied_service = EmbodiedService(manifest=manifest)

    with pytest.raises(json.JSONDecodeError):
        await tool.execute(action="list_policies")
