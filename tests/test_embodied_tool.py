"""Tests for the EmbodiedTool integration with the agent."""

from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.tool import EmbodiedTool


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
        "setup_show", "setup_update",
    ]
    assert action_schema["enum"] == expected_actions


_MOCK_SETUP = {
    "version": 2,
    "arms": {
        "follower": {
            "type": "so101_follower",
            "port": "/dev/ttyACM0",
            "calibration_dir": "/cal/f",
            "calibrated": False,
        },
        "leader": {
            "type": "so101_leader",
            "port": "/dev/ttyACM1",
            "calibration_dir": "/cal/l",
            "calibrated": False,
        },
    },
    "cameras": {
        "front": {"by_path": "", "by_id": "", "dev": "/dev/video0", "width": 640, "height": 480},
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
        patch("roboclaw.embodied.setup.update_setup"),
        patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner),
    ):
        result = await tool.execute(action="calibrate")

    assert "2 succeeded" in result
    assert "follower" in result
    assert "leader" in result
    assert mock_runner.run_interactive.call_count == 2
    assert mock_handoff.call_count == 4  # start+stop for each arm


@pytest.mark.asyncio
async def test_calibrate_no_arms() -> None:
    empty_setup = {**_MOCK_SETUP, "arms": {}}
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
        result = await tool.execute(action="record", dataset_name="test", task="grasp", num_episodes=5)

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
