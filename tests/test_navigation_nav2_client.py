"""Tests for the isolated Nav2 CLI wrapper."""

from __future__ import annotations

from roboclaw.embodied.navigation.nav2_client import Nav2Client
from roboclaw.embodied.ros2.discovery import CommandResult


def test_nav2_client_navigate_to_pose_builds_goal_and_parses_feedback(tmp_path) -> None:
    seen = {}

    def shell_runner(argv, cwd, env, timeout_s):
        seen["argv"] = argv
        return CommandResult(
            0,
            (
                "Goal accepted with ID: 123\n"
                "number_of_recoveries: 2\n"
                "distance_remaining: 0.15\n"
                "Goal finished with status: SUCCEEDED\n"
            ),
            "",
        )

    client = Nav2Client(repo_root=tmp_path, shell_runner=shell_runner)
    result = client.navigate_to_pose(x=1.5, y=-0.25, yaw=1.57, feedback=True)

    assert result["ok"] is True
    assert result["goal_succeeded"] is True
    assert result["goal_status"] == "SUCCEEDED"
    assert result["metrics"]["number_of_recoveries"] == 2
    assert result["metrics"]["distance_remaining"] == 0.15
    assert "send_goal" in seen["argv"][2]
    assert "/navigate_to_pose" in seen["argv"][2]


def test_nav2_client_follow_waypoints_parses_waypoint_feedback(tmp_path) -> None:
    client = Nav2Client(
        repo_root=tmp_path,
        shell_runner=lambda argv, cwd, env, timeout_s: CommandResult(
            0,
            (
                "Goal accepted with ID: 456\n"
                "current_waypoint: 1\n"
                "missed_waypoints: [2]\n"
                "Goal finished with status: SUCCEEDED\n"
            ),
            "",
        ),
    )

    result = client.follow_waypoints(
        poses=[
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0, "yaw": 0.5},
        ]
    )

    assert result["goal_succeeded"] is True
    assert result["metrics"]["current_waypoint"] == 1
    assert result["metrics"]["missed_waypoints"] == [2]


def test_nav2_client_requires_accepted_goal_and_terminal_success_status(tmp_path) -> None:
    client = Nav2Client(
        repo_root=tmp_path,
        shell_runner=lambda argv, cwd, env, timeout_s: CommandResult(
            0,
            "Goal accepted with ID: 123\nnumber_of_recoveries: 1\n",
            "",
        ),
    )

    result = client.navigate_to_pose(x=0.5, y=0.75)

    assert result["ok"] is True
    assert result["goal_accepted"] is True
    assert result["goal_status"] is None
    assert result["goal_succeeded"] is False


def test_nav2_client_cancel_goals_parses_cancel_return_codes(tmp_path) -> None:
    calls = []

    def shell_runner(argv, cwd, env, timeout_s):
        calls.append(argv[2])
        if "/navigate_to_pose/_action/cancel_goal" in argv[2]:
            return CommandResult(0, "return_code: 0\n", "")
        return CommandResult(0, "return_code: 3\n", "")

    client = Nav2Client(repo_root=tmp_path, shell_runner=shell_runner)
    result = client.cancel_goals()

    assert result["ok"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["ok"] is True
    assert result["attempts"][1]["ok"] is False
