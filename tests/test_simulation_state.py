"""Tests for isolated simulation state persistence."""

from __future__ import annotations

import pytest

from roboclaw.embodied.simulation.profiles import (
    DEFAULT_PROFILE_ID,
    get_profile,
    list_profiles,
)
from roboclaw.embodied.simulation.state import (
    default_simulation_state,
    get_simulation_state_path,
    load_simulation_state,
    save_simulation_state,
    sync_from_doctor_manifest,
)


def test_default_simulation_state_uses_default_profile() -> None:
    state = default_simulation_state()

    assert state["version"] == 1
    assert state["mode"] == "simulation"
    assert state["profile_id"] == DEFAULT_PROFILE_ID
    assert state["robot"] == "turtlebot3"
    assert state["simulator"] == "gazebo"
    assert "/navigate_to_pose" in state["capabilities"]["actions"]
    assert "map->odom" in state["capabilities"]["transforms"]
    assert state["last_discovery"] is None
    assert state["last_doctor"] is None


def test_simulation_state_path_isolated_from_manifest(tmp_path) -> None:
    path = get_simulation_state_path(tmp_path)

    assert path == tmp_path / "workspace" / "embodied" / "simulation_state.json"
    assert path.name != "manifest.json"


def test_save_and_load_simulation_state_round_trip(tmp_path) -> None:
    path = tmp_path / "simulation_state.json"
    state = default_simulation_state()
    state["paths"]["map"] = "robotics/ros_ws/src/roboclaw_tb3_sim/maps/map.yaml"

    saved = save_simulation_state(state, path)
    loaded = load_simulation_state(path)

    assert loaded == saved
    assert loaded["paths"]["map"].endswith("maps/map.yaml")


def test_sync_from_doctor_manifest_updates_state_without_manifest_file(tmp_path) -> None:
    doctor_manifest = {
        "profile_id": DEFAULT_PROFILE_ID,
        "mode": "simulation",
        "robot": "turtlebot3",
        "simulator": "gazebo",
        "checks": {
            "packages": {"nav2_bringup": True},
            "nodes": {"/bt_navigator": True},
            "topics": {"/scan": True, "/odom": True, "/tf": True},
            "actions": {"/navigate_to_pose": True},
            "services": {"/reset_simulation": False},
            "tf": {"map->odom": True, "odom->base_footprint": True},
        },
        "status": {
            "environment_installed": True,
            "runtime_up": True,
            "tf_ready": True,
            "nav_ready": False,
        },
        "decision": "needs_reconfiguration",
        "next_steps": ["Start or fix the missing Nav2 nodes, actions, or services."],
        "errors": [{
            "category": "navigation",
            "message": "Navigation capabilities are missing: /reset_simulation",
        }],
    }

    state = sync_from_doctor_manifest(doctor_manifest)
    save_simulation_state(state, get_simulation_state_path(tmp_path))

    assert state["profile_id"] == DEFAULT_PROFILE_ID
    assert state["capabilities"]["services"] == ["/reset_simulation"]
    assert state["sensors"] == ["lidar"]
    assert state["last_doctor"]["decision"] == "needs_reconfiguration"
    assert state["last_discovery"] == doctor_manifest
    assert not (tmp_path / "workspace" / "embodied" / "manifest.json").exists()


def test_profile_registry_lists_and_validates_profiles() -> None:
    profiles = list_profiles()

    assert [profile.profile_id for profile in profiles] == [DEFAULT_PROFILE_ID]
    assert get_profile(DEFAULT_PROFILE_ID).profile_id == DEFAULT_PROFILE_ID
    with pytest.raises(ValueError, match="Unknown simulation profile"):
        get_profile("missing_profile")
