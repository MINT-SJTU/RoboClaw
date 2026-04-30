"""Capability profiles for simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformCheck:
    """A TF transform expected to exist in the running simulation."""

    target_frame: str
    source_frame: str

    @property
    def key(self) -> str:
        return f"{self.target_frame}->{self.source_frame}"


@dataclass(frozen=True)
class SimulationProfile:
    """Expected ROS 2 capabilities for one simulation baseline."""

    profile_id: str = "turtlebot3_gazebo_nav2"
    mode: str = "simulation"
    robot: str = "turtlebot3"
    simulator: str = "gazebo"
    packages: tuple[str, ...] = (
        "turtlebot3_gazebo",
        "turtlebot3_navigation2",
        "nav2_bringup",
        "tf2_ros",
    )
    nodes: tuple[str, ...] = (
        "/bt_navigator",
        "/controller_server",
        "/planner_server",
    )
    topics: tuple[str, ...] = (
        "/cmd_vel",
        "/odom",
        "/scan",
        "/tf",
    )
    actions: tuple[str, ...] = (
        "/navigate_to_pose",
    )
    services: tuple[str, ...] = ()
    transforms: tuple[TransformCheck, ...] = (
        TransformCheck("map", "odom"),
        TransformCheck("odom", "base_footprint"),
    )

    def capability_dict(self) -> dict[str, list[str]]:
        """Return the profile capabilities in JSON-friendly form."""
        return {
            "packages": list(self.packages),
            "nodes": list(self.nodes),
            "topics": list(self.topics),
            "actions": list(self.actions),
            "services": list(self.services),
            "transforms": [transform.key for transform in self.transforms],
        }


DEFAULT_PROFILE_ID = "turtlebot3_gazebo_nav2"
DEFAULT_PROFILE = SimulationProfile()

_PROFILES: dict[str, SimulationProfile] = {
    DEFAULT_PROFILE.profile_id: DEFAULT_PROFILE,
}


def get_profile(profile_id: str | None = None) -> SimulationProfile:
    """Return a simulation capability profile by id."""
    requested = profile_id or DEFAULT_PROFILE_ID
    try:
        return _PROFILES[requested]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unknown simulation profile '{requested}'. Available: {available}"
        ) from exc


def default_profile() -> SimulationProfile:
    """Return the default simulation capability profile."""
    return DEFAULT_PROFILE


def list_profiles() -> tuple[SimulationProfile, ...]:
    """Return all registered simulation profiles."""
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))
