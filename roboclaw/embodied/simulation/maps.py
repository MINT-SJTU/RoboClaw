"""Named map registry for simulation navigation demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationMap:
    """A named map that can be selected by the agent during bringup."""

    map_id: str
    description: str
    path: str
    world_launch: str | None = None
    semantic_graph_path: str | None = None
    aliases: tuple[str, ...] = ()


DEFAULT_MAP_ID = "default"

DEFAULT_MAP = SimulationMap(
    map_id=DEFAULT_MAP_ID,
    description="Default TurtleBot3 map.",
    path="robotics/ros_ws/src/roboclaw_tb3_sim/maps/map.yaml",
    world_launch="turtlebot3_world.launch.py",
    aliases=("map", "default_map", "tb3"),
)

HOUSE_MAP = SimulationMap(
    map_id="house",
    description="House map for semantic navigation and room-to-room demos.",
    path="robotics/ros_ws/src/roboclaw_tb3_sim/maps/map_house.yaml",
    world_launch="turtlebot3_house.launch.py",
    semantic_graph_path="robotics/ros_ws/src/roboclaw_tb3_sim/maps/map_house.semantic.json",
    aliases=("semantic", "semantic_house", "house_map", "room", "room_to_room", "room2room"),
)

_MAPS: dict[str, SimulationMap] = {
    DEFAULT_MAP.map_id: DEFAULT_MAP,
    HOUSE_MAP.map_id: HOUSE_MAP,
}

_ALIASES: dict[str, str] = {
    alias: map_info.map_id
    for map_info in _MAPS.values()
    for alias in (map_info.map_id, *map_info.aliases)
}


def get_simulation_map(map_id: str | None = None) -> SimulationMap:
    """Return a named simulation map by id or alias."""
    requested = _normalize_map_id(map_id or DEFAULT_MAP_ID)
    canonical = _ALIASES.get(requested, requested)
    try:
        return _MAPS[canonical]
    except KeyError as exc:
        available = ", ".join(sorted(_MAPS))
        raise ValueError(f"Unknown simulation map '{map_id}'. Available: {available}") from exc


def list_simulation_maps() -> tuple[SimulationMap, ...]:
    """Return all registered simulation maps."""
    return tuple(_MAPS[key] for key in sorted(_MAPS))


def resolve_simulation_map_path(
    *,
    map_id: str | None = None,
    map_path: str | Path | None = None,
) -> tuple[str | Path | None, SimulationMap | None]:
    """Resolve an explicit path or named map selection.

    Non-empty explicit ``map_path`` always wins. ``map_id`` is used when no
    path was provided. Returning the map metadata lets service state record
    which named map was selected.
    """
    if map_path is not None and str(map_path).strip():
        return map_path, None
    if map_id is None:
        return None, None
    selected = get_simulation_map(map_id)
    return selected.path, selected


def _normalize_map_id(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
