"""Resolve semantic places to clear Nav2 goal poses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboclaw.embodied.navigation.semantic_graph import (
    SemanticGraph,
    SemanticPlace,
    SemanticPoint,
    SemanticPose,
)


DEFAULT_CLEARANCE_M = 0.25
DEFAULT_GOAL_STRIDE_M = 0.10


@dataclass(frozen=True)
class SemanticGoal:
    """A clear navigation goal selected for a semantic place."""

    place_id: str
    place_type: str
    pose: SemanticPose
    source: str
    clearance_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place_id,
            "place_type": self.place_type,
            "pose": self.pose.to_dict(),
            "source": self.source,
            "clearance_m": self.clearance_m,
        }


@dataclass(frozen=True)
class OccupancyGridMap:
    """Minimal ROS occupancy-map reader for semantic goal validation."""

    yaml_path: Path
    image_path: Path
    width: int
    height: int
    pixels: bytes
    resolution: float
    origin_x: float
    origin_y: float
    occupied_thresh: float
    free_thresh: float
    negate: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OccupancyGridMap":
        yaml_path = Path(path).expanduser()
        config = _read_simple_yaml(yaml_path)
        image_name = str(_required(config, "image", "occupancy map yaml"))
        image_path = Path(image_name).expanduser()
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path
        width, height, pixels = _read_pgm(image_path)
        origin = config.get("origin", [0.0, 0.0, 0.0])
        return cls(
            yaml_path=yaml_path,
            image_path=image_path,
            width=width,
            height=height,
            pixels=pixels,
            resolution=float(_required(config, "resolution", "occupancy map yaml")),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            occupied_thresh=float(config.get("occupied_thresh", 0.65)),
            free_thresh=float(config.get("free_thresh", 0.25)),
            negate=bool(int(config.get("negate", 0))),
        )

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = math.floor((x - self.origin_x) / self.resolution)
        map_y = math.floor((y - self.origin_y) / self.resolution)
        row = self.height - 1 - map_y
        return int(col), int(row)

    def pixel_to_world(self, col: int, row: int) -> tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (self.height - row - 0.5) * self.resolution
        return x, y

    def is_clear_world(self, x: float, y: float, clearance_m: float) -> bool:
        col, row = self.world_to_pixel(x, y)
        radius = max(0, math.ceil(clearance_m / self.resolution))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius and not self.is_free_cell(col + dx, row + dy):
                    return False
        return True

    def is_free_cell(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return False
        value = self.pixels[row * self.width + col]
        occupied_probability = value / 255.0 if self.negate else (255 - value) / 255.0
        return occupied_probability < self.free_thresh


class SemanticGoalResolver:
    """Ground semantic graph labels into free-space Nav2 poses."""

    def resolve(
        self,
        *,
        graph: SemanticGraph,
        place_label: str,
        occupancy_map_path: str | Path | None = None,
        clearance_m: float = DEFAULT_CLEARANCE_M,
        goal_stride_m: float = DEFAULT_GOAL_STRIDE_M,
    ) -> SemanticGoal:
        place = graph.resolve_place(place_label)
        map_path = Path(occupancy_map_path).expanduser() if occupancy_map_path else graph.resolve_map_path()
        grid = OccupancyGridMap.from_yaml(map_path)
        return self._select_goal(
            place=place,
            grid=grid,
            clearance_m=clearance_m,
            goal_stride_m=goal_stride_m,
        )

    def _select_goal(
        self,
        *,
        place: SemanticPlace,
        grid: OccupancyGridMap,
        clearance_m: float,
        goal_stride_m: float,
    ) -> SemanticGoal:
        explicit = self._explicit_candidate(place, grid, clearance_m)
        if explicit is not None:
            return explicit
        if place.region is None:
            raise ValueError(f"Semantic place '{place.place_id}' has no region or clear goal candidates.")
        pose = self._region_candidate(place, grid, clearance_m, goal_stride_m)
        return SemanticGoal(
            place_id=place.place_id,
            place_type=place.place_type,
            pose=pose,
            source="region_free_space",
            clearance_m=clearance_m,
        )

    @staticmethod
    def _explicit_candidate(
        place: SemanticPlace,
        grid: OccupancyGridMap,
        clearance_m: float,
    ) -> SemanticGoal | None:
        for candidate in place.goal_candidates:
            if grid.is_clear_world(candidate.x, candidate.y, clearance_m):
                return SemanticGoal(
                    place_id=place.place_id,
                    place_type=place.place_type,
                    pose=candidate,
                    source="goal_candidate",
                    clearance_m=clearance_m,
                )
        return None

    @staticmethod
    def _region_candidate(
        place: SemanticPlace,
        grid: OccupancyGridMap,
        clearance_m: float,
        goal_stride_m: float,
    ) -> SemanticPose:
        if place.region is None:
            raise ValueError(f"Semantic place '{place.place_id}' has no region.")
        polygon = place.region.polygon
        centroid = _polygon_centroid(polygon)
        stride = max(1, round(goal_stride_m / grid.resolution))
        col_min, row_min, col_max, row_max = _polygon_pixel_bounds(polygon, grid)

        best: tuple[float, SemanticPose] | None = None
        for row in range(row_min, row_max + 1, stride):
            for col in range(col_min, col_max + 1, stride):
                x, y = grid.pixel_to_world(col, row)
                if not _point_in_polygon(x, y, polygon):
                    continue
                if not grid.is_clear_world(x, y, clearance_m):
                    continue
                score = math.hypot(x - centroid.x, y - centroid.y)
                pose = SemanticPose(
                    x=x,
                    y=y,
                    yaw=place.preferred_yaw,
                    frame_id=place.region.frame_id,
                )
                if best is None or score < best[0]:
                    best = (score, pose)

        if best is None:
            raise ValueError(
                f"No clear goal found inside semantic place '{place.place_id}' "
                f"with clearance {clearance_m:.2f} m."
            )
        return best[1]


def _polygon_centroid(polygon: tuple[SemanticPoint, ...]) -> SemanticPoint:
    x = sum(point.x for point in polygon) / len(polygon)
    y = sum(point.y for point in polygon) / len(polygon)
    return SemanticPoint(x=x, y=y)


def _polygon_pixel_bounds(
    polygon: tuple[SemanticPoint, ...],
    grid: OccupancyGridMap,
) -> tuple[int, int, int, int]:
    cols: list[int] = []
    rows: list[int] = []
    for point in polygon:
        col, row = grid.world_to_pixel(point.x, point.y)
        cols.append(col)
        rows.append(row)
    return (
        max(0, min(cols) - 1),
        max(0, min(rows) - 1),
        min(grid.width - 1, max(cols) + 1),
        min(grid.height - 1, max(rows) + 1),
    )


def _point_in_polygon(x: float, y: float, polygon: tuple[SemanticPoint, ...]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        previous = polygon[j]
        intersects = (point.y > y) != (previous.y > y)
        if intersects:
            x_intersection = (previous.x - point.x) * (y - point.y) / (previous.y - point.y) + point.x
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid map yaml line: {raw_line}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_yaml_scalar(value.strip())
    return data


def _parse_yaml_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        parts = [part.strip() for part in value[1:-1].split(",") if part.strip()]
        return [float(part) for part in parts]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value:
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value.strip("\"'")


def _read_pgm(path: Path) -> tuple[int, int, bytes]:
    with path.open("rb") as stream:
        magic = _read_pgm_token(stream)
        width = int(_read_pgm_token(stream))
        height = int(_read_pgm_token(stream))
        max_value = int(_read_pgm_token(stream))
        if max_value > 255:
            raise ValueError("Only 8-bit PGM map images are supported.")
        if magic == b"P5":
            pixels = stream.read(width * height)
        elif magic == b"P2":
            pixels = bytes(int(_read_pgm_token(stream)) for _ in range(width * height))
        else:
            raise ValueError(f"Unsupported PGM magic: {magic!r}")
    if len(pixels) != width * height:
        raise ValueError(f"PGM image '{path}' has incomplete pixel data.")
    return width, height, pixels


def _read_pgm_token(stream: Any) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            if token:
                return bytes(token)
            raise ValueError("Unexpected end of PGM header.")
        if char == b"#":
            stream.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} requires field '{key}'.")
    return mapping[key]
