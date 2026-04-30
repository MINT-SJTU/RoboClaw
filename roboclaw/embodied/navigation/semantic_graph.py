"""Semantic place graph for map-grounded navigation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GRAPH_VERSION = 1


@dataclass(frozen=True)
class SemanticPoint:
    """A 2D point in a semantic map frame."""

    x: float
    y: float

    @classmethod
    def from_mapping(cls, value: Any) -> "SemanticPoint":
        if not isinstance(value, dict):
            raise ValueError("semantic point must be an object.")
        return cls(
            x=float(_required(value, "x", "semantic point")),
            y=float(_required(value, "y", "semantic point")),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class SemanticPose:
    """A navigation goal pose associated with a semantic place."""

    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"

    @classmethod
    def from_mapping(cls, value: Any, *, default_yaw: float = 0.0) -> "SemanticPose":
        if not isinstance(value, dict):
            raise ValueError("semantic pose must be an object.")
        return cls(
            x=float(_required(value, "x", "semantic pose")),
            y=float(_required(value, "y", "semantic pose")),
            yaw=float(value.get("yaw", default_yaw)),
            frame_id=str(value.get("frame_id", "map") or "map"),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw, "frame_id": self.frame_id}


@dataclass(frozen=True)
class SemanticRegion:
    """A polygonal region occupied by a room or place label."""

    region_id: str
    frame_id: str
    polygon: tuple[SemanticPoint, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "SemanticRegion":
        if not isinstance(value, dict):
            raise ValueError("semantic region must be an object.")
        polygon = tuple(SemanticPoint.from_mapping(point) for point in value.get("polygon", []))
        if len(polygon) < 3:
            raise ValueError("semantic region polygon must contain at least 3 points.")
        return cls(
            region_id=str(value.get("id", "") or ""),
            frame_id=str(value.get("frame_id", "map") or "map"),
            polygon=polygon,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "frame_id": self.frame_id,
            "polygon": [point.to_dict() for point in self.polygon],
        }
        if self.region_id:
            data["id"] = self.region_id
        return data


@dataclass(frozen=True)
class SemanticPlace:
    """A named semantic place that can be grounded to a goal pose."""

    place_id: str
    place_type: str
    aliases: tuple[str, ...]
    regions: tuple[SemanticRegion, ...]
    goal_candidates: tuple[SemanticPose, ...]
    preferred_yaw: float

    @classmethod
    def from_mapping(cls, value: Any) -> "SemanticPlace":
        if not isinstance(value, dict):
            raise ValueError("semantic place must be an object.")
        place_id = normalize_place_label(str(_required(value, "id", "semantic place")))
        preferred_yaw = float(value.get("preferred_yaw", 0.0))
        candidates = tuple(
            SemanticPose.from_mapping(candidate, default_yaw=preferred_yaw)
            for candidate in value.get("goal_candidates", [])
        )
        regions = _regions_from_mapping(value)
        if not regions and not candidates:
            raise ValueError(f"semantic place '{place_id}' requires regions or goal_candidates.")
        return cls(
            place_id=place_id,
            place_type=str(value.get("type", "place") or "place"),
            aliases=tuple(normalize_place_label(alias) for alias in value.get("aliases", [])),
            regions=regions,
            goal_candidates=candidates,
            preferred_yaw=preferred_yaw,
        )

    def matches(self, label: str) -> bool:
        normalized = normalize_place_label(label)
        return normalized == self.place_id or normalized in self.aliases

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.place_id,
            "type": self.place_type,
            "aliases": list(self.aliases),
            "preferred_yaw": self.preferred_yaw,
            "region_count": len(self.regions),
            "goal_candidate_count": len(self.goal_candidates),
        }
        if self.regions:
            data["regions"] = [region.to_dict() for region in self.regions]
        return data


@dataclass(frozen=True)
class SemanticGraph:
    """A room/place graph tied to one occupancy map."""

    graph_id: str
    map_id: str
    map_path: str
    places: tuple[SemanticPlace, ...]
    edges: tuple[dict[str, Any], ...]
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticGraph":
        source_path = Path(path).expanduser()
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("semantic graph file must contain a JSON object.")
        version = int(data.get("version", GRAPH_VERSION))
        if version != GRAPH_VERSION:
            raise ValueError(f"Unsupported semantic graph version: {version}")
        places = tuple(SemanticPlace.from_mapping(place) for place in data.get("places", []))
        if not places:
            raise ValueError("semantic graph must define at least one place.")
        edges = tuple(_copy_edge(edge) for edge in data.get("edges", []))
        return cls(
            graph_id=str(data.get("id", source_path.stem) or source_path.stem),
            map_id=str(data.get("map_id", "") or ""),
            map_path=str(data.get("map_path", "") or ""),
            places=places,
            edges=edges,
            source_path=source_path,
        )

    def resolve_place(self, label: str) -> SemanticPlace:
        for place in self.places:
            if place.matches(label):
                return place
        available = ", ".join(place.place_id for place in self.places)
        raise ValueError(f"Unknown semantic place '{label}'. Available: {available}")

    def resolve_map_path(self) -> Path:
        if not self.map_path:
            raise ValueError("semantic graph does not declare map_path.")
        path = Path(self.map_path).expanduser()
        if path.is_absolute():
            return path
        if self.source_path is None:
            return path
        return self.source_path.parent / path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.graph_id,
            "map_id": self.map_id,
            "map_path": self.map_path,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "places": [place.to_dict() for place in self.places],
            "edges": list(self.edges),
        }


def load_semantic_graph(path: str | Path) -> SemanticGraph:
    """Load a semantic graph from a JSON file."""
    return SemanticGraph.from_file(path)


def normalize_place_label(value: str) -> str:
    """Normalize human place labels for exact graph lookup."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _copy_edge(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("semantic graph edge must be an object.")
    return dict(value)


def _regions_from_mapping(value: dict[str, Any]) -> tuple[SemanticRegion, ...]:
    regions_data = value.get("regions", [])
    if not isinstance(regions_data, list):
        raise ValueError("semantic place field 'regions' must be a list.")
    return tuple(SemanticRegion.from_mapping(region) for region in regions_data)


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} requires field '{key}'.")
    return mapping[key]
