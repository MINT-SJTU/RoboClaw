"""Tests for semantic map grounding."""

from __future__ import annotations

import json
from pathlib import Path

from roboclaw.embodied.navigation.semantic_goal import SemanticGoalResolver
from roboclaw.embodied.navigation.semantic_graph import load_semantic_graph


def write_semantic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pgm_path = tmp_path / "map.pgm"
    pgm_path.write_text(
        "\n".join(
            [
                "P2",
                "8 8",
                "255",
                "0 0 0 0 0 0 0 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 255 255 255 255 255 255 0",
                "0 0 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "map.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "image: map.pgm",
                "resolution: 1.0",
                "origin: [0.0, 0.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    graph_path = tmp_path / "map.semantic.json"
    graph_path.write_text(
        json.dumps(
            {
                "version": 1,
                "id": "test_graph",
                "map_id": "test_map",
                "map_path": "map.yaml",
                "places": [
                    {
                        "id": "bedroom",
                        "type": "room",
                        "aliases": ["lower_right_room"],
                        "region": {
                            "frame_id": "map",
                            "polygon": [
                                {"x": 1.0, "y": 1.0},
                                {"x": 6.0, "y": 1.0},
                                {"x": 6.0, "y": 6.0},
                                {"x": 1.0, "y": 6.0},
                            ],
                        },
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return graph_path, yaml_path


def test_semantic_graph_resolves_bedroom_to_clear_goal(tmp_path: Path) -> None:
    graph_path, _ = write_semantic_fixture(tmp_path)
    graph = load_semantic_graph(graph_path)

    goal = SemanticGoalResolver().resolve(graph=graph, place_label="bedroom")

    assert goal.place_id == "bedroom"
    assert goal.source == "region_free_space"
    assert 1.0 < goal.pose.x < 6.0
    assert 1.0 < goal.pose.y < 6.0


def test_semantic_graph_resolves_alias(tmp_path: Path) -> None:
    graph_path, _ = write_semantic_fixture(tmp_path)
    graph = load_semantic_graph(graph_path)

    place = graph.resolve_place("lower right room")

    assert place.place_id == "bedroom"
