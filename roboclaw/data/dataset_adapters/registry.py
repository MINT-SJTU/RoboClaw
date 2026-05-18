"""Dataset adapter resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from .lerobot import LeRobotAdapter
from .mapping import MappingAdapter

_MAPPING_FILENAMES = (
    "dataset_mapping.json",
    "mapping.json",
    ".roboclaw/dataset_mapping.json",
)


def resolve_dataset_adapter(
    dataset_path: Path,
    mapping: dict[str, Any] | None = None,
) -> DatasetAdapter:
    """Return the adapter for *dataset_path*.

    Explicit mappings win.  Otherwise, a mapping file opts into MappingAdapter.
    Existing LeRobot-style datasets remain the default path.
    """
    if mapping:
        return MappingAdapter(dataset_path, mapping)
    mapping_from_file = _load_mapping_file(dataset_path)
    if mapping_from_file:
        return MappingAdapter(dataset_path, mapping_from_file)
    return LeRobotAdapter(dataset_path)


def _load_mapping_file(dataset_path: Path) -> dict[str, Any] | None:
    for relative_path in _MAPPING_FILENAMES:
        path = dataset_path / relative_path
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"dataset mapping file must contain a JSON object: {path}")
        return payload
    return None
