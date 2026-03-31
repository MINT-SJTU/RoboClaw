"""Dataset listing and deletion utilities for the web UI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger


def list_datasets(root: Path) -> list[dict]:
    """Scan *root* for LeRobot dataset directories and return metadata summaries.

    A valid dataset directory contains a ``meta/info.json`` file.
    """
    if not root.is_dir():
        return []

    datasets: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        info = _read_dataset_info(entry)
        if info is not None:
            datasets.append(info)
    return datasets


def get_dataset_info(root: Path, name: str) -> dict | None:
    """Return metadata for a single dataset, or None if not found."""
    dataset_dir = root / name
    if not dataset_dir.is_dir():
        return None
    return _read_dataset_info(dataset_dir)


def delete_dataset(root: Path, name: str) -> None:
    """Delete a dataset directory. Raises ValueError if it does not exist."""
    dataset_dir = root / name
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset '{name}' not found in {root}")
    logger.info("Deleting dataset: {}", dataset_dir)
    shutil.rmtree(dataset_dir)


def _read_dataset_info(dataset_dir: Path) -> dict | None:
    """Read meta/info.json and build a summary dict. Returns None if invalid."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None

    raw = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = raw.get("total_episodes", 0)
    total_frames = raw.get("total_frames", 0)
    fps = raw.get("fps", 0)

    # Collect episode lengths if available
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    episode_lengths: list[int] = []
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").strip().splitlines():
            ep = json.loads(line)
            length = ep.get("length", 0)
            episode_lengths.append(length)

    return {
        "name": dataset_dir.name,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "episode_lengths": episode_lengths,
        "features": list(raw.get("features", {}).keys()),
        "robot_type": raw.get("robot_type", ""),
    }
