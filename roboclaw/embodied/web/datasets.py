"""Dataset listing and deletion utilities for the web UI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

from roboclaw.embodied.setup import load_setup


# ---------------------------------------------------------------------------
# Shared path helpers (used by workflow_routes, explorer_routes, etc.)
# ---------------------------------------------------------------------------


def datasets_root() -> Path:
    """Return the datasets root directory from setup.json."""
    setup = load_setup()
    root = setup.get("datasets", {}).get("root", "")
    if not root:
        raise HTTPException(status_code=500, detail="No datasets root configured in setup.json")
    return Path(root)


def resolve_dataset_path(name: str) -> Path:
    """Resolve a dataset name to its full path on disk.

    All candidates are ``.resolve()``-d and checked to stay strictly inside
    the datasets root, preventing path-traversal attacks via ``../`` segments.
    """
    root = datasets_root()
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Datasets root '{root}' does not exist")
    resolved_root = root.resolve()

    def _is_safe(path: Path) -> bool:
        rp = path.resolve()
        return rp.is_dir() and str(rp).startswith(str(resolved_root) + "/")

    direct = root / name
    if _is_safe(direct):
        return direct.resolve()

    for parent in root.iterdir():
        candidate = parent / name
        if parent.is_dir() and _is_safe(candidate):
            return candidate.resolve()

    raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")


# ---------------------------------------------------------------------------
# Feature name extraction (used by workflow_routes, explorer_routes, etc.)
# ---------------------------------------------------------------------------


def extract_action_names(info: dict) -> list[str]:
    """Extract action joint names from info.json features."""
    features = info.get("features", {})
    names = features.get("action", {}).get("names", [])
    return [str(n) for n in names] if isinstance(names, list) else []


def extract_state_names(info: dict) -> list[str]:
    """Extract state joint names from info.json features."""
    features = info.get("features", {})
    for key in ("observation.state", "state"):
        names = features.get(key, {}).get("names", [])
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    return []


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------


def list_datasets(root: Path) -> list[dict]:
    """Scan *root* for LeRobot dataset directories and return metadata summaries.

    A valid dataset directory contains a ``meta/info.json`` file.
    Scans up to 2 levels deep (handles ``root/local/dataset_name/`` layout).
    """
    if not root.is_dir():
        return []

    datasets: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        info = _read_dataset_info(root, entry)
        if info is not None:
            datasets.append(info)
            continue
        # Scan one level deeper (e.g. root/local/*)
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir():
                continue
            info = _read_dataset_info(root, sub)
            if info is not None:
                datasets.append(info)
    return datasets


def get_dataset_info(root: Path, name: str) -> dict | None:
    """Return metadata for a single dataset, or None if not found."""
    dataset_dir = (root / name).resolve()
    if not str(dataset_dir).startswith(str(root.resolve())):
        return None
    if not dataset_dir.is_dir():
        return None
    return _read_dataset_info(root, dataset_dir)


def delete_dataset(root: Path, name: str) -> None:
    """Delete a dataset directory. Raises ValueError if it does not exist."""
    dataset_dir = (root / name).resolve()
    if not str(dataset_dir).startswith(str(root.resolve())):
        raise ValueError(f"Dataset '{name}' not found in {root}")
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset '{name}' not found in {root}")
    logger.info("Deleting dataset: {}", dataset_dir)
    shutil.rmtree(dataset_dir)


def _read_dataset_info(root: Path, dataset_dir: Path) -> dict | None:
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
        "name": dataset_dir.relative_to(root).as_posix(),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "episode_lengths": episode_lengths,
        "features": list(raw.get("features", {}).keys()),
        "robot_type": raw.get("robot_type", ""),
        "source_dataset": raw.get("repo_id") or raw.get("dataset_id") or dataset_dir.relative_to(root).as_posix(),
    }
