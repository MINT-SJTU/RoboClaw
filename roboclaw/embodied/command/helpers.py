"""Shared helpers for command building and service layer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.manifest.binding import Binding

_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ActionError(RuntimeError):
    """User-facing embodied action error."""


def group_arms(arms: list[Binding]) -> dict[str, list[Binding]]:
    """Split arms into followers and leaders.

    For bimanual (2 arms per role), sort by alias so left_* < right_*.
    """
    grouped: dict[str, list[Binding]] = {"followers": [], "leaders": []}
    for arm in arms:
        if arm.is_follower:
            grouped["followers"].append(arm)
        elif arm.is_leader:
            grouped["leaders"].append(arm)
    for role in ("followers", "leaders"):
        if len(grouped[role]) == 2:
            grouped[role].sort(key=lambda a: (0 if "left" in a.alias else 1))
    return grouped


def validate_dataset_name(name: str) -> None:
    """Raise ValueError if name is not a valid dataset slug."""
    if not name or not _DATASET_NAME_RE.match(name):
        raise ValueError(
            "dataset_name must be a non-empty ASCII slug "
            "(letters, numbers, underscores, hyphens)."
        )


def dataset_path(manifest: Any, name: str, fallback: Path | None = None) -> Path:
    """Resolve dataset root path for a given dataset name."""
    root = manifest.snapshot.get("datasets", {}).get("root", "")
    if root:
        return Path(root).expanduser() / "local" / name
    if fallback:
        return fallback.expanduser() / "local" / name
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets" / "local" / name


def logs_dir() -> Path:
    """Return the embodied jobs log directory."""
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "jobs"


def resolve_cameras(cameras: list[Binding]) -> dict[str, dict[str, Any]]:
    """Build camera config dict for lerobot CLI from manifest camera bindings."""
    result: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        if not cam.alias or not cam.port:
            continue
        config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": cam.port,
            "width": cam.interface.width,
            "height": cam.interface.height,
            "fps": cam.interface.fps or 30,
        }
        if hasattr(cam.interface, "fourcc") and cam.interface.fourcc:
            config["fourcc"] = cam.interface.fourcc
        result[cam.alias] = config
    return result


def resolve_action_arms(manifest: Any, arms_filter: str = "") -> list[Binding]:
    """Resolve arms from manifest, optionally filtered by alias or port."""
    configured = manifest.arms
    if not configured:
        return []
    if not arms_filter:
        return list(configured)
    tokens = {t.strip() for t in arms_filter.split(",") if t.strip()}
    resolved = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            raise ActionError(f"Duplicate arm identifier '{token}'.")
        seen.add(token)
        arm = next((a for a in configured if a.alias == token or a.port == token), None)
        if arm is None:
            raise ActionError(f"No arm with alias or port '{token}' in manifest.")
        resolved.append(arm)
    return resolved
