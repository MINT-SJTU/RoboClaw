"""Shared helpers for command building and service layer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.manifest.binding import ArmBinding, ArmRole, CameraBinding

_DATASET_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ActionError(RuntimeError):
    """User-facing embodied action error."""


def group_arms(arms: list[ArmBinding]) -> dict[str, list[ArmBinding]]:
    """Split arms into followers and leaders.

    For bimanual arms with explicit sides, keep left before right.
    """
    grouped: dict[str, list[ArmBinding]] = {"followers": [], "leaders": []}
    for arm in arms:
        if arm.role is ArmRole.FOLLOWER:
            grouped["followers"].append(arm)
        elif arm.role is ArmRole.LEADER:
            grouped["leaders"].append(arm)
    for role in ("followers", "leaders"):
        if len(grouped[role]) == 2 and {arm.side for arm in grouped[role]} == {"left", "right"}:
            grouped[role].sort(key=lambda arm: 0 if arm.side == "left" else 1)
    return grouped


def resolve_bimanual_pair(
    arms: list[ArmBinding], role: str,
) -> tuple[ArmBinding, ArmBinding]:
    """Return the left/right pair for a bimanual role."""
    if len(arms) != 2:
        raise ActionError(
            f"Bimanual {role} requires exactly 2 arms, got {len(arms)}."
        )
    sides = {arm.side for arm in arms}
    if sides != {"left", "right"}:
        aliases = [arm.alias for arm in arms]
        raise ActionError(
            f"Bimanual {role} must include one 'left' arm and one 'right' arm; "
            f"got aliases {aliases} with sides {[arm.side for arm in arms]}."
        )
    left = next(arm for arm in arms if arm.side == "left")
    right = next(arm for arm in arms if arm.side == "right")
    return left, right


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


def policy_path(manifest: Any, name: str) -> Path:
    """Resolve policy root path for a given policy name."""
    root = manifest.snapshot.get("policies", {}).get("root", "")
    if root:
        return Path(root).expanduser() / name
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "policies" / name


def logs_dir() -> Path:
    """Return the embodied jobs log directory."""
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "jobs"


def resolve_cameras(cameras: list[CameraBinding]) -> dict[str, dict[str, Any]]:
    """Build camera config dict for lerobot CLI from manifest camera bindings."""
    from roboclaw.embodied.embodiment.hardware.scan import (
        resolve_camera_interface,
        scan_cameras,
    )

    scanned_cameras = scan_cameras() if cameras else []
    result: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        if not cam.alias or not cam.port:
            continue
        resolved = resolve_camera_interface(cam.port, scanned_cameras)
        runtime_address = resolved.runtime_address
        if not runtime_address:
            raise ActionError(f"Camera '{cam.alias}' is disconnected.")
        index_or_path: str | int = int(runtime_address) if runtime_address.isdigit() else runtime_address
        config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": index_or_path,
            "width": cam.interface.width,
            "height": cam.interface.height,
            "fps": cam.interface.fps or 30,
        }
        if hasattr(cam.interface, "fourcc") and cam.interface.fourcc:
            config["fourcc"] = cam.interface.fourcc
        result[cam.alias] = config
    return result


def resolve_action_arms(manifest: Any, arms_filter: str = "") -> list[ArmBinding]:
    """Resolve arms from manifest, optionally filtered by alias or port."""
    configured = manifest.arms
    if not configured:
        return []
    if not arms_filter:
        return list(configured)
    resolved = []
    seen: set[str] = set()
    for raw in arms_filter.split(","):
        token = raw.strip()
        if not token:
            continue
        if token in seen:
            raise ActionError(f"Duplicate arm identifier '{token}'.")
        seen.add(token)
        arm = next((a for a in configured if a.alias == token or a.port == token), None)
        if arm is None:
            raise ActionError(f"No arm with alias or port '{token}' in manifest.")
        resolved.append(arm)
    return resolved
