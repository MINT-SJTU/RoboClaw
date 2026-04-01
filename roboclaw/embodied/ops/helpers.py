"""Shared helpers for embodied tool operations."""

from __future__ import annotations

import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from roboclaw.embodied.setup import get_roboclaw_home

_NO_TTY_MSG = "This action requires a local terminal. Run: roboclaw agent"
_BIMANUAL_ID = "bimanual"
_DEFAULT_REPLAY_ROOT = Path("~/.cache/huggingface/lerobot").expanduser()
_DATASET_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ActionError(Exception):
    """User-facing embodied action error."""


def _resolve_action_arms(setup: dict[str, Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return _resolve_arms(setup, kwargs.get("arms", ""))
    except ValueError as exc:
        raise ActionError(str(exc)) from exc


def _resolve_arms(setup: dict[str, Any], arms_str: str) -> list[dict[str, Any]]:
    configured = setup.get("arms", [])
    if not configured:
        return []
    ports = _split_arm_tokens(arms_str)
    if not ports:
        return list(configured)
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for port in ports:
        if port in seen:
            raise ValueError(f"Duplicate arm port '{port}' in arms.")
        seen.add(port)
        arm = next((item for item in configured if item.get("port") == port), None)
        if arm is None:
            raise ValueError(f"No arm with port '{port}' found in setup.")
        resolved.append(arm)
    return resolved


def _split_arm_tokens(arms_str: str) -> list[str]:
    if not arms_str:
        return []
    return [token.strip() for token in arms_str.split(",") if token.strip()]


def _group_arms(arms: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"followers": [], "leaders": []}
    for arm in arms:
        arm_type = arm.get("type", "")
        if "follower" in arm_type:
            grouped["followers"].append(arm)
            continue
        if "leader" in arm_type:
            grouped["leaders"].append(arm)
    # Sort by alias so that "left_*" comes before "right_*".
    # Bimanual callers rely on [0]=left, [1]=right; without sorting,
    # the order depends on setup.json array position which is fragile.
    for role in ("followers", "leaders"):
        if len(grouped[role]) == 2:
            grouped[role].sort(key=lambda a: (0 if "left" in a.get("alias", "") else 1))
    return grouped


def _validate_pairing(followers: list[dict[str, Any]], leaders: list[dict[str, Any]]) -> str | None:
    if not followers:
        return "No follower arms configured."
    if not leaders:
        return "No leader arms configured."
    if len(followers) != len(leaders):
        return f"Follower/leader count mismatch: {len(followers)} followers, {len(leaders)} leaders."
    if len(followers) not in {1, 2}:
        return f"Unsupported arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
    return None


async def _run_tty(
    tty_handoff: Any, runner: Any, argv: list[str], label: str,
) -> tuple[int, str]:
    await tty_handoff(start=True, label=label)
    try:
        return await runner.run_interactive(argv)
    finally:
        await tty_handoff(start=False, label=label)


async def _run(runner: Any, argv: list[str]) -> str:
    returncode, stdout, stderr = await runner.run(argv)
    if returncode != 0:
        return f"Command failed (exit {returncode}).\nstdout: {stdout}\nstderr: {stderr}"
    return stdout or "Done."


def _is_interrupted(returncode: int) -> bool:
    return returncode in {130, -2}


def _format_tty_failure(prefix: str, returncode: int, stderr_text: str) -> str:
    message = f"{prefix} (exit {returncode})."
    stderr_text = stderr_text.strip()
    if not stderr_text:
        return message
    return f"{message}\nstderr: {stderr_text}"


def _dataset_root(setup: dict[str, Any], fallback: Path | None = None) -> Path:
    root = setup.get("datasets", {}).get("root", "")
    if root:
        return Path(root).expanduser()
    if fallback is not None:
        return fallback.expanduser()
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"


def _dataset_path(
    setup: dict[str, Any], dataset_name: str, fallback: Path | None = None,
) -> Path:
    return _dataset_root(setup, fallback) / "local" / dataset_name


def _validate_dataset_name(dataset_name: str) -> str | None:
    if not dataset_name or not _DATASET_SLUG_RE.match(dataset_name):
        return "dataset_name must be a non-empty ASCII slug (letters, numbers, underscores, hyphens)."
    return None


def _arm_id(arm: dict[str, Any]) -> str:
    arm_id = Path(arm.get("calibration_dir", "")).name
    if not arm_id:
        raise ValueError(f"Arm '{arm.get('alias', 'unknown')}' has no serial-based calibration_dir.")
    return arm_id


@contextmanager
def _bimanual_cal_dirs(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
) -> Iterator[tuple[str, str]]:
    with TemporaryDirectory(prefix="roboclaw-bimanual-robot-") as robot_dir:
        with TemporaryDirectory(prefix="roboclaw-bimanual-teleop-") as teleop_dir:
            _stage_bimanual_arm_pair(followers[0], followers[1], robot_dir)
            if leaders:
                _stage_bimanual_arm_pair(leaders[0], leaders[1], teleop_dir)
            yield robot_dir, teleop_dir


def _stage_bimanual_arm_pair(
    left_arm: dict[str, Any], right_arm: dict[str, Any], target_dir: str,
) -> None:
    target = Path(target_dir)
    for side, arm in [("left", left_arm), ("right", right_arm)]:
        serial = _arm_id(arm)
        source = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
        shutil.copy2(source, target / f"bimanual_{side}.json")


def _camera_previews_dir() -> Path:
    return get_roboclaw_home() / "workspace" / "embodied" / "camera_previews"


def _logs_dir() -> Path:
    """Return the embodied jobs log directory under ROBOCLAW_HOME."""
    return get_roboclaw_home() / "workspace" / "embodied" / "jobs"
