"""Synchronous embodied setup actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.embodied.ops.helpers import _camera_previews_dir

_ACTION_DESCRIPTIONS = {
    "doctor": "Check LeRobot availability and show the current embodied setup.",
    "identify": "Launch the interactive arm-identification flow for detected serial ports.",
    "describe": "Explain adjustable parameters for a target embodied action.",
    "calibrate": "Calibrate one or more configured arms. If arms is omitted, calibrate every uncalibrated arm.",
    "teleoperate": "Run live teleoperation. Select arms with a comma-separated port list.",
    "record": "Record a dataset with one follower/leader pair or two pairs for bimanual capture.",
    "replay": "Replay a recorded dataset episode on one or two follower arms.",
    "train": "Start ACT training for a recorded dataset as a detached job.",
    "run_policy": "Run a trained policy: use embodied_control(action='record', checkpoint_path='...') with one follower arm.",
    "job_status": "Inspect the status and recent logs for a detached training job.",
    "setup_show": "Show the embodied setup JSON with configured arms, cameras, and roots.",
    "set_arm": "Create or update one configured arm alias.",
    "rename_arm": "Rename an existing configured arm alias.",
    "remove_arm": "Remove one configured arm alias.",
    "set_camera": "Assign a scanned camera to a stable camera name.",
    "preview_cameras": "Capture one preview image for each scanned camera.",
    "remove_camera": "Remove a configured camera.",
    "set_hand": "Create or update one configured hand alias.",
    "remove_hand": "Remove a configured hand alias.",
    "hand_open": "Open all fingers of a dexterous hand.",
    "hand_close": "Close all fingers of a dexterous hand.",
    "hand_pose": "Set individual finger positions on a dexterous hand.",
    "hand_status": "Read current finger angles and forces from a dexterous hand.",
    "list_datasets": "List recorded datasets with episode counts.",
    "list_policies": "List trained policy checkpoints.",
}


def _do_setup_show(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.scan import scan_cameras, scan_serial_ports
    from roboclaw.embodied.setup import load_setup

    setup = load_setup()
    setup["scanned_ports"] = scan_serial_ports()
    setup["scanned_cameras"] = scan_cameras()
    return json.dumps(setup, indent=2, ensure_ascii=False)


def _do_describe(kwargs: dict[str, Any]) -> str:
    target_action = kwargs.get("target_action", "")
    if not target_action:
        return json.dumps(_ACTION_DESCRIPTIONS, indent=2, ensure_ascii=False)
    if target_action not in _ACTION_DESCRIPTIONS:
        return f"Unknown target_action: {target_action}"
    return f"{target_action}: {_ACTION_DESCRIPTIONS[target_action]}"


def _do_set_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import arm_display_name, find_arm, set_arm

    alias = kwargs.get("alias", "")
    arm_type = kwargs.get("arm_type", "")
    port = kwargs.get("port", "")
    if not all([alias, arm_type, port]):
        return "set_arm requires alias, arm_type, and port."
    updated = set_arm(alias, arm_type, port)
    arm = find_arm(updated["arms"], alias)
    display = arm_display_name(arm)
    return f"Arm '{display}' configured.\n{json.dumps(arm, indent=2)}"


def _do_rename_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_arm, rename_arm

    old_alias = kwargs.get("alias", "")
    new_alias = kwargs.get("new_alias", "")
    if not old_alias or not new_alias:
        return "rename_arm requires alias and new_alias."
    updated = rename_arm(old_alias, new_alias)
    arm = find_arm(updated["arms"], new_alias)
    return f"Arm renamed from '{old_alias}' to '{new_alias}'.\n{json.dumps(arm, indent=2)}"


def _do_remove_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_arm

    alias = kwargs.get("alias", "")
    if not alias:
        return "remove_arm requires alias."
    remove_arm(alias)
    return f"Arm '{alias}' removed."


def _do_set_camera(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_camera, set_camera

    name = kwargs.get("camera_name", "")
    index = kwargs.get("camera_index")
    if not name or index is None:
        return "set_camera requires camera_name and camera_index."
    updated = set_camera(name, index)
    cam = find_camera(updated["cameras"], name)
    return f"Camera '{name}' configured.\n{json.dumps(cam, indent=2)}"


def _do_preview_cameras(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.scan import capture_camera_frames, scan_cameras

    scanned_cameras = scan_cameras()
    if not scanned_cameras:
        return "No cameras detected."
    try:
        previews = capture_camera_frames(scanned_cameras, _camera_previews_dir())
    except RuntimeError as exc:
        return f"Camera preview failed: {exc}"
    if not previews:
        return "No camera previews captured."
    return json.dumps(previews, indent=2, ensure_ascii=False)


def _do_remove_camera(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_camera

    name = kwargs.get("camera_name", "")
    if not name:
        return "remove_camera requires camera_name."
    remove_camera(name)
    return f"Camera '{name}' removed."


def _do_set_hand(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_hand, set_hand

    alias = kwargs.get("alias", "")
    hand_type = kwargs.get("hand_type", "")
    port = kwargs.get("port", "")
    if not all([alias, hand_type, port]):
        return "set_hand requires alias, hand_type, and port."
    updated = set_hand(alias, hand_type, port)
    hand = find_hand(updated["hands"], alias)
    return f"Hand '{alias}' configured.\n{json.dumps(hand, indent=2)}"


def _do_remove_hand(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_hand

    alias = kwargs.get("alias", "")
    if not alias:
        return "remove_hand requires alias."
    remove_hand(alias)
    return f"Hand '{alias}' removed."


def _do_list_datasets(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import ensure_setup

    setup = ensure_setup()
    root = Path(setup.get("datasets", {}).get("root", "")) / "local"
    if not root.exists():
        return "No datasets found."
    datasets = []
    for d in sorted(root.iterdir()):
        info_path = d / "meta" / "info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        datasets.append({
            "name": d.name,
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "fps": info.get("fps", 0),
        })
    if not datasets:
        return "No datasets found."
    return json.dumps(datasets, indent=2, ensure_ascii=False)


def _do_list_policies(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import ensure_setup

    setup = ensure_setup()
    root = Path(setup.get("policies", {}).get("root", ""))
    if not root.exists():
        return "No policies found."
    policies = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        last = d / "checkpoints" / "last" / "pretrained_model"
        if not last.exists():
            continue
        entry = {"name": d.name, "checkpoint": str(last)}
        tcfg = last / "train_config.json"
        if tcfg.exists():
            try:
                cfg = json.loads(tcfg.read_text())
            except (json.JSONDecodeError, OSError):
                cfg = {}
            entry["dataset"] = cfg.get("dataset", {}).get("repo_id", "")
            entry["steps"] = cfg.get("steps", 0)
        policies.append(entry)
    if not policies:
        return "No policies found."
    return json.dumps(policies, indent=2, ensure_ascii=False)


SYNC_DISPATCH: dict[str, Any] = {
    "setup_show": _do_setup_show,
    "describe": _do_describe,
    "set_arm": _do_set_arm,
    "rename_arm": _do_rename_arm,
    "remove_arm": _do_remove_arm,
    "set_camera": _do_set_camera,
    "preview_cameras": _do_preview_cameras,
    "remove_camera": _do_remove_camera,
    "set_hand": _do_set_hand,
    "remove_hand": _do_remove_hand,
    "list_datasets": _do_list_datasets,
    "list_policies": _do_list_policies,
}
