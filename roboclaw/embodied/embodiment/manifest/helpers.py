"""Pure helper functions for manifest state management."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.arm.registry import all_arm_types
from roboclaw.embodied.embodiment.hand.registry import all_hand_types, get_hand_spec

# ── Constants ──────────────────────────────────────────────────────────

_ARM_TYPES = all_arm_types()
_ARM_FIELDS = {"alias", "type", "port", "calibration_dir", "calibrated"}
_HAND_TYPES = all_hand_types()
_HAND_FIELDS = {"alias", "type", "port", "slave_id"}
_CAMERA_FIELDS = {"alias", "side", "port", "width", "height", "fps", "fourcc"}
_VALID_TOP_KEYS = {"version", "arms", "hands", "cameras", "datasets", "policies"}


# ── Path helpers ───────────────────────────────────────────────────────


def get_roboclaw_home(home: str | Path | None = None) -> Path:
    """Return RoboClaw home directory, honoring ROBOCLAW_HOME env var."""
    if home is not None:
        return Path(home).expanduser()
    return Path(os.environ.get("ROBOCLAW_HOME", "~/.roboclaw")).expanduser()


def get_manifest_path(home: Path | None = None) -> Path:
    """Return the manifest.json path under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "manifest.json"



def get_calibration_root(home: Path | None = None) -> Path:
    """Return the calibration directory under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "calibration"


# ── Default manifest ──────────────────────────────────────────────────


def _default_manifest(home: Path | None = None) -> dict[str, Any]:
    """Build a fresh default manifest dict with paths under *home*."""
    base = (home or get_roboclaw_home()) / "workspace" / "embodied"
    return {
        "version": 2,
        "arms": [],
        "hands": [],
        "cameras": [],
        "datasets": {"root": str(base / "datasets")},
        "policies": {"root": str(base / "policies")},
    }


# ── Device finders ────────────────────────────────────────────────────


def _find_by_alias(items: list[dict], alias: str) -> dict | None:
    """Find an item in a list of dicts by its 'alias' field."""
    for item in items:
        if item.get("alias") == alias:
            return item
    return None


def find_arm(arms: list[dict], alias: str) -> dict | None:
    return _find_by_alias(arms, alias)


def find_camera(cameras: list[dict], alias: str) -> dict | None:
    return _find_by_alias(cameras, alias)


def find_hand(hands: list[dict], alias: str) -> dict | None:
    return _find_by_alias(hands, alias)


def arm_display_name(arm: Any) -> str:
    """Return user-friendly display name: the arm's alias."""
    if hasattr(arm, "alias"):
        return getattr(arm, "alias") or "unnamed"
    return arm.get("alias", "unnamed")


# ── Port resolution ───────────────────────────────────────────────────


def _resolve_port(port: str, scanned_ports: list) -> str:
    """Resolve a volatile port (e.g. /dev/ttyACM0) to a stable by_id path.

    Accepts list[SerialInterface] from scan_serial_ports().
    """
    if port.startswith("/dev/serial/"):
        return port
    for entry in scanned_ports:
        if entry.dev != port:
            continue
        if entry.by_id:
            return entry.by_id
        return port
    return port


def _resolve_serial_interface(port: str) -> "SerialInterface":
    """Scan ports, resolve a volatile dev path, return a SerialInterface."""
    from roboclaw.embodied.embodiment.hardware.scan import scan_serial_ports
    from roboclaw.embodied.embodiment.interface.serial import SerialInterface

    resolved = _resolve_port(port, scan_serial_ports())
    return SerialInterface(
        by_id=resolved if resolved.startswith("/dev/serial/") else "",
        dev=resolved,
    )


def _extract_serial_number(port: str) -> str:
    """Extract serial number from a by_id port path.

    E.g. "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00" -> "5B14032630"
    Falls back to the full filename if no pattern matches.
    """
    filename = Path(port).name
    m = re.search(r"_([A-Za-z0-9]+)(?:-if\d+)?$", filename)
    if m:
        return m.group(1)
    return filename


# ── Validation ────────────────────────────────────────────────────────


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate manifest against schema. Raises ValueError on invalid data."""
    invalid_top = set(manifest.keys()) - _VALID_TOP_KEYS
    if invalid_top:
        raise ValueError(f"Unknown top-level keys: {invalid_top}")
    _validate_arms(manifest.get("arms", []))
    _validate_hands(manifest.get("hands", []))
    _validate_cameras(manifest.get("cameras", []))


def _validate_arms(arms: Any) -> None:
    _validate_device_list(arms, _ARM_FIELDS, _ARM_TYPES, "Arm")


def _validate_hands(hands: Any) -> None:
    _validate_device_list(hands, _HAND_FIELDS, _HAND_TYPES, "Hand")


def _validate_cameras(cameras: Any) -> None:
    if not isinstance(cameras, list):
        raise ValueError("'cameras' must be a list.")
    for cam in cameras:
        if not isinstance(cam, dict):
            raise ValueError(f"Each camera must be a dict, got {type(cam).__name__}.")
        alias = cam.get("alias")
        if not alias:
            raise ValueError("Camera entry missing required 'alias' field.")
        if not cam.get("port"):
            raise ValueError(f"Camera '{alias}' missing required 'port' field.")
        side = cam.get("side")
        if side not in ("left", "right"):
            raise ValueError(
                f"Camera '{alias}' has invalid 'side' {side!r}; expected 'left' or 'right'."
            )
        bad = set(cam.keys()) - _CAMERA_FIELDS
        if bad:
            raise ValueError(f"Camera '{alias}' has unknown fields: {bad}")


def _validate_device_list(
    items: Any, allowed_fields: set, allowed_types: tuple, label: str,
) -> None:
    if not isinstance(items, list):
        raise ValueError(f"'{label.lower()}s' must be a list.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each {label.lower()} entry must be a dict, got {type(item).__name__}.")
        alias = item.get("alias", "<unknown>")
        bad_fields = set(item.keys()) - allowed_fields
        if bad_fields:
            raise ValueError(f"{label} '{alias}' has unknown fields: {bad_fields}")
        item_type = item.get("type")
        if item_type is not None and item_type not in allowed_types:
            raise ValueError(f"{label} '{alias}' has invalid type '{item_type}'.")


def _ensure_unique_port(arms: list[dict], alias: str, port: str) -> None:
    for arm in arms:
        if arm.get("alias") == alias:
            continue
        if arm.get("port") == port:
            raise ValueError(f"Port '{port}' is already assigned to arm '{arm['alias']}'.")


# ── Calibration file management ───────────────────────────────────────


def _refresh_calibration_state(manifest: dict[str, Any]) -> bool:
    """Migrate None.json and recompute calibrated from disk for all arms. Returns True if anything changed."""
    changed = False
    for arm in manifest.get("arms", []):
        cal_dir = Path(arm.get("calibration_dir", ""))
        serial = cal_dir.name
        if not serial or not cal_dir.exists():
            continue
        _migrate_none_calibration_file(cal_dir, serial)
        on_disk = _has_calibration_file(cal_dir, serial)
        if arm.get("calibrated") != on_disk:
            arm["calibrated"] = on_disk
            changed = True
    return changed


def _has_calibration_file(calibration_dir: Path, serial: str) -> bool:
    return (calibration_dir / f"{serial}.json").exists()


def _migrate_none_calibration_file(calibration_dir: Path, serial: str) -> None:
    legacy = calibration_dir / "None.json"
    target = calibration_dir / f"{serial}.json"
    if legacy.exists() and not target.exists():
        legacy.rename(target)


def load_calibration(arm: Any) -> dict[str, Any]:
    """Load calibration JSON for an arm. Returns empty dict if not found."""
    if hasattr(arm, "calibration_dir"):
        cal_dir = getattr(arm, "calibration_dir")
    else:
        cal_dir = arm.get("calibration_dir", "")
    if not cal_dir:
        return {}
    serial = Path(cal_dir).name
    cal_path = Path(cal_dir).expanduser() / f"{serial}.json"
    if not cal_path.exists():
        return {}
    return json.loads(cal_path.read_text(encoding="utf-8"))


# ── Bimanual calibration directory management ─────────────────────────


def ensure_bimanual_cal_dir(
    left_arm: Any, right_arm: Any, role: str,
) -> str:
    """Return a persistent bimanual calibration directory, creating/refreshing if needed."""
    target_dir = get_calibration_root() / f"bimanual_{role}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for side, arm in [("left", left_arm), ("right", right_arm)]:
        raw = arm.get("calibration_dir", "") if isinstance(arm, dict) else arm.calibration_dir
        cal_dir = Path(raw).expanduser()
        serial = cal_dir.name
        source = cal_dir / f"{serial}.json"
        if not source.exists():
            continue
        dest = target_dir / f"bimanual_{side}.json"
        if dest.exists() and source.stat().st_mtime <= dest.stat().st_mtime:
            continue
        shutil.copy2(source, dest)
    return str(target_dir)


def refresh_bimanual_cal_dirs(manifest: dict[str, Any]) -> None:
    """Eagerly refresh bimanual calibration dirs if a bimanual pair exists."""
    from loguru import logger

    arms = manifest.get("arms", [])
    followers = [a for a in arms if "follower" in a.get("type", "")]
    leaders = [a for a in arms if "leader" in a.get("type", "")]
    try:
        if len(followers) == 2:
            ensure_bimanual_cal_dir(followers[0], followers[1], "followers")
        if len(leaders) == 2:
            ensure_bimanual_cal_dir(leaders[0], leaders[1], "leaders")
    except Exception:
        logger.opt(exception=True).warning("Failed to refresh bimanual calibration dirs")


# ── Hand probing ──────────────────────────────────────────────────────


def _probe_hand_slave_id(hand_type: str, port: str) -> int:
    """Auto-detect slave_id by probing the serial port."""
    from roboclaw.embodied.embodiment.hand.modbus import probe_modbus_slave_ids

    spec = get_hand_spec(hand_type)
    candidates = list(spec.probe_candidates) if spec.probe_candidates else list(range(1, 17))
    found = probe_modbus_slave_ids(
        port, spec.baudrate, candidates, spec.probe_register, spec.probe_register_count,
    )
    if not found:
        raise ValueError(f"No {hand_type} hand detected on this port.")
    if len(found) > 1:
        raise ValueError(f"Multiple devices detected on this port (found {len(found)}). Only one hand per port is supported.")
    return found[0]


# ── Free-function API (for subprocess / CLI / fallback paths) ────────


def _lazy_manifest(path: Path | None = None) -> "Manifest":
    from roboclaw.embodied.embodiment.manifest import Manifest

    return Manifest(path=path) if path else Manifest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).snapshot


def save_manifest(manifest: dict[str, Any], path: Path | None = None) -> None:
    _validate_manifest(manifest)
    target = path or get_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_manifest(path: Path | None = None) -> "Manifest":
    manifest = _lazy_manifest(path)
    manifest.ensure()
    return manifest


def set_arm(alias: str, arm_type: str, port: str, *, path: Path | None = None) -> dict[str, Any]:
    interface = _resolve_serial_interface(port)
    m = _lazy_manifest(path)
    m.set_arm(alias, arm_type, interface)
    return m.snapshot


def remove_arm(alias: str, path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).remove_arm(alias)


def rename_arm(old_alias: str, new_alias: str, *, path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).rename_arm(old_alias, new_alias)


def mark_arm_calibrated(alias: str, path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).mark_arm_calibrated(alias)


def set_camera(
    name: str, camera_index: int, side: str, path: Path | None = None,
) -> dict[str, Any]:
    from roboclaw.embodied.embodiment.hardware.scan import scan_cameras

    scanned = scan_cameras()
    if camera_index < 0 or camera_index >= len(scanned):
        raise ValueError(
            f"camera_index {camera_index} out of range. "
            f"Found {len(scanned)} camera(s)."
        )
    interface = scanned[camera_index]
    if not interface.address:
        raise ValueError(f"Scanned camera at index {camera_index} has no usable path.")
    m = _lazy_manifest(path)
    m.set_camera(name, interface, side)
    return m.snapshot


def remove_camera(name: str, path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).remove_camera(name)


def set_hand(alias: str, hand_type: str, port: str, *, path: Path | None = None) -> dict[str, Any]:
    if hand_type not in _HAND_TYPES:
        raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of {_HAND_TYPES}.")
    interface = _resolve_serial_interface(port)
    slave_id = _probe_hand_slave_id(hand_type, interface.address)
    m = _lazy_manifest(path)
    m.set_hand(alias, hand_type, interface, slave_id)
    return m.snapshot


def remove_hand(alias: str, path: Path | None = None) -> dict[str, Any]:
    return _lazy_manifest(path).remove_hand(alias)
