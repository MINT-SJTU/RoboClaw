"""Setup management — single source of truth for the user's embodied configuration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ARM_TYPES = ("so101_follower", "so101_leader", "koch_follower", "koch_leader")
_ARM_FIELDS = {"alias", "type", "port", "calibration_dir", "calibrated"}
_HAND_TYPES = ("inspire_rh56", "revo2")
_HAND_FIELDS = {"alias", "type", "port", "slave_id"}
_CAMERA_FIELDS = {"alias", "port", "width", "height", "fps", "fourcc"}
_VALID_TOP_KEYS = {"version", "arms", "hands", "cameras", "datasets", "policies"}


# ── Generic device helpers ───────────────────────────────────────────


def _find_by_alias(items: list[dict], alias: str) -> dict | None:
    """Find an item in a list of dicts by its 'alias' field."""
    for item in items:
        if item.get("alias") == alias:
            return item
    return None


def _remove_device(key: str, alias: str, path: Path | None = None) -> dict[str, Any]:
    """Generic load/find/remove/save for a device list (arms, hands, etc.)."""
    path = path or get_setup_path()
    setup = load_setup(path)
    items = setup.get(key, [])
    item = _find_by_alias(items, alias)
    if item is None:
        label = key.rstrip("s")  # "arms" -> "arm"
        raise ValueError(f"No {label} with alias '{alias}' in setup.")
    items.remove(item)
    save_setup(setup, path)
    return setup


def _validate_device_list(
    items: Any, allowed_fields: set, allowed_types: tuple, label: str,
) -> None:
    """Validate a list of device entries (arms, hands, etc.)."""
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


def get_roboclaw_home(home: str | Path | None = None) -> Path:
    """Return RoboClaw home directory, honoring ROBOCLAW_HOME env var."""
    if home is not None:
        return Path(home).expanduser()
    return Path(os.environ.get("ROBOCLAW_HOME", "~/.roboclaw")).expanduser()


def get_setup_path(home: Path | None = None) -> Path:
    """Return the setup.json path under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "setup.json"


def get_calibration_root(home: Path | None = None) -> Path:
    """Return the calibration directory under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "calibration"


def _default_setup(home: Path | None = None) -> dict[str, Any]:
    """Build a fresh default setup dict with paths under *home*."""
    base = (home or get_roboclaw_home()) / "workspace" / "embodied"
    return {
        "version": 2,
        "arms": [],
        "hands": [],
        "cameras": [],
        "datasets": {"root": str(base / "datasets")},
        "policies": {"root": str(base / "policies")},
    }


def load_setup(path: Path | None = None) -> dict[str, Any]:
    """Load setup.json, return defaults if not found. Refreshes calibration state from disk."""
    path = path or get_setup_path()
    if not path.exists():
        return _default_setup()
    setup = json.loads(path.read_text(encoding="utf-8"))
    setup.pop("scanned_ports", None)
    setup.pop("scanned_cameras", None)
    if isinstance(setup.get("cameras"), dict):
        setup["cameras"] = []
        save_setup(setup, path)
    if _refresh_calibration_state(setup):
        save_setup(setup, path)
    return setup


def save_setup(setup: dict[str, Any], path: Path | None = None) -> None:
    """Write setup.json, creating parent dirs if needed."""
    path = path or get_setup_path()
    _validate_setup(setup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_setup(path: Path | None = None) -> dict[str, Any]:
    """Create a fresh setup.json. Called during onboard."""
    path = path or get_setup_path()
    setup = _default_setup()
    save_setup(setup, path)
    return setup


def ensure_setup(path: Path | None = None) -> dict[str, Any]:
    """Load setup.json if exists, otherwise create with defaults (no scan) and return."""
    path = path or get_setup_path()
    if path.exists():
        return load_setup(path)
    defaults = _default_setup()
    save_setup(defaults, path)
    return defaults



def mark_arm_calibrated(alias: str, path: Path | None = None) -> dict[str, Any]:
    """Mark an arm as calibrated by alias."""
    path = path or get_setup_path()
    setup = load_setup(path)
    arm = find_arm(setup.get("arms", []), alias)
    if not arm:
        raise ValueError(f"No arm with alias '{alias}' in setup.")
    arm["calibrated"] = True
    save_setup(setup, path)
    return setup


def _resolve_port(port: str, scanned_ports: list[dict]) -> str:
    """Resolve a volatile port (e.g. /dev/ttyACM0) to a stable by_id path.

    If port already starts with /dev/serial/by-id/ or /dev/serial/by-path/, keep as-is.
    Otherwise look up in scanned_ports and prefer by_id > by_path > original.
    """
    if port.startswith("/dev/serial/"):
        return port
    for entry in scanned_ports:
        if entry.get("dev") != port:
            continue
        by_id = entry.get("by_id", "")
        if by_id:
            return by_id
        return port
    return port


def _extract_serial_number(port: str) -> str:
    """Extract serial number from a by_id port path.

    E.g. "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00" -> "5B14032630"
    Falls back to the full filename if no pattern matches.
    """
    filename = Path(port).name
    # Match serial number: last segment before optional -ifNN suffix
    m = re.search(r"_([A-Za-z0-9]+)(?:-if\d+)?$", filename)
    if m:
        return m.group(1)
    return filename


# ── Structured mutators (exposed as agent actions) ──────────────────


def set_arm(
    alias: str, arm_type: str, port: str, *, path: Path | None = None,
) -> dict[str, Any]:
    """Add or update an arm by alias. Auto-fills calibration_dir and calibration state."""
    if arm_type not in _ARM_TYPES:
        raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
    if not port:
        raise ValueError("Arm port is required.")
    if not alias:
        raise ValueError("Arm alias is required.")
    from roboclaw.embodied.scan import scan_serial_ports
    path = path or get_setup_path()
    setup = load_setup(path)
    port = _resolve_port(port, scan_serial_ports())
    serial = _extract_serial_number(port)
    calibration_dir = get_calibration_root() / serial
    _migrate_none_calibration_file(calibration_dir, serial)
    existing = find_arm(setup.setdefault("arms", []), alias)
    _ensure_unique_port(setup["arms"], alias, port)
    entry: dict[str, Any] = {
        "alias": alias,
        "type": arm_type,
        "port": port,
        "calibration_dir": str(calibration_dir),
        "calibrated": _has_calibration_file(calibration_dir, serial),
    }
    arms = setup["arms"]
    if existing is not None:
        idx = arms.index(existing)
        arms[idx] = entry
    else:
        arms.append(entry)
    save_setup(setup, path)
    return setup


def arm_display_name(arm: dict) -> str:
    """Return user-friendly display name: the arm's alias."""
    return arm.get("alias", "unnamed")


def find_arm(arms: list[dict], alias: str) -> dict | None:
    """Find an arm in the arms list by alias. Returns the dict or None."""
    return _find_by_alias(arms, alias)


def remove_arm(alias: str, path: Path | None = None) -> dict[str, Any]:
    """Remove an arm by alias."""
    return _remove_device("arms", alias, path)


def rename_arm(old_alias: str, new_alias: str, *, path: Path | None = None) -> dict[str, Any]:
    """Rename an arm alias without changing any hardware-backed fields."""
    path = path or get_setup_path()
    if not old_alias:
        raise ValueError("Old arm alias is required.")
    if not new_alias:
        raise ValueError("New arm alias is required.")
    setup = load_setup(path)
    arms = setup.get("arms", [])
    arm = find_arm(arms, old_alias)
    if arm is None:
        raise ValueError(f"No arm with alias '{old_alias}' in setup.")
    if old_alias != new_alias and find_arm(arms, new_alias) is not None:
        raise ValueError(f"Arm alias '{new_alias}' already exists.")
    arm["alias"] = new_alias
    save_setup(setup, path)
    return setup


def set_hand(alias: str, hand_type: str, port: str, *, path: Path | None = None) -> dict[str, Any]:
    """Add or update a dexterous hand by alias."""
    if hand_type not in _HAND_TYPES:
        raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of {_HAND_TYPES}.")
    if not port:
        raise ValueError("Hand port is required.")
    if not alias:
        raise ValueError("Hand alias is required.")
    from roboclaw.embodied.scan import scan_serial_ports
    path = path or get_setup_path()
    setup = load_setup(path)
    port = _resolve_port(port, scan_serial_ports())

    # Auto-detect slave_id
    slave_id = _probe_hand_slave_id(hand_type, port)

    entry: dict[str, Any] = {"alias": alias, "type": hand_type, "port": port, "slave_id": slave_id}
    hands = setup.setdefault("hands", [])
    existing = find_hand(hands, alias)
    if existing is not None:
        hands[hands.index(existing)] = entry
    else:
        hands.append(entry)
    save_setup(setup, path)
    return setup


def _probe_hand_slave_id(hand_type: str, port: str) -> int:
    """Auto-detect slave_id by probing the serial port."""
    _PROBE_MODULES = {
        "inspire_rh56": "roboclaw.embodied.embodiment.hand.inspire_rh56",
        "revo2": "roboclaw.embodied.embodiment.hand.revo2",
    }
    module_path = _PROBE_MODULES.get(hand_type)
    if not module_path:
        raise ValueError(f"No probe available for hand type '{hand_type}'.")
    import importlib
    probe_fn = getattr(importlib.import_module(module_path), "probe_slave_ids")
    found = probe_fn(port)
    if not found:
        raise ValueError(f"No {hand_type} hand detected on this port.")
    if len(found) > 1:
        raise ValueError(f"Multiple devices detected on this port (found {len(found)}). Only one hand per port is supported.")
    return found[0]


def remove_hand(alias: str, path: Path | None = None) -> dict[str, Any]:
    """Remove a hand by alias."""
    return _remove_device("hands", alias, path)


def find_hand(hands: list[dict], alias: str) -> dict | None:
    """Find a hand in the hands list by alias. Returns the dict or None."""
    return _find_by_alias(hands, alias)


def set_camera(name: str, camera_index: int, path: Path | None = None) -> dict[str, Any]:
    """Add or update a camera by picking from live-scanned cameras by index."""
    from roboclaw.embodied.scan import scan_cameras

    path = path or get_setup_path()
    if not name:
        raise ValueError("Camera alias is required.")
    setup = load_setup(path)
    scanned = scan_cameras()
    if camera_index < 0 or camera_index >= len(scanned):
        raise ValueError(
            f"camera_index {camera_index} out of range. "
            f"Found {len(scanned)} camera(s)."
        )
    source = scanned[camera_index]
    port = source.get("by_path") or source.get("by_id") or source.get("dev", "")
    if not port:
        raise ValueError(f"Scanned camera at index {camera_index} has no usable path.")
    entry: dict[str, Any] = {
        "alias": name,
        "port": port,
        "width": source.get("width", 640),
        "height": source.get("height", 480),
    }
    if source.get("fps"):
        entry["fps"] = source["fps"]
    if source.get("fourcc"):
        entry["fourcc"] = source["fourcc"]
    cameras = setup.setdefault("cameras", [])
    existing = find_camera(cameras, name)
    if existing is not None:
        cameras[cameras.index(existing)] = entry
    else:
        cameras.append(entry)
    save_setup(setup, path)
    return setup


def remove_camera(name: str, path: Path | None = None) -> dict[str, Any]:
    """Remove a camera by alias."""
    path = path or get_setup_path()
    setup = load_setup(path)
    cameras = setup.get("cameras", [])
    cam = find_camera(cameras, name)
    if cam is None:
        raise ValueError(f"No camera with alias '{name}' in setup.")
    cameras.remove(cam)
    save_setup(setup, path)
    return setup


def find_camera(cameras: list[dict], alias: str) -> dict | None:
    """Find a camera in the cameras list by alias. Returns the dict or None."""
    return _find_by_alias(cameras, alias)


# ── Validation ───────────────────────────────────────────────────────


def _validate_setup(setup: dict[str, Any]) -> None:
    """Validate setup against schema. Raises ValueError on invalid data."""
    invalid_top = set(setup.keys()) - _VALID_TOP_KEYS
    if invalid_top:
        raise ValueError(f"Unknown top-level keys: {invalid_top}")
    _validate_arms(setup.get("arms", []))
    _validate_hands(setup.get("hands", []))
    _validate_cameras(setup.get("cameras", []))


def _validate_arms(arms: Any) -> None:
    """Validate all arm entries."""
    _validate_device_list(arms, _ARM_FIELDS, _ARM_TYPES, "Arm")


def _validate_hands(hands: Any) -> None:
    """Validate all hand entries."""
    _validate_device_list(hands, _HAND_FIELDS, _HAND_TYPES, "Hand")


def _validate_cameras(cameras: Any) -> None:
    """Validate all camera entries."""
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
        bad = set(cam.keys()) - _CAMERA_FIELDS
        if bad:
            raise ValueError(f"Camera '{alias}' has unknown fields: {bad}")


def _ensure_unique_port(arms: list[dict], alias: str, port: str) -> None:
    for arm in arms:
        if arm.get("alias") == alias:
            continue
        if arm.get("port") == port:
            raise ValueError(f"Port '{port}' is already assigned to arm '{arm['alias']}'.")


def _refresh_calibration_state(setup: dict[str, Any]) -> bool:
    """Migrate None.json and recompute calibrated from disk for all arms. Returns True if anything changed."""
    changed = False
    for arm in setup.get("arms", []):
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
