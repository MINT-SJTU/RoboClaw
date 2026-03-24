"""Setup management — single source of truth for the user's embodied configuration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SETUP_PATH = Path("~/.roboclaw/workspace/embodied/setup.json").expanduser()

_DEFAULT_SETUP: dict[str, Any] = {
    "version": 1,
    "robot": {
        "type": "",
        "port": "",
    },
    "cameras": [],
    "calibration": {
        "status": "missing",
        "dir": "",
    },
    "datasets": {
        "root": str(Path("~/.roboclaw/workspace/embodied/datasets").expanduser()),
    },
    "scanned_ports": [],
}


def load_setup(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Load setup.json, return defaults if not found."""
    if not path.exists():
        return copy.deepcopy(_DEFAULT_SETUP)
    return json.loads(path.read_text(encoding="utf-8"))


def save_setup(setup: dict[str, Any], path: Path = SETUP_PATH) -> None:
    """Write setup.json, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_setup_with_scan(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Create setup.json with auto-detected hardware. Called during onboard."""
    from roboclaw.embodied.scan import scan_cameras, scan_serial_ports

    setup = copy.deepcopy(_DEFAULT_SETUP)
    ports = scan_serial_ports()
    setup["scanned_ports"] = ports
    if len(ports) == 1:
        setup["robot"]["port"] = ports[0]["path"]
    cameras = scan_cameras()
    setup["cameras"] = [
        {
            "path": c.get("by_path") or c["dev"],
            "by_id": c.get("by_id", ""),
            "dev": c["dev"],
            "type": "opencv",
            "width": c["width"],
            "height": c["height"],
        }
        for c in cameras
    ]
    save_setup(setup, path)
    return setup


def ensure_setup(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Load setup.json if exists, otherwise create with defaults (no scan) and return."""
    if path.exists():
        return load_setup(path)
    defaults = copy.deepcopy(_DEFAULT_SETUP)
    save_setup(defaults, path)
    return defaults


def update_setup(updates: dict[str, Any], path: Path = SETUP_PATH) -> dict[str, Any]:
    """Merge updates into existing setup and save. Returns the merged result."""
    setup = load_setup(path)
    _deep_merge(setup, updates)
    save_setup(setup, path)
    return setup


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base, mutating base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
