"""Isolated simulation state for navigation-first workflows.

This module intentionally does not read or write the hardware manifest. It uses
the same ROBOCLAW_HOME convention, but persists simulation-only facts to
simulation_state.json until the navigation path is ready to merge.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from roboclaw.embodied.simulation.profiles import (
    DEFAULT_PROFILE,
    SimulationProfile,
    default_profile,
    get_profile,
)


STATE_VERSION = 1
_PATH_KEYS = ("launch", "map", "map_id", "world", "config", "semantic_graph")
_CAPABILITY_KEYS = ("packages", "nodes", "topics", "actions", "services", "transforms")


def get_roboclaw_home(home: str | Path | None = None) -> Path:
    """Return RoboClaw home directory, honoring ROBOCLAW_HOME."""
    if home is not None:
        return Path(home).expanduser()
    return Path(os.environ.get("ROBOCLAW_HOME", "~/.roboclaw")).expanduser()


def get_simulation_state_path(home: str | Path | None = None) -> Path:
    """Return the isolated simulation state path."""
    return get_roboclaw_home(home) / "workspace" / "embodied" / "simulation_state.json"


def default_simulation_state(
    profile: SimulationProfile | str | None = None,
) -> dict[str, Any]:
    """Return a fresh simulation state using the selected capability profile."""
    selected = _resolve_profile(profile)
    return {
        "version": STATE_VERSION,
        "mode": selected.mode,
        "profile_id": selected.profile_id,
        "robot": selected.robot,
        "simulator": selected.simulator,
        "paths": {key: "" for key in _PATH_KEYS},
        "capabilities": selected.capability_dict(),
        "sensors": [],
        "last_discovery": None,
        "last_doctor": None,
    }


def load_simulation_state(path: str | Path | None = None) -> dict[str, Any]:
    """Load simulation state, returning defaults when it has not been created."""
    target = Path(path).expanduser() if path is not None else get_simulation_state_path()
    if not target.exists():
        return default_simulation_state()
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("simulation_state.json must contain a JSON object.")
    state = _normalize_state(data)
    _validate_state(state)
    return state


def save_simulation_state(
    state: dict[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically persist simulation state."""
    normalized = _normalize_state(state)
    _validate_state(normalized)
    target = Path(path).expanduser() if path is not None else get_simulation_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, target)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return normalized


def sync_from_doctor_manifest(
    manifest: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return state updated with the latest simulation doctor manifest."""
    if not isinstance(manifest, dict):
        raise ValueError("doctor manifest must be a dict.")

    profile_id = str(manifest.get("profile_id") or DEFAULT_PROFILE.profile_id)
    current = _normalize_state(state or default_simulation_state(_profile_or_default(profile_id)))
    updated = copy.deepcopy(current)

    updated["version"] = STATE_VERSION
    updated["mode"] = str(manifest.get("mode") or updated["mode"])
    updated["profile_id"] = profile_id
    updated["robot"] = str(manifest.get("robot") or updated["robot"])
    updated["simulator"] = str(manifest.get("simulator") or updated["simulator"])

    checks = manifest.get("checks", {})
    if isinstance(checks, dict):
        updated["capabilities"] = _capabilities_from_checks(checks, updated["capabilities"])
        updated["sensors"] = _sensors_from_topics(checks.get("topics", {}))

    updated["last_discovery"] = copy.deepcopy(manifest)
    updated["last_doctor"] = {
        "status": copy.deepcopy(manifest.get("status", {})),
        "decision": manifest.get("decision"),
        "next_steps": _copy_list(manifest.get("next_steps", [])),
        "errors": _copy_list(manifest.get("errors", [])),
    }

    _validate_state(updated)
    return updated


def _resolve_profile(profile: SimulationProfile | str | None) -> SimulationProfile:
    if profile is None:
        return default_profile()
    if isinstance(profile, SimulationProfile):
        return profile
    return get_profile(profile)


def _profile_or_default(profile_id: str) -> SimulationProfile:
    try:
        return get_profile(profile_id)
    except ValueError:
        return default_profile()


def _normalize_state(data: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(data.get("profile_id") or DEFAULT_PROFILE.profile_id)
    base = default_simulation_state(_profile_or_default(profile_id))
    normalized = _deep_merge(base, data)
    normalized["version"] = int(normalized.get("version", STATE_VERSION))
    normalized["profile_id"] = profile_id
    normalized["paths"] = _normalize_string_map(normalized.get("paths", {}), _PATH_KEYS)
    normalized["capabilities"] = _normalize_capabilities(normalized.get("capabilities", {}))
    normalized["sensors"] = _normalize_string_list(normalized.get("sensors", []))
    return normalized


def _validate_state(state: dict[str, Any]) -> None:
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"Unsupported simulation state version: {state.get('version')}")
    for key in ("mode", "profile_id", "robot", "simulator"):
        if not isinstance(state.get(key), str) or not state[key]:
            raise ValueError(f"simulation state field '{key}' must be a non-empty string.")
    if not isinstance(state.get("paths"), dict):
        raise ValueError("simulation state field 'paths' must be a dict.")
    if not isinstance(state.get("capabilities"), dict):
        raise ValueError("simulation state field 'capabilities' must be a dict.")
    if not isinstance(state.get("sensors"), list):
        raise ValueError("simulation state field 'sensors' must be a list.")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_string_map(value: Any, keys: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    return {key: str(value.get(key, "") or "") for key in keys}


def _normalize_capabilities(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        value = {}
    return {
        key: _normalize_string_list(value.get(key, []))
        for key in _CAPABILITY_KEYS
    }


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _copy_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return copy.deepcopy(value)


def _capabilities_from_checks(
    checks: dict[str, Any],
    fallback: dict[str, list[str]],
) -> dict[str, list[str]]:
    capabilities = copy.deepcopy(fallback)
    for key in _CAPABILITY_KEYS:
        check_key = "tf" if key == "transforms" else key
        values = checks.get(check_key)
        if isinstance(values, dict):
            capabilities[key] = list(values.keys())
    return _normalize_capabilities(capabilities)


def _sensors_from_topics(topics: Any) -> list[str]:
    if not isinstance(topics, dict):
        return []
    present = {topic for topic, ok in topics.items() if ok}
    sensors: list[str] = []
    if "/scan" in present:
        sensors.append("lidar")
    if "/imu" in present or "/imu/data" in present:
        sensors.append("imu")
    if any("image" in topic or "camera" in topic for topic in present):
        sensors.append("camera")
    return sensors
