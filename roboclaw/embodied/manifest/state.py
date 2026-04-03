"""Manifest — centralized state guardian for robot hardware configuration."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from roboclaw.embodied.events import ConfigChangedEvent, EventBus
from roboclaw.embodied.manifest.helpers import (
    _ARM_TYPES,
    _HAND_TYPES,
    _default_manifest,
    _ensure_unique_port,
    _extract_serial_number,
    _has_calibration_file,
    _migrate_none_calibration_file,
    _probe_hand_slave_id,
    _refresh_calibration_state,
    _resolve_port,
    _validate_manifest,
    find_arm,
    find_camera,
    find_hand,
    get_calibration_root,
    get_manifest_path,
    refresh_bimanual_cal_dirs,
)


class Manifest:
    """Single guardian of robot hardware configuration.

    Holds manifest state in memory. All reads return deep copies (zero I/O).
    All writes are locked, validated, atomically persisted, and emit events.
    """

    def __init__(
        self,
        path: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._path = path or get_manifest_path()
        self._bus = event_bus
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    # ── Internal I/O ──────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Load from disk. Migrate setup.json -> manifest.json if needed."""
        if not self._path.exists():
            legacy = self._path.parent / "setup.json"
            if legacy.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(legacy), str(self._path))
        if not self._path.exists():
            return _default_manifest()
        data = json.loads(self._path.read_text(encoding="utf-8"))
        data.pop("scanned_ports", None)
        data.pop("scanned_cameras", None)
        if isinstance(data.get("cameras"), dict):
            data["cameras"] = []
        _refresh_calibration_state(data)
        return data

    def _persist(self) -> None:
        """Atomic write: tempfile + os.replace."""
        _validate_manifest(self._data)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, str(self._path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _emit(self, change_type: str, device_alias: str = "") -> None:
        """Emit ConfigChangedEvent. Fire-and-forget in async context."""
        if not self._bus:
            return
        import asyncio

        event = ConfigChangedEvent(change_type=change_type, device_alias=device_alias)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._bus.emit(event))
        except RuntimeError:
            pass  # no event loop running (CLI context)

    # ── Read properties (zero I/O, return deepcopy) ───────────────────

    @property
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    @property
    def arms(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data.get("arms", []))

    @property
    def cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data.get("cameras", []))

    @property
    def hands(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._data.get("hands", []))

    def find_arm(self, alias: str) -> dict | None:
        with self._lock:
            found = find_arm(self._data.get("arms", []), alias)
            return copy.deepcopy(found) if found else None

    def find_camera(self, alias: str) -> dict | None:
        with self._lock:
            found = find_camera(self._data.get("cameras", []), alias)
            return copy.deepcopy(found) if found else None

    def find_hand(self, alias: str) -> dict | None:
        with self._lock:
            found = find_hand(self._data.get("hands", []), alias)
            return copy.deepcopy(found) if found else None

    # ── Write methods (lock + validate + persist + emit) ──────────────

    def set_arm(
        self, alias: str, arm_type: str, port: str,
    ) -> dict[str, Any]:
        """Add or update an arm. Returns the arm entry dict."""
        if arm_type not in _ARM_TYPES:
            raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
        if not port:
            raise ValueError("Arm port is required.")
        if not alias:
            raise ValueError("Arm alias is required.")
        from roboclaw.embodied.hardware.scan import scan_serial_ports

        port = _resolve_port(port, scan_serial_ports())
        serial = _extract_serial_number(port)
        calibration_dir = get_calibration_root() / serial
        _migrate_none_calibration_file(calibration_dir, serial)

        with self._lock:
            arms = self._data.setdefault("arms", [])
            _ensure_unique_port(arms, alias, port)
            entry: dict[str, Any] = {
                "alias": alias,
                "type": arm_type,
                "port": port,
                "calibration_dir": str(calibration_dir),
                "calibrated": _has_calibration_file(calibration_dir, serial),
            }
            existing = find_arm(arms, alias)
            if existing is not None:
                arms[arms.index(existing)] = entry
            else:
                arms.append(entry)
            self._persist()
            result = copy.deepcopy(entry)

        self._emit("arm_added", alias)
        return result

    def remove_arm(self, alias: str) -> dict[str, Any]:
        """Remove arm by alias. Returns updated manifest dict."""
        with self._lock:
            arms = self._data.get("arms", [])
            arm = find_arm(arms, alias)
            if arm is None:
                raise ValueError(f"No arm with alias '{alias}' in manifest.")
            arms.remove(arm)
            self._persist()
            result = copy.deepcopy(self._data)

        self._emit("arm_removed", alias)
        return result

    def rename_arm(self, old_alias: str, new_alias: str) -> dict[str, Any]:
        """Rename arm. Returns updated manifest dict."""
        if not old_alias:
            raise ValueError("Old arm alias is required.")
        if not new_alias:
            raise ValueError("New arm alias is required.")

        with self._lock:
            arms = self._data.get("arms", [])
            arm = find_arm(arms, old_alias)
            if arm is None:
                raise ValueError(f"No arm with alias '{old_alias}' in manifest.")
            if old_alias != new_alias and find_arm(arms, new_alias) is not None:
                raise ValueError(f"Arm alias '{new_alias}' already exists.")
            arm["alias"] = new_alias
            self._persist()
            result = copy.deepcopy(self._data)

        self._emit("arm_renamed", new_alias)
        return result

    def mark_arm_calibrated(self, alias: str) -> dict[str, Any]:
        """Mark arm as calibrated. Returns updated manifest dict."""
        with self._lock:
            arms = self._data.get("arms", [])
            arm = find_arm(arms, alias)
            if arm is None:
                raise ValueError(f"No arm with alias '{alias}' in manifest.")
            arm["calibrated"] = True
            self._persist()
            data_copy = copy.deepcopy(self._data)

        refresh_bimanual_cal_dirs(data_copy)
        self._emit("arm_calibrated", alias)
        return data_copy

    def set_camera(self, name: str, camera_index: int) -> dict[str, Any]:
        """Add/update camera from scanned list. Returns the camera entry dict."""
        if not name:
            raise ValueError("Camera alias is required.")
        from roboclaw.embodied.hardware.scan import scan_cameras

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

        with self._lock:
            cameras = self._data.setdefault("cameras", [])
            existing = find_camera(cameras, name)
            if existing is not None:
                cameras[cameras.index(existing)] = entry
            else:
                cameras.append(entry)
            self._persist()
            result = copy.deepcopy(entry)

        self._emit("camera_added", name)
        return result

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove camera by alias. Returns updated manifest dict."""
        with self._lock:
            cameras = self._data.get("cameras", [])
            cam = find_camera(cameras, name)
            if cam is None:
                raise ValueError(f"No camera with alias '{name}' in manifest.")
            cameras.remove(cam)
            self._persist()
            result = copy.deepcopy(self._data)

        self._emit("camera_removed", name)
        return result

    def set_hand(
        self, alias: str, hand_type: str, port: str,
    ) -> dict[str, Any]:
        """Add/update hand. Returns the hand entry dict."""
        if hand_type not in _HAND_TYPES:
            raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of {_HAND_TYPES}.")
        if not port:
            raise ValueError("Hand port is required.")
        if not alias:
            raise ValueError("Hand alias is required.")
        from roboclaw.embodied.hardware.scan import scan_serial_ports

        port = _resolve_port(port, scan_serial_ports())
        slave_id = _probe_hand_slave_id(hand_type, port)

        entry: dict[str, Any] = {
            "alias": alias,
            "type": hand_type,
            "port": port,
            "slave_id": slave_id,
        }

        with self._lock:
            hands = self._data.setdefault("hands", [])
            existing = find_hand(hands, alias)
            if existing is not None:
                hands[hands.index(existing)] = entry
            else:
                hands.append(entry)
            self._persist()
            result = copy.deepcopy(entry)

        self._emit("hand_added", alias)
        return result

    def remove_hand(self, alias: str) -> dict[str, Any]:
        """Remove hand by alias. Returns updated manifest dict."""
        with self._lock:
            hands = self._data.get("hands", [])
            hand = find_hand(hands, alias)
            if hand is None:
                raise ValueError(f"No hand with alias '{alias}' in manifest.")
            hands.remove(hand)
            self._persist()
            result = copy.deepcopy(self._data)

        self._emit("hand_removed", alias)
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-read from disk. Use after identify subprocess writes."""
        with self._lock:
            self._data = self._load()

    def ensure(self) -> dict[str, Any]:
        """If manifest doesn't exist, create defaults and persist. Return snapshot."""
        with self._lock:
            if not self._path.exists():
                self._data = _default_manifest()
                self._persist()
            return copy.deepcopy(self._data)
