"""Manifest — centralized state guardian for robot hardware configuration."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.embodied.board.board import Board

from roboclaw.embodied.board.channels import CH_CONFIG
from roboclaw.embodied.embodiment.manifest.guard import InterfaceGuard
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest.binding import Binding
from roboclaw.embodied.embodiment.manifest.helpers import (
    _ARM_TYPES,
    _HAND_TYPES,
    _default_manifest,
    _extract_serial_number,
    _has_calibration_file,
    _migrate_none_calibration_file,
    _refresh_calibration_state,
    _validate_manifest,
    get_calibration_root,
    get_manifest_path,
    refresh_bimanual_cal_dirs,
)


class Manifest:
    """Single guardian of robot hardware configuration."""

    def __init__(
        self,
        path: Path | None = None,
        board: "Board | None" = None,
    ) -> None:
        self._path = path or get_manifest_path()
        self._board = board
        self._lock = threading.Lock()
        self._guards: dict[str, InterfaceGuard] = {}
        self._version = 2
        self._datasets: dict[str, Any] = {}
        self._policies: dict[str, Any] = {}
        self._bindings: dict[str, Binding] = {}
        self._load()

    # ── Internal I/O ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load from disk. Migrate setup.json -> manifest.json if needed."""
        if not self._path.exists():
            legacy = self._path.parent / "setup.json"
            if legacy.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(legacy), str(self._path))

        data = _default_manifest()
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            data.pop("scanned_ports", None)
            data.pop("scanned_cameras", None)
            if isinstance(data.get("cameras"), dict):
                data["cameras"] = []
            _refresh_calibration_state(data)

        self._version = int(data.get("version", 2))
        self._datasets = dict(data.get("datasets", {}))
        self._policies = dict(data.get("policies", {}))
        self._bindings = {}
        for kind in ("arms", "cameras", "hands"):
            for item in data.get(kind, []):
                binding = Binding.from_dict(item, kind[:-1], self._guards)
                self._bindings[binding.alias] = binding

    def _persist(self) -> None:
        """Atomic write: tempfile + os.replace."""
        snapshot = self._snapshot_unlocked()
        _validate_manifest(snapshot)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, str(self._path))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _emit(self, change_type: str, device_alias: str = "") -> None:
        """Emit config change via Board. Fire-and-forget in async context."""
        if not self._board:
            return
        import time as _time

        self._board.emit_sync(CH_CONFIG, {
            "change_type": change_type,
            "device_alias": device_alias,
            "timestamp": _time.time(),
        })

    def _prune_guard(self, port: str) -> None:
        """Remove guard for *port* if no other device still references it."""
        if not port:
            return
        if any(binding.port == port for binding in self._bindings.values()):
            return
        self._guards.pop(port, None)

    def _binding_lists(self) -> dict[str, list[dict[str, Any]]]:
        grouped = {"arms": [], "cameras": [], "hands": []}
        for binding in self._bindings.values():
            grouped[f"{binding.kind}s"].append(binding.to_dict())
        return grouped

    def _snapshot_unlocked(self) -> dict[str, Any]:
        grouped = self._binding_lists()
        return {
            "version": self._version,
            "arms": grouped["arms"],
            "hands": grouped["hands"],
            "cameras": grouped["cameras"],
            "datasets": dict(self._datasets),
            "policies": dict(self._policies),
        }

    def _guard_for_binding(self, interface: SerialInterface | VideoInterface) -> InterfaceGuard:
        key = interface.stable_id
        if not key:
            return InterfaceGuard(interface)
        if key not in self._guards:
            self._guards[key] = InterfaceGuard(interface)
        return self._guards[key]

    def _require_binding(self, alias: str, kind: str) -> Binding:
        binding = self._bindings.get(alias)
        if binding is None or binding.kind != kind:
            raise ValueError(f"No {kind} with alias '{alias}' in manifest.")
        return binding

    def _store_binding(self, binding: Binding) -> None:
        existing = self._bindings.get(binding.alias)
        if existing is not None and existing.kind != binding.kind:
            raise ValueError(
                f"Alias '{binding.alias}' is already assigned to {existing.kind} '{binding.alias}'."
            )
        self._bindings[binding.alias] = binding

    # ── Read properties ───────────────────────────────────────────────

    @property
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    @property
    def arms(self) -> list[Binding]:
        with self._lock:
            return [binding for binding in self._bindings.values() if binding.kind == "arm"]

    @property
    def cameras(self) -> list[Binding]:
        with self._lock:
            return [binding for binding in self._bindings.values() if binding.kind == "camera"]

    @property
    def hands(self) -> list[Binding]:
        with self._lock:
            return [binding for binding in self._bindings.values() if binding.kind == "hand"]

    @property
    def bindings(self) -> list[Binding]:
        with self._lock:
            return list(self._bindings.values())

    def find_arm(self, alias: str) -> Binding | None:
        with self._lock:
            binding = self._bindings.get(alias)
            if binding is None or binding.kind != "arm":
                return None
            return binding

    def find_camera(self, alias: str) -> Binding | None:
        with self._lock:
            binding = self._bindings.get(alias)
            if binding is None or binding.kind != "camera":
                return None
            return binding

    def find_hand(self, alias: str) -> Binding | None:
        with self._lock:
            binding = self._bindings.get(alias)
            if binding is None or binding.kind != "hand":
                return None
            return binding

    def find_binding(self, alias: str) -> Binding | None:
        with self._lock:
            return self._bindings.get(alias)

    def get_guard(self, port: str) -> InterfaceGuard | None:
        return self._guards.get(port)

    # ── Write methods ─────────────────────────────────────────────────

    def set_arm(
        self, alias: str, arm_type: str, interface: SerialInterface,
    ) -> Binding:
        if arm_type not in _ARM_TYPES:
            raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
        if not alias:
            raise ValueError("Arm alias is required.")
        port = interface.address
        if not port:
            raise ValueError("Arm interface has no usable address.")

        serial = _extract_serial_number(port)
        calibration_dir = get_calibration_root() / serial
        _migrate_none_calibration_file(calibration_dir, serial)

        with self._lock:
            for binding in self._bindings.values():
                if binding.kind != "arm":
                    continue
                if binding.alias != alias and binding.port == port:
                    raise ValueError(
                        f"Port '{port}' is already assigned to arm '{binding.alias}'."
                    )
            binding = Binding(
                alias=alias,
                interface=interface,
                guard=self._guard_for_binding(interface),
                calibration_dir=str(calibration_dir),
                calibrated=_has_calibration_file(calibration_dir, serial),
                _kind="arm",
                _type_name=arm_type,
            )
            self._store_binding(binding)
            self._persist()

        self._emit("arm_added", alias)
        return binding

    def remove_arm(self, alias: str) -> dict[str, Any]:
        with self._lock:
            arm = self._require_binding(alias, "arm")
            del self._bindings[alias]
            self._persist()
            self._prune_guard(arm.port)
            result = self._snapshot_unlocked()

        self._emit("arm_removed", alias)
        return result

    def rename_arm(self, old_alias: str, new_alias: str) -> dict[str, Any]:
        if not old_alias:
            raise ValueError("Old arm alias is required.")
        if not new_alias:
            raise ValueError("New arm alias is required.")

        with self._lock:
            arm = self._require_binding(old_alias, "arm")
            existing = self._bindings.get(new_alias)
            if old_alias != new_alias and existing is not None:
                raise ValueError(f"Alias '{new_alias}' already exists.")
            renamed = Binding(
                alias=new_alias,
                spec=arm.spec,
                interface=arm.interface,
                guard=arm.guard,
                calibration_dir=arm.calibration_dir,
                calibrated=arm.calibrated,
                slave_id=arm.slave_id,
                _kind=arm.kind,
                _type_name=arm.type_name,
            )
            del self._bindings[old_alias]
            self._bindings[new_alias] = renamed
            self._persist()
            result = self._snapshot_unlocked()

        self._emit("arm_renamed", new_alias)
        return result

    def mark_arm_calibrated(self, alias: str) -> dict[str, Any]:
        with self._lock:
            arm = self._require_binding(alias, "arm")
            self._bindings[alias] = Binding(
                alias=arm.alias,
                spec=arm.spec,
                interface=arm.interface,
                guard=arm.guard,
                calibration_dir=arm.calibration_dir,
                calibrated=True,
                slave_id=arm.slave_id,
                _kind=arm.kind,
                _type_name=arm.type_name,
            )
            self._persist()
            result = self._snapshot_unlocked()

        refresh_bimanual_cal_dirs(result)
        self._emit("arm_calibrated", alias)
        return result

    def set_camera(self, name: str, interface: VideoInterface) -> Binding:
        if not name:
            raise ValueError("Camera alias is required.")
        port = interface.address
        if not port:
            raise ValueError("Camera interface has no usable address.")

        with self._lock:
            binding = Binding(
                alias=name,
                interface=interface,
                guard=self._guard_for_binding(interface),
                _kind="camera",
                _type_name="opencv",
            )
            self._store_binding(binding)
            self._persist()

        self._emit("camera_added", name)
        return binding

    def remove_camera(self, name: str) -> dict[str, Any]:
        with self._lock:
            cam = self._require_binding(name, "camera")
            del self._bindings[name]
            self._persist()
            self._prune_guard(cam.port)
            result = self._snapshot_unlocked()

        self._emit("camera_removed", name)
        return result

    def rename_camera(self, old_name: str, new_name: str) -> dict[str, Any]:
        if not old_name:
            raise ValueError("Old camera alias is required.")
        if not new_name:
            raise ValueError("New camera alias is required.")

        with self._lock:
            camera = self._require_binding(old_name, "camera")
            existing = self._bindings.get(new_name)
            if old_name != new_name and existing is not None:
                raise ValueError(f"Alias '{new_name}' already exists.")
            renamed = Binding(
                alias=new_name,
                spec=camera.spec,
                interface=camera.interface,
                guard=camera.guard,
                calibration_dir=camera.calibration_dir,
                calibrated=camera.calibrated,
                slave_id=camera.slave_id,
                _kind=camera.kind,
                _type_name=camera.type_name,
            )
            del self._bindings[old_name]
            self._bindings[new_name] = renamed
            self._persist()
            result = self._snapshot_unlocked()

        self._emit("camera_renamed", new_name)
        return result

    def set_hand(
        self, alias: str, hand_type: str,
        interface: SerialInterface, slave_id: int,
    ) -> Binding:
        if hand_type not in _HAND_TYPES:
            raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of {_HAND_TYPES}.")
        if not alias:
            raise ValueError("Hand alias is required.")
        port = interface.address
        if not port:
            raise ValueError("Hand interface has no usable address.")

        from roboclaw.embodied.embodiment.hand.registry import get_hand_spec

        with self._lock:
            binding = Binding(
                alias=alias,
                spec=get_hand_spec(hand_type),
                interface=interface,
                guard=self._guard_for_binding(interface),
                slave_id=slave_id,
                _kind="hand",
                _type_name=hand_type,
            )
            self._store_binding(binding)
            self._persist()

        self._emit("hand_added", alias)
        return binding

    def remove_hand(self, alias: str) -> dict[str, Any]:
        with self._lock:
            hand = self._require_binding(alias, "hand")
            del self._bindings[alias]
            self._persist()
            self._prune_guard(hand.port)
            result = self._snapshot_unlocked()

        self._emit("hand_removed", alias)
        return result

    def rename_hand(self, old_alias: str, new_alias: str) -> dict[str, Any]:
        if not old_alias:
            raise ValueError("Old hand alias is required.")
        if not new_alias:
            raise ValueError("New hand alias is required.")

        with self._lock:
            hand = self._require_binding(old_alias, "hand")
            existing = self._bindings.get(new_alias)
            if old_alias != new_alias and existing is not None:
                raise ValueError(f"Alias '{new_alias}' already exists.")
            renamed = Binding(
                alias=new_alias,
                spec=hand.spec,
                interface=hand.interface,
                guard=hand.guard,
                calibration_dir=hand.calibration_dir,
                calibrated=hand.calibrated,
                slave_id=hand.slave_id,
                _kind=hand.kind,
                _type_name=hand.type_name,
            )
            del self._bindings[old_alias]
            self._bindings[new_alias] = renamed
            self._persist()
            result = self._snapshot_unlocked()

        self._emit("hand_renamed", new_alias)
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reload(self) -> None:
        with self._lock:
            self._load()

    def ensure(self) -> dict[str, Any]:
        with self._lock:
            if not self._path.exists():
                self._version = 2
                defaults = _default_manifest()
                self._datasets = dict(defaults["datasets"])
                self._policies = dict(defaults["policies"])
                self._bindings = {}
                self._persist()
            return self._snapshot_unlocked()
