"""Manifest — centralized state guardian for robot hardware configuration."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from roboclaw.embodied.board.board import Board

from roboclaw.embodied.board.channels import CH_CONFIG
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest.binding import (
    ArmBinding,
    Binding,
    CameraBinding,
    HandBinding,
    load_binding,
    validate_arm_side,
)
from roboclaw.embodied.embodiment.manifest.guard import InterfaceGuard
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
        self._file_mtime: float = 0.0
        self._last_sync_check: float = 0.0
        self._load()

    # ── Internal I/O ──────────────────────────────────────────────────

    _SYNC_INTERVAL = 0.5  # seconds — skip stat() if checked recently

    def _file_mtime_on_disk(self) -> float:
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _maybe_reload(self) -> None:
        """Reload from disk if another process has written the file.

        Must be called with self._lock held.
        """
        now = time.monotonic()
        if now - self._last_sync_check < self._SYNC_INTERVAL:
            return
        self._last_sync_check = now
        current = self._file_mtime_on_disk()
        if current > self._file_mtime:
            self._load()

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
        scanned_cameras: list[VideoInterface] | None = None
        if data.get("cameras"):
            from roboclaw.embodied.embodiment.hardware.scan import scan_cameras

            scanned_cameras = scan_cameras()
        for kind in ("arms", "cameras", "hands"):
            for item in data.get(kind, []):
                binding = load_binding(
                    item,
                    kind[:-1],
                    self._guards,
                    scanned_cameras=scanned_cameras if kind == "cameras" else None,
                )
                self._bindings[binding.alias] = binding
        self._file_mtime = self._file_mtime_on_disk()

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
            self._file_mtime = self._file_mtime_on_disk()
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @contextmanager
    def _rollback_bindings_on_error(self) -> Iterator[None]:
        """Restore the prior ``_bindings`` snapshot if the block raises."""
        snapshot = dict(self._bindings)
        try:
            yield
        except Exception:
            self._bindings = snapshot
            raise

    def _emit(self, change_type: str, device_alias: str = "") -> None:
        """Emit config change via Board. Fire-and-forget in async context."""
        if not self._board:
            return
        self._board.emit_sync(CH_CONFIG, {
            "change_type": change_type,
            "device_alias": device_alias,
            "timestamp": time.time(),
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
            if isinstance(binding, ArmBinding):
                grouped["arms"].append(binding.to_dict())
            elif isinstance(binding, CameraBinding):
                grouped["cameras"].append(binding.to_dict())
            else:
                grouped["hands"].append(binding.to_dict())
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

    def _require_arm(self, alias: str) -> ArmBinding:
        binding = self._bindings.get(alias)
        if not isinstance(binding, ArmBinding):
            raise ValueError(f"No arm with alias '{alias}' in manifest.")
        return binding

    def _require_camera(self, alias: str) -> CameraBinding:
        binding = self._bindings.get(alias)
        if not isinstance(binding, CameraBinding):
            raise ValueError(f"No camera with alias '{alias}' in manifest.")
        return binding

    def _require_hand(self, alias: str) -> HandBinding:
        binding = self._bindings.get(alias)
        if not isinstance(binding, HandBinding):
            raise ValueError(f"No hand with alias '{alias}' in manifest.")
        return binding

    def _store_binding(self, binding: Binding) -> None:
        existing = self._bindings.get(binding.alias)
        if existing is not None and type(existing) is not type(binding):
            existing_kind = _binding_kind_name(existing)
            new_kind = _binding_kind_name(binding)
            raise ValueError(
                f"Alias '{binding.alias}' is already assigned to {existing_kind} '{binding.alias}', "
                f"not {new_kind}."
            )
        self._bindings[binding.alias] = binding

    # ── Read properties ───────────────────────────────────────────────

    @property
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_reload()
            return self._snapshot_unlocked()

    @property
    def arms(self) -> list[ArmBinding]:
        with self._lock:
            self._maybe_reload()
            return [binding for binding in self._bindings.values() if isinstance(binding, ArmBinding)]

    @property
    def cameras(self) -> list[CameraBinding]:
        with self._lock:
            self._maybe_reload()
            return [binding for binding in self._bindings.values() if isinstance(binding, CameraBinding)]

    @property
    def hands(self) -> list[HandBinding]:
        with self._lock:
            self._maybe_reload()
            return [binding for binding in self._bindings.values() if isinstance(binding, HandBinding)]

    @property
    def bindings(self) -> list[Binding]:
        with self._lock:
            self._maybe_reload()
            return list(self._bindings.values())

    def find_arm(self, alias: str) -> ArmBinding | None:
        with self._lock:
            self._maybe_reload()
            binding = self._bindings.get(alias)
            if not isinstance(binding, ArmBinding):
                return None
            return binding

    def find_camera(self, alias: str) -> CameraBinding | None:
        with self._lock:
            self._maybe_reload()
            binding = self._bindings.get(alias)
            if not isinstance(binding, CameraBinding):
                return None
            return binding

    def find_hand(self, alias: str) -> HandBinding | None:
        with self._lock:
            self._maybe_reload()
            binding = self._bindings.get(alias)
            if not isinstance(binding, HandBinding):
                return None
            return binding

    def find_binding(self, alias: str) -> Binding | None:
        with self._lock:
            self._maybe_reload()
            return self._bindings.get(alias)

    def get_guard(self, port: str) -> InterfaceGuard | None:
        return self._guards.get(port)

    # ── Write methods ─────────────────────────────────────────────────

    def set_arm(
        self, alias: str, arm_type: str, interface: SerialInterface, side: str = "",
    ) -> ArmBinding:
        if arm_type not in _ARM_TYPES:
            raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
        if not alias:
            raise ValueError("Arm alias is required.")
        validate_arm_side(side, alias)
        port = interface.address
        if not port:
            raise ValueError("Arm interface has no usable address.")

        serial = _extract_serial_number(port)
        calibration_dir = get_calibration_root() / serial
        _migrate_none_calibration_file(calibration_dir, serial)

        with self._lock:
            for binding in self._bindings.values():
                if not isinstance(binding, ArmBinding):
                    continue
                if binding.alias != alias and binding.port == port:
                    raise ValueError(
                        f"Port '{port}' is already assigned to arm '{binding.alias}'."
                    )
            binding = ArmBinding(
                alias=alias,
                interface=interface,
                guard=self._guard_for_binding(interface),
                arm_type=arm_type,
                calibration_dir=str(calibration_dir),
                calibrated=_has_calibration_file(calibration_dir, serial),
                side=side,
            )
            with self._rollback_bindings_on_error():
                self._store_binding(binding)
                self._persist()

        self._emit("arm_added", alias)
        return binding

    def remove_arm(self, alias: str) -> dict[str, Any]:
        with self._lock:
            arm = self._require_arm(alias)
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
            arm = self._require_arm(old_alias)
            existing = self._bindings.get(new_alias)
            if old_alias != new_alias and existing is not None:
                raise ValueError(f"Alias '{new_alias}' already exists.")
            with self._rollback_bindings_on_error():
                renamed = arm.renamed(new_alias)
                del self._bindings[old_alias]
                self._bindings[new_alias] = renamed
                self._persist()
            result = self._snapshot_unlocked()

        self._emit("arm_renamed", new_alias)
        return result

    def mark_arm_calibrated(self, alias: str) -> dict[str, Any]:
        with self._lock:
            with self._rollback_bindings_on_error():
                arm = self._require_arm(alias)
                self._bindings[alias] = arm.with_calibrated()
                self._persist()
            result = self._snapshot_unlocked()

        refresh_bimanual_cal_dirs(result)
        self._emit("arm_calibrated", alias)
        return result

    def set_camera(
        self, name: str, interface: VideoInterface, side: str = "",
    ) -> CameraBinding:
        if not name:
            raise ValueError("Camera alias is required.")
        from roboclaw.embodied.embodiment.manifest.binding import validate_camera_side
        validate_camera_side(side, name)
        if side and not name.startswith(f"{side}_"):
            raise ValueError(
                f"Camera alias '{name}' must start with '{side}_' to match its side."
            )
        port = interface.address
        if not port:
            raise ValueError("Camera interface has no usable address.")

        with self._lock:
            binding = CameraBinding(
                alias=name,
                interface=interface,
                guard=self._guard_for_binding(interface),
                side=side,
            )
            self._store_binding(binding)
            self._persist()

        self._emit("camera_added", name)
        return binding

    def remove_camera(self, name: str) -> dict[str, Any]:
        with self._lock:
            cam = self._require_camera(name)
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
            camera = self._require_camera(old_name)
            existing = self._bindings.get(new_name)
            if old_name != new_name and existing is not None:
                raise ValueError(f"Alias '{new_name}' already exists.")
            if camera.side and not new_name.startswith(f"{camera.side}_"):
                raise ValueError(
                    f"Camera alias '{new_name}' must start with '{camera.side}_' to match its side."
                )
            renamed = camera.renamed(new_name)
            del self._bindings[old_name]
            self._bindings[new_name] = renamed
            self._persist()
            result = self._snapshot_unlocked()

        self._emit("camera_renamed", new_name)
        return result

    def set_hand(
        self, alias: str, hand_type: str,
        interface: SerialInterface, slave_id: int,
    ) -> HandBinding:
        if hand_type not in _HAND_TYPES:
            raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of {_HAND_TYPES}.")
        if not alias:
            raise ValueError("Hand alias is required.")
        port = interface.address
        if not port:
            raise ValueError("Hand interface has no usable address.")

        from roboclaw.embodied.embodiment.hand.registry import get_hand_spec

        with self._lock:
            binding = HandBinding(
                alias=alias,
                interface=interface,
                guard=self._guard_for_binding(interface),
                hand_type=hand_type,
                spec=get_hand_spec(hand_type),
                slave_id=slave_id,
            )
            self._store_binding(binding)
            self._persist()

        self._emit("hand_added", alias)
        return binding

    def remove_hand(self, alias: str) -> dict[str, Any]:
        with self._lock:
            hand = self._require_hand(alias)
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
            hand = self._require_hand(old_alias)
            existing = self._bindings.get(new_alias)
            if old_alias != new_alias and existing is not None:
                raise ValueError(f"Alias '{new_alias}' already exists.")
            renamed = hand.renamed(new_alias)
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


def _binding_kind_name(binding: Binding) -> str:
    if isinstance(binding, ArmBinding):
        return "arm"
    if isinstance(binding, CameraBinding):
        return "camera"
    return "hand"
