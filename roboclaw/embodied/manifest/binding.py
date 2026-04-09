"""Binding — links an embodiment spec to a physical interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roboclaw.embodied.guard import InterfaceGuard
from roboclaw.embodied.interface.base import Interface
from roboclaw.embodied.interface.serial import SerialInterface
from roboclaw.embodied.interface.video import VideoInterface


@dataclass
class Binding:
    """A named link between a device type and a physical Interface.

    Bindings are the runtime representation of manifest entries.
    Each Binding is guarded by an InterfaceGuard for mutual exclusion.
    """

    alias: str
    interface: Interface
    guard: InterfaceGuard
    spec: Any = None              # HandSpec for hands, None for arms/cameras
    calibration_dir: str = ""     # arm-specific
    calibrated: bool = False      # arm-specific
    slave_id: int = 0             # hand-specific
    _kind: str = field(default="", repr=False)
    _type_name: str = field(default="", repr=False)

    @property
    def port(self) -> str:
        return self.interface.address

    @property
    def type_name(self) -> str:
        return self._type_name

    @property
    def arm_id(self) -> str:
        if not self.calibration_dir:
            return ""
        return Path(self.calibration_dir).name

    @property
    def is_follower(self) -> bool:
        return self.kind == "arm" and "follower" in self.type_name

    @property
    def is_leader(self) -> bool:
        return self.kind == "arm" and "leader" in self.type_name

    @property
    def connected(self) -> bool:
        return self.interface.exists

    @property
    def kind(self) -> str:
        return self._kind

    # ── Serialization (backward-compatible with manifest.json) ────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to manifest-compatible dict format."""
        if isinstance(self.interface, SerialInterface):
            return self._arm_or_hand_dict()
        if isinstance(self.interface, VideoInterface):
            return self._camera_dict()
        raise ValueError(f"Unknown interface type: {type(self.interface)}")

    def _arm_or_hand_dict(self) -> dict[str, Any]:
        """Serialize arm or hand binding."""
        if self._kind == "hand":
            return {
                "alias": self.alias,
                "type": self.type_name,
                "port": self.interface.address,
                "slave_id": self.slave_id,
            }
        # Arm
        return {
            "alias": self.alias,
            "type": self.type_name,
            "port": self.interface.address,
            "calibration_dir": self.calibration_dir,
            "calibrated": self.calibrated,
        }

    def _camera_dict(self) -> dict[str, Any]:
        """Serialize camera binding."""
        assert isinstance(self.interface, VideoInterface)
        d: dict[str, Any] = {
            "alias": self.alias,
            "port": self.interface.address,
            "width": self.interface.width,
            "height": self.interface.height,
        }
        if self.interface.fps:
            d["fps"] = self.interface.fps
        if self.interface.fourcc:
            d["fourcc"] = self.interface.fourcc
        return d

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        kind: str,
        guards: dict[str, InterfaceGuard],
    ) -> Binding:
        """Reconstruct a Binding from a manifest dict entry.

        Args:
            data: The manifest entry dict (arm, camera, or hand).
            kind: One of "arm", "camera", "hand".
            guards: Shared guard registry keyed by interface.stable_id.
                    New guards are created and inserted if not present.
        """
        if kind == "camera":
            return cls._camera_from_dict(data, guards)
        if kind == "hand":
            return cls._hand_from_dict(data, guards)
        if kind == "arm":
            return cls._arm_from_dict(data, guards)
        raise ValueError(f"Unknown binding kind: {kind!r}")

    @classmethod
    def _arm_from_dict(
        cls, data: dict[str, Any], guards: dict[str, InterfaceGuard],
    ) -> Binding:
        arm_type = data["type"]
        interface = SerialInterface(
            by_id=data.get("port", ""),
            dev=data.get("dev", ""),
        )
        guard = _ensure_guard(interface, guards)
        return cls(
            alias=data["alias"],
            interface=interface,
            guard=guard,
            calibration_dir=data.get("calibration_dir", ""),
            calibrated=data.get("calibrated", False),
            _kind="arm",
            _type_name=arm_type,
        )

    @classmethod
    def _hand_from_dict(
        cls, data: dict[str, Any], guards: dict[str, InterfaceGuard],
    ) -> Binding:
        from roboclaw.embodied.embodiment.hand.registry import get_hand_spec

        spec = get_hand_spec(data["type"])
        interface = SerialInterface(by_id=data.get("port", ""))
        guard = _ensure_guard(interface, guards)
        return cls(
            alias=data["alias"],
            spec=spec,
            interface=interface,
            guard=guard,
            slave_id=data.get("slave_id", 0),
            _kind="hand",
            _type_name=data["type"],
        )

    @classmethod
    def _camera_from_dict(
        cls, data: dict[str, Any], guards: dict[str, InterfaceGuard],
    ) -> Binding:
        interface = VideoInterface(
            dev=data.get("port", ""),
            width=data.get("width", 640),
            height=data.get("height", 480),
            fps=data.get("fps", 30),
            fourcc=data.get("fourcc", ""),
        )
        guard = _ensure_guard(interface, guards)
        return cls(
            alias=data["alias"],
            interface=interface,
            guard=guard,
            _kind="camera",
            _type_name="opencv",
        )


def _ensure_guard(
    interface: Interface, guards: dict[str, InterfaceGuard],
) -> InterfaceGuard:
    """Get or create a guard for the interface, keyed by stable_id."""
    key = interface.stable_id
    if not key:
        # No stable identity — guard cannot be shared across bindings.
        # This happens with degenerate config entries (no dev/by_id/by_path).
        from loguru import logger
        logger.warning("Interface has no stable_id, guard will not be shared: {}", interface)
        return InterfaceGuard(interface)
    if key not in guards:
        guards[key] = InterfaceGuard(interface)
    return guards[key]
