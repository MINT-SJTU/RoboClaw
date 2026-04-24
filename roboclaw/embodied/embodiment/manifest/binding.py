"""Typed manifest bindings for arm, hand, and camera devices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from roboclaw.embodied.embodiment.arm.registry import get_role
from roboclaw.embodied.embodiment.interface.base import Interface
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest.guard import InterfaceGuard

_VALID_SIDES = ("left", "right")
_TBinding = TypeVar("_TBinding", bound="Binding")


class ArmRole(str, Enum):
    FOLLOWER = "follower"
    LEADER = "leader"


def validate_side(side: str, alias: str = "", *, kind: str) -> None:
    """Raise ValueError if *side* is not '', 'left', or 'right'."""
    if side and side not in _VALID_SIDES:
        label = f" for {kind} {alias!r}" if alias else ""
        raise ValueError(
            f"Invalid {kind} side {side!r}{label}; "
            "expected 'left', 'right', or empty (single arm)."
        )


def validate_camera_side(side: str, alias: str = "") -> None:
    validate_side(side, alias, kind="camera")


def validate_arm_side(side: str, alias: str = "") -> None:
    validate_side(side, alias, kind="arm")


def infer_side_from_alias(alias: str) -> str:
    """Infer a device side from conventional left_/right_ aliases."""
    normalized = alias.strip().lower()
    for side in _VALID_SIDES:
        if normalized == side:
            return side
        if normalized.startswith(f"{side}_") or normalized.startswith(f"{side}-"):
            return side
    return ""


@dataclass(frozen=True)
class Binding:
    """Common immutable binding state shared by all device bindings."""

    alias: str
    interface: Interface
    guard: InterfaceGuard

    @property
    def port(self) -> str:
        return self.interface.address

    @property
    def connected(self) -> bool:
        return self.interface.exists

    def renamed(self: _TBinding, alias: str) -> _TBinding:
        return replace(self, alias=alias)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ArmBinding(Binding):
    """Typed manifest binding for a robot arm."""

    arm_type: str
    calibration_dir: str
    calibrated: bool = False
    side: str = ""

    def __post_init__(self) -> None:
        validate_arm_side(self.side, self.alias)

    @property
    def arm_id(self) -> str:
        if not self.calibration_dir:
            return ""
        return Path(self.calibration_dir).name

    @property
    def role(self) -> ArmRole:
        return ArmRole(get_role(self.arm_type))

    def with_calibrated(self, calibrated: bool = True) -> ArmBinding:
        return replace(self, calibrated=calibrated)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "alias": self.alias,
            "type": self.arm_type,
            "port": self.interface.address,
            "calibration_dir": self.calibration_dir,
            "calibrated": self.calibrated,
        }
        if self.side:
            data["side"] = self.side
        return data


@dataclass(frozen=True)
class HandBinding(Binding):
    """Typed manifest binding for a hand/end-effector."""

    hand_type: str
    spec: Any
    slave_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "type": self.hand_type,
            "port": self.interface.address,
            "slave_id": self.slave_id,
        }


@dataclass(frozen=True)
class CameraBinding(Binding):
    """Typed manifest binding for a camera."""

    side: str = ""

    def __post_init__(self) -> None:
        validate_camera_side(self.side, self.alias)

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.interface, VideoInterface)
        data: dict[str, Any] = {
            "alias": self.alias,
            "side": self.side,
            "port": self.interface.address,
            "width": self.interface.width,
            "height": self.interface.height,
        }
        if self.interface.fps:
            data["fps"] = self.interface.fps
        if self.interface.fourcc:
            data["fourcc"] = self.interface.fourcc
        return data


def load_binding(
    data: dict[str, Any],
    kind: str,
    guards: dict[str, InterfaceGuard],
    *,
    scanned_cameras: list[VideoInterface] | None = None,
) -> Binding:
    """Reconstruct a typed binding from a manifest dict entry."""
    if kind == "camera":
        return _camera_from_dict(data, guards, scanned_cameras=scanned_cameras)
    if kind == "hand":
        return _hand_from_dict(data, guards)
    if kind == "arm":
        return _arm_from_dict(data, guards)
    raise ValueError(f"Unknown binding kind: {kind!r}")


def _arm_from_dict(
    data: dict[str, Any],
    guards: dict[str, InterfaceGuard],
) -> ArmBinding:
    side = data.get("side", "") or infer_side_from_alias(data.get("alias", ""))
    validate_arm_side(side, data.get("alias", ""))
    interface = SerialInterface(
        by_id=data.get("port", ""),
        dev=data.get("dev", ""),
    )
    return ArmBinding(
        alias=data["alias"],
        interface=interface,
        guard=_ensure_guard(interface, guards),
        arm_type=data["type"],
        calibration_dir=data.get("calibration_dir", ""),
        calibrated=data.get("calibrated", False),
        side=side,
    )


def _hand_from_dict(
    data: dict[str, Any],
    guards: dict[str, InterfaceGuard],
) -> HandBinding:
    from roboclaw.embodied.embodiment.hand.registry import get_hand_spec

    interface = SerialInterface(by_id=data.get("port", ""))
    return HandBinding(
        alias=data["alias"],
        interface=interface,
        guard=_ensure_guard(interface, guards),
        hand_type=data["type"],
        spec=get_hand_spec(data["type"]),
        slave_id=data.get("slave_id", 0),
    )


def _camera_from_dict(
    data: dict[str, Any],
    guards: dict[str, InterfaceGuard],
    *,
    scanned_cameras: list[VideoInterface] | None = None,
) -> CameraBinding:
    side = data.get("side", "")
    validate_camera_side(side, data.get("alias", ""))
    if scanned_cameras is None:
        interface = VideoInterface.from_stable_address(
            data.get("port", ""),
            width=data.get("width", 640),
            height=data.get("height", 480),
            fps=data.get("fps", 30),
            fourcc=data.get("fourcc", ""),
        )
    else:
        from roboclaw.embodied.embodiment.hardware.scan import resolve_camera_interface

        resolved = resolve_camera_interface(data.get("port", ""), scanned_cameras)
        interface = replace(
            resolved,
            width=data.get("width", resolved.width),
            height=data.get("height", resolved.height),
            fps=data.get("fps", resolved.fps),
            fourcc=data.get("fourcc", resolved.fourcc),
        )
    return CameraBinding(
        alias=data["alias"],
        interface=interface,
        guard=_ensure_guard(interface, guards),
        side=side,
    )


def _ensure_guard(
    interface: Interface,
    guards: dict[str, InterfaceGuard],
) -> InterfaceGuard:
    """Get or create a guard for the interface, keyed by stable_id."""
    key = interface.stable_id
    if not key:
        from loguru import logger

        logger.warning("Interface has no stable_id, guard will not be shared: {}", interface)
        return InterfaceGuard(interface)
    if key not in guards:
        guards[key] = InterfaceGuard(interface)
    return guards[key]
