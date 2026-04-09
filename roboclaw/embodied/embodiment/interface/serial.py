from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.embodiment.interface.base import Interface

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.hardware.motion_detector import MotionDetector


@dataclass(frozen=True)
class SerialInterface(Interface):
    """A serial (USB-UART) hardware interface."""

    dev: str = ""  # /dev/ttyACM0
    by_id: str = ""  # /dev/serial/by-id/usb-1a86_...
    by_path: str = ""  # /dev/serial/by-path/pci-...
    bus_type: str = ""  # "feetech" / "dynamixel" / "modbus"
    motor_ids: tuple[int, ...] = ()
    interface_type: str = field(default="serial", init=False)

    def __post_init__(self) -> None:
        from roboclaw.embodied.embodiment.hardware.motion_detector import MotionDetector

        object.__setattr__(self, "_motion_detector", MotionDetector(self))

    @property
    def motion_detector(self) -> MotionDetector:
        return object.__getattribute__(self, "_motion_detector")

    @property
    def label(self) -> str:
        """Human-readable short label: last segment of by_id, fallback to dev."""
        if self.by_id:
            tail = self.by_id.rsplit("/", 1)[-1]
            return tail or self.dev or "?"
        return self.dev or "?"

    @property
    def address(self) -> str:
        return self.by_id or self.by_path or self.dev

    @property
    def stable_id(self) -> str:
        return self.by_id or self.by_path or self.dev

    @property
    def exists(self) -> bool:
        addr = self.address
        return bool(addr) and os.path.exists(addr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dev": self.dev,
            "by_id": self.by_id,
            "by_path": self.by_path,
            "label": self.label,
            "bus_type": self.bus_type,
            "motor_ids": list(self.motor_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerialInterface:
        return cls(
            dev=data.get("dev", ""),
            by_id=data.get("by_id", ""),
            by_path=data.get("by_path", ""),
            bus_type=data.get("bus_type", ""),
            motor_ids=tuple(data.get("motor_ids", ())),
        )
