from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.interface.base import Interface


@dataclass(frozen=True)
class SerialInterface(Interface):
    """A serial (USB-UART) hardware interface."""

    dev: str = ""  # /dev/ttyACM0
    by_id: str = ""  # /dev/serial/by-id/usb-1a86_...
    by_path: str = ""  # /dev/serial/by-path/pci-...
    bus_type: str = ""  # "feetech" / "dynamixel" / "modbus"
    motor_ids: tuple[int, ...] = ()
    interface_type: str = field(default="serial", init=False)

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
