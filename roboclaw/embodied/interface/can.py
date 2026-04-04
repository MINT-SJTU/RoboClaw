from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.interface.base import Interface


@dataclass(frozen=True)
class CANInterface(Interface):
    """A CAN bus hardware interface (skeleton)."""

    channel: str = ""  # "can0"
    bitrate: int = 1_000_000
    interface_type: str = field(default="can", init=False)

    @property
    def address(self) -> str:
        return self.channel

    @property
    def stable_id(self) -> str:
        return self.channel

    @property
    def exists(self) -> bool:
        return bool(self.channel) and os.path.exists(
            f"/sys/class/net/{self.channel}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "bitrate": self.bitrate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CANInterface:
        return cls(
            channel=data.get("channel", ""),
            bitrate=data.get("bitrate", 1_000_000),
        )
