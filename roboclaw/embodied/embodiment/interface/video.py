from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.embodiment.interface.base import Interface


@dataclass(frozen=True)
class VideoInterface(Interface):
    """A video capture hardware interface."""

    dev: str = ""  # Runtime source: /dev/video0 on Linux, AVFoundation index on macOS
    by_id: str = ""
    by_path: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = ""
    interface_type: str = field(default="video", init=False)

    @property
    def is_index_device(self) -> bool:
        return self.dev.isdigit()

    @property
    def label(self) -> str:
        """Human-readable short label: by_path preferred for cameras (stable across reboots)."""
        if self.is_index_device and self.by_path:
            return self.by_path
        best = self.by_path or self.by_id
        if best:
            return best.rsplit("/", 1)[-1] or self.dev or "?"
        return self.dev or "?"

    @property
    def address(self) -> str:
        return self.stable_id

    @property
    def runtime_address(self) -> str:
        return self.dev or self.by_path or self.by_id

    @property
    def preview_address(self) -> str:
        if self.is_index_device and self.by_id:
            return self.by_id
        return self.runtime_address

    @property
    def stable_id(self) -> str:
        return self.by_id or self.by_path or self.dev

    @property
    def exists(self) -> bool:
        addr = self.runtime_address
        if addr.isdigit():
            return True
        return bool(addr) and os.path.exists(addr)

    def matches(self, reference: str) -> bool:
        if not reference:
            return False
        return reference in {self.stable_id, self.by_id, self.by_path, self.dev}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dev": self.dev,
            "by_id": self.by_id,
            "by_path": self.by_path,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
        }

    @classmethod
    def from_stable_address(cls, address: str, **kwargs: Any) -> VideoInterface:
        if address.startswith("/dev/v4l/by-id/"):
            return cls(by_id=address, **kwargs)
        if address.startswith("/dev/v4l/by-path/"):
            return cls(by_path=address, **kwargs)
        if address.startswith("/dev/"):
            return cls(dev=address, **kwargs)
        return cls(by_id=address, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoInterface:
        return cls(
            dev=data.get("dev", ""),
            by_id=data.get("by_id", ""),
            by_path=data.get("by_path", ""),
            width=data.get("width", 640),
            height=data.get("height", 480),
            fps=data.get("fps", 30),
            fourcc=data.get("fourcc", ""),
        )
