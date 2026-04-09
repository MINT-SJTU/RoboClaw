from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.embodiment.interface.base import Interface


@dataclass(frozen=True)
class VideoInterface(Interface):
    """A video capture hardware interface."""

    dev: str = ""  # /dev/video0
    by_id: str = ""
    by_path: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = ""
    interface_type: str = field(default="video", init=False)

    @property
    def label(self) -> str:
        """Human-readable short label: last segment of by_id, fallback to dev."""
        if self.by_id:
            tail = self.by_id.rsplit("/", 1)[-1]
            return tail or self.dev or "?"
        return self.dev or "?"

    @property
    def address(self) -> str:
        return self.by_path or self.by_id or self.dev

    @property
    def stable_id(self) -> str:
        return self.by_path or self.by_id or self.dev

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
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
        }

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
