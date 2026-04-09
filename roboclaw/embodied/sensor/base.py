"""SensorSpec — base class for all sensor types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    """Base class for all sensor specifications."""

    name: str = ""


@dataclass(frozen=True)
class CameraSpec(SensorSpec):
    """Static specification for a camera type."""

    default_width: int = 640
    default_height: int = 480
    default_fps: int = 30
