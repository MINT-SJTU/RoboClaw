from __future__ import annotations

from roboclaw.embodied.embodiment.interface.base import Interface
from roboclaw.embodied.embodiment.interface.can import CANInterface
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface

__all__ = ["Interface", "SerialInterface", "VideoInterface", "CANInterface"]
