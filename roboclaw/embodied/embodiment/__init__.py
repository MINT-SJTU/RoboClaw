"""Embodiment — static spec hierarchy for all hardware types."""

from roboclaw.embodied.embodiment.base import EmbodimentSpec
from roboclaw.embodied.embodiment.arm.base import ArmSpec, MotorArmSpec, ServoArmSpec
from roboclaw.embodied.embodiment.hand.base import HandSpec
from roboclaw.embodied.embodiment.humanoid import HumanoidSpec
from roboclaw.embodied.embodiment.wheeled import WheeledSpec
from roboclaw.embodied.sensor.base import CameraSpec, SensorSpec

__all__ = [
    "EmbodimentSpec",
    "ArmSpec",
    "ServoArmSpec",
    "MotorArmSpec",
    "HandSpec",
    "HumanoidSpec",
    "WheeledSpec",
    "SensorSpec",
    "CameraSpec",
]
