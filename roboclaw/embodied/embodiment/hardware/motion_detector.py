"""MotionDetector — detects physical movement on a SerialInterface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from roboclaw.embodied.embodiment.hardware.motion import MOTION_THRESHOLD, detect_motion

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.interface.serial import SerialInterface


@dataclass
class MotionResult:
    """Result of a motion poll."""

    delta: int
    moved: bool
    positions: dict[int, int]


class MotionDetector:
    """Detects physical movement on a SerialInterface by comparing servo positions."""

    def __init__(self, interface: SerialInterface) -> None:
        self._interface = interface
        self._baseline: dict[int, int] = {}
        self._last_positions: dict[int, int] = {}

    @property
    def interface(self) -> SerialInterface:
        return self._interface

    @property
    def has_baseline(self) -> bool:
        return len(self._baseline) > 0

    def capture_baseline(self) -> dict[int, int]:
        """Read and store current positions as the motion reference."""
        self._baseline = self._read_positions()
        self._last_positions = dict(self._baseline)
        return dict(self._baseline)

    def poll(self) -> MotionResult:
        """Read current positions and compute fresh motion from the previous poll.

        This detector only answers whether a new movement happened in this
        sampling window. Higher-level setup flows decide whether that fresh
        motion should latch the currently selected arm.
        """
        current = self._read_positions()
        reference = self._last_positions or self._baseline
        delta = detect_motion(reference, current)
        self._last_positions = dict(current)
        return MotionResult(delta=delta, moved=delta > MOTION_THRESHOLD, positions=current)

    def reset(self) -> None:
        """Clear the stored baseline."""
        self._baseline = {}
        self._last_positions = {}

    def _read_positions(self) -> dict[int, int]:
        """Read servo positions via the appropriate prober."""
        from roboclaw.embodied.embodiment.hardware.probers import get_prober

        path = self._interface.dev or self._interface.by_id or self._interface.by_path
        bus_type = self._interface.bus_type or "feetech"
        prober = get_prober(bus_type)
        return prober.read_positions(path, list(self._interface.motor_ids))
