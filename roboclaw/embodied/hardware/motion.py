"""Motion detection and port resolution helpers."""
from __future__ import annotations

from roboclaw.embodied.hardware.probers import get_prober
from roboclaw.embodied.interface.serial import SerialInterface

MOTION_THRESHOLD = 50


def detect_motion(baseline: dict[int, int], current: dict[int, int]) -> int:
    """Compute total absolute delta between baseline and current positions."""
    total = 0
    for mid, base_val in baseline.items():
        cur_val = current.get(mid)
        if cur_val is None:
            continue
        total += abs(cur_val - base_val)
    return total


def read_positions_for_port(port: SerialInterface) -> dict[int, int]:
    """Read positions using the correct protocol prober for a probed port."""
    path = resolve_port_path(port)
    bus_type = port.bus_type or "feetech"
    prober = get_prober(bus_type)
    return prober.read_positions(path, list(port.motor_ids))


def resolve_port_path(port: SerialInterface) -> str:
    """Pick the best device path from a scanned port entry."""
    return port.dev or port.by_id or port.by_path


def resolve_port_by_id(port: SerialInterface) -> str:
    """Pick a stable identifier for set_arm (prefer by_id)."""
    return port.by_id or port.dev or port.by_path
