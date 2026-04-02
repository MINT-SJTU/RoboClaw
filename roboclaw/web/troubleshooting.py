"""Troubleshooting decision map and fault snapshot generator.

Provides user-facing troubleshooting steps for each hardware fault type,
plus a snapshot generator for tech support reports.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from roboclaw.embodied.hardware_monitor import HardwareFault


@dataclass
class TroubleshootEntry:
    can_recheck: bool
    step_count: int


TROUBLESHOOT_MAP: dict[str, TroubleshootEntry] = {
    "arm_disconnected": TroubleshootEntry(can_recheck=True, step_count=6),
    "arm_timeout": TroubleshootEntry(can_recheck=True, step_count=4),
    "arm_not_calibrated": TroubleshootEntry(can_recheck=False, step_count=1),
    "camera_disconnected": TroubleshootEntry(can_recheck=True, step_count=4),
    "camera_frame_drop": TroubleshootEntry(can_recheck=True, step_count=3),
    "record_crashed": TroubleshootEntry(can_recheck=False, step_count=2),
}


def get_troubleshoot_map_json() -> dict[str, dict[str, Any]]:
    """Return the troubleshooting map as a JSON-serializable dict."""
    return {key: asdict(entry) for key, entry in TROUBLESHOOT_MAP.items()}


def generate_fault_snapshot(
    setup: dict[str, Any],
    faults: list[HardwareFault],
    stderr_tail: str,
) -> dict[str, Any]:
    """Generate a fault snapshot for tech support.

    Includes: setup.json content, active faults, last 50 lines of stderr, timestamp.
    """
    return {
        "timestamp": time.time(),
        "setup": setup,
        "faults": [f.to_dict() for f in faults],
        "stderr_tail": stderr_tail,
    }
