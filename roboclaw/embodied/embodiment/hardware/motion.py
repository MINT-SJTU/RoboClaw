"""Motion detection helpers."""
from __future__ import annotations

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
