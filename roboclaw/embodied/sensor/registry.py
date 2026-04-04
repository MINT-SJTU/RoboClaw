"""Sensor spec registry — camera types and other sensors."""

from __future__ import annotations

from roboclaw.embodied.sensor.base import CameraSpec


# ---------------------------------------------------------------------------
# OpenCV camera (LeRobot default)
# ---------------------------------------------------------------------------

OPENCV_CAMERA = CameraSpec(name="opencv")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, CameraSpec] = {
    "opencv": OPENCV_CAMERA,
}


def get_camera_spec(name: str) -> CameraSpec:
    """Look up camera spec by type name."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown camera type: '{name}'")
    return spec


def all_camera_types() -> tuple[str, ...]:
    """Return all registered camera type names."""
    return tuple(_REGISTRY.keys())
