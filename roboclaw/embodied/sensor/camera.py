"""Camera helpers for embodied actions."""

from __future__ import annotations

from typing import Any

from roboclaw.embodied.sensor.registry import OPENCV_CAMERA


def resolve_cameras(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cameras = manifest.get("cameras", [])
    result: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        alias = cam.get("alias", "")
        port = cam.get("port", "")
        if not alias or not port:
            continue
        config: dict[str, Any] = {
            "type": OPENCV_CAMERA.name,
            "index_or_path": port,
            "width": cam.get("width", OPENCV_CAMERA.default_width),
            "height": cam.get("height", OPENCV_CAMERA.default_height),
            "fps": cam.get("fps") or OPENCV_CAMERA.default_fps,
        }
        fourcc = cam.get("fourcc")
        if fourcc is not None:
            config["fourcc"] = fourcc
        result[alias] = config
    return result
