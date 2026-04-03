"""Camera helpers for embodied actions."""

from __future__ import annotations

from typing import Any


def resolve_cameras(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cameras = manifest.get("cameras", [])
    result: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        alias = cam.get("alias", "")
        port = cam.get("port", "")
        if not alias or not port:
            continue
        config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": port,
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps") or 30,
        }
        fourcc = cam.get("fourcc")
        if fourcc is not None:
            config["fourcc"] = fourcc
        result[alias] = config
    return result
