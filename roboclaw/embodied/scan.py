"""Hardware scanning — detect serial ports and cameras."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path


def scan_serial_ports() -> list[dict[str, str]]:
    """Scan /dev/serial/by-id/ for connected serial devices."""
    by_id_dir = Path("/dev/serial/by-id")
    if not by_id_dir.exists():
        return []
    ports = []
    for entry in sorted(by_id_dir.iterdir()):
        if not entry.is_symlink():
            continue
        target = os.path.realpath(str(entry))
        if not os.path.exists(target):
            continue
        ports.append({"id": entry.name, "path": str(entry), "target": target})
    return ports


def scan_cameras() -> list[dict[str, str | int]]:
    """Scan cameras via /dev/v4l/by-path/, probe with OpenCV, enrich with by-id."""
    try:
        import cv2
    except ImportError:
        return []

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        by_path_map = _build_v4l_by_path_map()
        by_id_map = _build_v4l_by_id_map()
        return _probe_cameras(cv2, by_path_map, by_id_map)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _build_v4l_by_path_map() -> dict[str, str]:
    """Map /dev/videoN -> /dev/v4l/by-path/... path."""
    result = {}
    by_path_dir = Path("/dev/v4l/by-path")
    if not by_path_dir.exists():
        return result
    for entry in by_path_dir.iterdir():
        if not entry.is_symlink():
            continue
        target = os.path.realpath(str(entry))
        result[target] = str(entry)
    return result


def _build_v4l_by_id_map() -> dict[str, str]:
    """Map /dev/videoN -> /dev/v4l/by-id/... path."""
    result = {}
    by_id_dir = Path("/dev/v4l/by-id")
    if not by_id_dir.exists():
        return result
    for entry in by_id_dir.iterdir():
        if not entry.is_symlink():
            continue
        target = os.path.realpath(str(entry))
        result[target] = str(entry)
    return result


def _probe_cameras(cv2, by_path_map: dict, by_id_map: dict) -> list[dict[str, str | int]]:
    """Try opening each /dev/videoN, return those that work with by-path/by-id info."""
    cameras = []
    for dev in sorted(glob.glob("/dev/video*")):
        m = re.match(r"/dev/video(\d+)$", dev)
        if not m:
            continue
        info = _try_open_camera(cv2, int(m.group(1)), dev, by_path_map, by_id_map)
        if info:
            cameras.append(info)
    return cameras


def _try_open_camera(cv2, index: int, dev: str, by_path_map: dict, by_id_map: dict) -> dict[str, str | int] | None:
    """Open a single camera by index, return info dict or None."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_path = os.path.realpath(dev)
        return {
            "dev": dev,
            "by_path": by_path_map.get(real_path, ""),
            "by_id": by_id_map.get(real_path, ""),
            "width": w,
            "height": h,
        }
    finally:
        cap.release()
