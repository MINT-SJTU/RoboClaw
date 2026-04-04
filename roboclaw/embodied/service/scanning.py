"""Scanning sub-service: port/camera scanning and motion detection.

Manages the embodiment lock internally so callers (routes, CLI) don't
have to coordinate acquire/release themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roboclaw.embodied.hardware.discovery import HardwareDiscovery

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


class ScanningService:
    """Port/camera scanning and motion detection with embodiment locking."""

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent
        self._scanner = HardwareDiscovery()

    @property
    def motion_active(self) -> bool:
        return self._scanner.motion_active

    # -- Locking scan operations ----------------------------------------------

    def run_full_scan(self, model: str = "") -> dict:
        """Scan ports + cameras with embodiment lock.

        Returns Interface objects (SerialInterface / VideoInterface).
        """
        self._parent.acquire_embodiment("scanning")
        try:
            if model:
                ports = self._scanner.discover(model)
            else:
                ports = self._scanner.discover_all()
            cameras = self._scanner.discover_cameras()
            return {"ports": ports, "cameras": cameras}
        finally:
            self._parent.release_embodiment()

    def capture_previews(self, output_dir: str) -> list[dict]:
        """Capture camera previews with embodiment lock."""
        self._parent.acquire_embodiment("camera-preview")
        try:
            return self._scanner.capture_camera_previews(output_dir)
        finally:
            self._parent.release_embodiment()

    def start_motion_detection(self) -> int:
        """Start motion detection — acquires embodiment until stop."""
        self._parent.acquire_embodiment("motion-detection")
        try:
            return self._scanner.start_motion_detection()
        except Exception:
            self._parent.release_embodiment()
            raise

    def stop_motion_detection(self) -> None:
        """Stop motion detection and release the embodiment lock."""
        self._scanner.stop_motion_detection()
        self._parent.release_embodiment(owner="motion-detection")

    def poll_motion(self) -> list[dict]:
        return self._scanner.poll_motion()
