"""SetupSession — stateful setup workflow, sub-service of EmbodiedService.

Drives the discover → identify → assign → commit workflow.
Owns embodiment locking for scan/motion operations.
Shared by CLI agent and Web UI.
"""
from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.engine.helpers import _NO_TTY_MSG, _camera_previews_dir, _format_tty_failure, _run_tty
from roboclaw.embodied.hardware.discovery import HardwareDiscovery
from roboclaw.embodied.hardware.scan import restore_stderr, suppress_stderr
from roboclaw.embodied.interface import Interface, SerialInterface, VideoInterface

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class SetupPhase(str, Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    ASSIGNING = "assigning"
    IDENTIFYING = "identifying"
    COMMITTED = "committed"


@dataclass
class Assignment:
    """A pending assignment — interface matched to alias + spec, not yet committed."""

    alias: str
    spec_name: str  # e.g. "so101_follower", "inspire_rh56", "opencv"
    interface: Interface
    slave_id: int = 0  # hand-specific, set during assign if needed


class SetupSession:
    """Drives the discover → identify → assign → commit workflow.

    This is a direct sub-service of EmbodiedService, replacing ScanningService.
    It manages embodiment locking internally.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent
        self._phase = SetupPhase.IDLE
        self._model: str = ""
        self._discovery: HardwareDiscovery | None = None
        self._candidates: list[SerialInterface | VideoInterface] = []
        self._assignments: list[Assignment] = []

    @property
    def phase(self) -> SetupPhase:
        return self._phase

    @property
    def motion_active(self) -> bool:
        return self._phase == SetupPhase.IDENTIFYING

    @property
    def candidates(self) -> list[SerialInterface | VideoInterface]:
        return list(self._candidates)

    @property
    def assignments(self) -> list[Assignment]:
        return list(self._assignments)

    @property
    def unassigned(self) -> list[SerialInterface | VideoInterface]:
        assigned_ids = {a.interface.stable_id for a in self._assignments}
        return [c for c in self._candidates if c.stable_id not in assigned_ids]

    # -- Scan ----------------------------------------------------------------

    def run_full_scan(self, model: str = "") -> dict[str, Any]:
        """Scan ports + cameras. Resets session state.

        Acquires/releases embodiment lock during scan.
        Returns {ports: list[SerialInterface], cameras: list[VideoInterface]}.
        """
        self._parent.acquire_embodiment("scanning")
        try:
            self._reset()
            self._model = model
            self._phase = SetupPhase.DISCOVERING
            self._discovery = HardwareDiscovery()
            if model:
                serial = self._discovery.discover(model)
            else:
                serial = self._discovery.discover_all()
            video = self._discovery.discover_cameras()
            self._candidates = [*serial, *video]
            self._phase = SetupPhase.ASSIGNING
            ports = [c for c in self._candidates if isinstance(c, SerialInterface)]
            cameras = [c for c in self._candidates if isinstance(c, VideoInterface)]
            return {"ports": ports, "cameras": cameras}
        except Exception:
            self._reset()
            raise
        finally:
            self._parent.release_embodiment()

    def capture_previews(self, output_dir: str) -> list[dict]:
        """Capture camera preview frames."""
        self._parent.acquire_embodiment("camera-preview")
        try:
            discovery = self._discovery or HardwareDiscovery()
            if not discovery.scanned_cameras:
                discovery.discover_cameras()
            return discovery.capture_camera_previews(output_dir)
        finally:
            self._parent.release_embodiment()

    async def identify(self, manifest: Any, kwargs: dict[str, Any], tty_handoff: Any) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.hardware import scan as scan_module

        if not tty_handoff:
            return _NO_TTY_MSG
        ports = scan_module.scan_serial_ports()
        if not ports:
            return "No serial ports detected."
        ports_json = json.dumps([port.to_dict() for port in ports])
        argv = [sys.executable, "-m", "roboclaw.embodied.identify", ports_json]
        rc, stderr_text = await _run_tty(
            tty_handoff,
            LocalLeRobotRunner(),
            argv,
            "identify-arms",
        )
        if rc == 0:
            result = "Arm identification complete."
        else:
            result = _format_tty_failure("Arm identification failed", rc, stderr_text)
        self._parent.manifest.reload()
        return result

    def preview_cameras(self) -> str | list[dict[str, Any]]:
        from roboclaw.embodied.hardware.scan import capture_camera_frames, scan_cameras

        scanned_cameras = scan_cameras()
        if not scanned_cameras:
            return "No cameras detected."
        previews = capture_camera_frames(scanned_cameras, _camera_previews_dir())
        if not previews:
            return "No camera previews captured."
        return self._previews_to_multimodal(previews, scanned_cameras)

    # -- Identify (motion detection) -----------------------------------------

    def start_motion_detection(self) -> int:
        """Start motion detection. Acquires embodiment lock until stop."""
        if self._phase == SetupPhase.IDLE:
            raise RuntimeError("No scan performed. Run scan first.")
        self._parent.acquire_embodiment("motion-detection")
        try:
            return self._start_identify()
        except Exception:
            self._parent.release_embodiment()
            raise

    def _start_identify(self) -> int:
        serial = [c for c in self.unassigned if isinstance(c, SerialInterface)]
        if not serial:
            raise RuntimeError("No serial interfaces to identify.")
        saved = suppress_stderr()
        try:
            for iface in serial:
                iface.motion_detector.capture_baseline()
        finally:
            restore_stderr(saved)
        self._phase = SetupPhase.IDENTIFYING
        return len(serial)

    def stop_motion_detection(self) -> None:
        """Stop motion detection and release embodiment lock."""
        if self._phase == SetupPhase.IDENTIFYING:
            for c in self._candidates:
                if isinstance(c, SerialInterface):
                    c.motion_detector.reset()
            self._phase = SetupPhase.ASSIGNING
        self._parent.release_embodiment(owner="motion-detection")

    def poll_motion(self) -> list[dict[str, Any]]:
        """Poll motion on unassigned serial candidates."""
        if self._phase != SetupPhase.IDENTIFYING:
            raise RuntimeError("Motion detection not started.")
        serial = [c for c in self.unassigned if isinstance(c, SerialInterface)]
        results: list[dict[str, Any]] = []
        saved = suppress_stderr()
        try:
            for iface in serial:
                result = iface.motion_detector.poll()
                results.append({
                    "stable_id": iface.stable_id,
                    "dev": iface.dev,
                    "by_id": iface.by_id,
                    "motor_ids": list(iface.motor_ids),
                    "delta": result.delta,
                    "moved": result.moved,
                })
        finally:
            restore_stderr(saved)
        return results

    # -- Assign / Commit -----------------------------------------------------

    def assign(
        self, interface_stable_id: str, alias: str, spec_name: str,
    ) -> Assignment:
        """Assign a discovered interface to an alias and spec."""
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot assign in {self._phase} phase.")
        interface = None
        for c in self.unassigned:
            if c.stable_id == interface_stable_id:
                interface = c
                break
        if interface is None:
            raise ValueError(
                f"Interface {interface_stable_id} not found or already assigned."
            )
        if any(a.alias == alias for a in self._assignments):
            raise ValueError(f"Alias '{alias}' already assigned.")
        assignment = Assignment(alias=alias, spec_name=spec_name, interface=interface)
        self._assignments.append(assignment)
        return assignment

    def unassign(self, alias: str) -> None:
        """Remove an assignment, returning the interface to unassigned."""
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot unassign in {self._phase} phase.")
        for i, a in enumerate(self._assignments):
            if a.alias == alias:
                self._assignments.pop(i)
                return
        raise ValueError(f"No assignment with alias '{alias}'.")

    def commit(self) -> int:
        """Write all assignments to manifest.

        Transitions: ASSIGNING/IDENTIFYING → COMMITTED.
        Returns number of bindings created.
        """
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot commit in {self._phase} phase.")
        if not self._assignments:
            raise RuntimeError("No assignments to commit.")
        if self._phase == SetupPhase.IDENTIFYING:
            self.stop_motion_detection()
        manifest = self._parent.manifest
        for a in self._assignments:
            self._commit_one(manifest, a)
        count = len(self._assignments)
        self._phase = SetupPhase.COMMITTED
        return count

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state for API responses."""
        return {
            "phase": self._phase.value,
            "model": self._model,
            "candidates": [
                {"stable_id": c.stable_id, "interface_type": c.interface_type, **c.to_dict()}
                for c in self._candidates
            ],
            "assignments": [
                {
                    "alias": a.alias,
                    "spec_name": a.spec_name,
                    "interface_stable_id": a.interface.stable_id,
                }
                for a in self._assignments
            ],
            "unassigned": [c.stable_id for c in self.unassigned],
        }

    # -- Internal ------------------------------------------------------------

    def _reset(self) -> None:
        self._phase = SetupPhase.IDLE
        self._model = ""
        self._discovery = None
        self._candidates = []
        self._assignments = []

    @staticmethod
    def _previews_to_multimodal(
        previews: list[dict[str, str]],
        scanned: list[VideoInterface],
    ) -> list[dict[str, Any]]:
        cam_by_source: dict[str, VideoInterface] = {}
        for cam in scanned:
            for value in (cam.by_path, cam.by_id, cam.dev):
                if value:
                    cam_by_source[value] = cam

        blocks: list[dict[str, Any]] = []
        summary_lines = [
            f"Detected {len(scanned)} camera(s). Preview images below — "
            "suggest a descriptive name for each based on what you see "
            "(e.g. top, left_wrist, right_wrist, front, side)."
        ]
        empty_cam = VideoInterface()
        for index, preview in enumerate(previews):
            cam_info = cam_by_source.get(preview.get("camera", ""), empty_cam)
            summary_lines.append(
                f"\nCamera {index}: dev={cam_info.dev or '?'} "
                f"({cam_info.width}x{cam_info.height} @ {cam_info.fps}fps)"
            )
            img_path = Path(preview.get("image_path", ""))
            if not img_path.is_file():
                continue
            raw = img_path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            blocks.append({"type": "text", "text": f"Camera {index}:"})
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        blocks.insert(0, {"type": "text", "text": "\n".join(summary_lines)})
        return blocks

    @staticmethod
    def _commit_one(manifest: Any, assignment: Assignment) -> None:
        from roboclaw.embodied.embodiment.arm.registry import all_arm_types
        from roboclaw.embodied.embodiment.hand.registry import all_hand_types

        if assignment.spec_name in all_arm_types():
            manifest.set_arm(assignment.alias, assignment.spec_name, assignment.interface)
        elif assignment.spec_name in all_hand_types():
            manifest.set_hand(
                assignment.alias, assignment.spec_name,
                assignment.interface, assignment.slave_id,
            )
        elif isinstance(assignment.interface, VideoInterface):
            manifest.set_camera(assignment.alias, assignment.interface)
        else:
            raise ValueError(f"Unknown spec type: {assignment.spec_name}")
