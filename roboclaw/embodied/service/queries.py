"""Query sub-service: read-only status, listings, and previews."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.hardware_monitor import (
    ArmStatus,
    CameraStatus,
    check_arm_status,
    check_camera_status,
)
from roboclaw.embodied.ops.helpers import _camera_previews_dir, group_arms
from roboclaw.embodied.setup import load_setup

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService

_ACTION_DESCRIPTIONS = {
    "scan": "Scan for serial ports with motors and available cameras.",
    "doctor": "Check LeRobot availability and show the current embodied setup.",
    "identify": "Launch the interactive arm-identification flow for detected serial ports.",
    "describe": "Explain adjustable parameters for a target embodied action.",
    "calibrate": "Calibrate one or more configured arms. If arms is omitted, calibrate every uncalibrated arm.",
    "teleoperate": "Run live teleoperation. Select arms with a comma-separated port list.",
    "record": "Record a dataset with one follower/leader pair or two pairs for bimanual capture.",
    "replay": "Replay a recorded dataset episode on one or two follower arms.",
    "train": "Start ACT training for a recorded dataset as a detached job.",
    "run_policy": "Run a trained policy: use embodied_control(action='record', checkpoint_path='...') with one follower arm.",
    "job_status": "Inspect the status and recent logs for a detached training job.",
    "setup_show": "Show the embodied setup JSON with configured arms, cameras, and roots.",
    "set_arm": "Create or update one configured arm alias.",
    "rename_arm": "Rename an existing configured arm alias.",
    "remove_arm": "Remove one configured arm alias.",
    "set_camera": "Assign a scanned camera to a stable camera name.",
    "preview_cameras": "Capture one preview image for each scanned camera.",
    "remove_camera": "Remove a configured camera.",
    "set_hand": "Create or update one configured hand alias.",
    "remove_hand": "Remove a configured hand alias.",
    "hand_open": "Open all fingers of a dexterous hand.",
    "hand_close": "Close all fingers of a dexterous hand.",
    "hand_pose": "Set individual finger positions on a dexterous hand.",
    "hand_status": "Read current finger angles and forces from a dexterous hand.",
    "list_datasets": "List recorded datasets with episode counts.",
    "list_policies": "List trained policy checkpoints.",
}


def _compute_readiness(
    arms: list[dict[str, Any]],
    arm_statuses: list[ArmStatus],
    camera_statuses: list[CameraStatus],
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    grouped = group_arms(arms)
    if not grouped["followers"]:
        missing.append("No follower arm configured")
    if not grouped["leaders"]:
        missing.append("No leader arm configured")
    for s in arm_statuses:
        if not s.connected:
            missing.append(f"Arm '{s.alias}' is disconnected")
        elif not s.calibrated:
            missing.append(f"Arm '{s.alias}' is not calibrated")
    for s in camera_statuses:
        if not s.connected:
            missing.append(f"Camera '{s.alias}' is disconnected")
    f, l = grouped["followers"], grouped["leaders"]
    if f and l and len(f) != len(l):
        missing.append(f"Follower/leader count mismatch: {len(f)} vs {len(l)}")
    return len(missing) == 0, missing


def _previews_to_multimodal(
    previews: list[dict[str, str]], scanned: list[dict],
) -> list[dict]:
    """Convert camera previews to multimodal content blocks with embedded images."""
    cam_by_source: dict[str, dict] = {}
    for cam in scanned:
        for key in ("by_path", "by_id", "dev"):
            if val := cam.get(key):
                cam_by_source[val] = cam

    blocks: list[dict] = []
    summary_lines = [
        f"Detected {len(scanned)} camera(s). Preview images below — "
        "suggest a descriptive name for each based on what you see "
        "(e.g. top, left_wrist, right_wrist, front, side)."
    ]
    for i, preview in enumerate(previews):
        cam_info = cam_by_source.get(preview.get("camera", ""), {})
        summary_lines.append(
            f"\nCamera {i}: dev={cam_info.get('dev', '?')} "
            f"({cam_info.get('width', '?')}x{cam_info.get('height', '?')} "
            f"@ {cam_info.get('fps', '?')}fps)"
        )
        img_path = Path(preview.get("image_path", ""))
        if img_path.is_file():
            raw = img_path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            blocks.append({"type": "text", "text": f"Camera {i}:"})
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    blocks.insert(0, {"type": "text", "text": "\n".join(summary_lines)})
    return blocks


class QueryService:
    """Read-only queries: setup, hardware status, datasets, policies, previews."""

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    def get_current_config(self) -> dict[str, Any]:
        """Return current setup config (arms, cameras, hands)."""
        setup = load_setup()
        return {
            "arms": setup.get("arms", []),
            "cameras": setup.get("cameras", []),
            "hands": setup.get("hands", []),
        }

    def get_setup(self) -> str:
        """Return setup JSON enriched with hardware connectivity status.

        Reuses get_hardware_status() so CLI and web share the same
        connectivity / calibration checks.

        Note: scan_serial_ports() and scan_cameras() are called directly
        from roboclaw.embodied.scan (low-level I/O), not through
        ScanningService. This is intentional — these are read-only probes
        that don't drive motors, so they don't need the embodiment lock.
        The ScanningService locking is for operations that interact with
        motor control or hold resources open (e.g. motion detection).
        """
        from roboclaw.embodied.scan import scan_cameras, scan_serial_ports

        setup = load_setup()
        setup["scanned_ports"] = scan_serial_ports()
        setup["scanned_cameras"] = scan_cameras()

        hw = self.get_hardware_status(setup)
        setup["hardware_status"] = {
            "ready": hw["ready"],
            "missing": hw["missing"],
            "arms": hw["arms"],
            "cameras": hw["cameras"],
        }
        return json.dumps(setup, indent=2, ensure_ascii=False)

    def describe_actions(self, target_action: str = "") -> str:
        if not target_action:
            return json.dumps(_ACTION_DESCRIPTIONS, indent=2, ensure_ascii=False)
        if target_action not in _ACTION_DESCRIPTIONS:
            return f"Unknown target_action: {target_action}"
        return f"{target_action}: {_ACTION_DESCRIPTIONS[target_action]}"

    def list_datasets(self) -> str:
        from roboclaw.embodied.setup import ensure_setup

        setup = ensure_setup()
        root = Path(setup.get("datasets", {}).get("root", "")) / "local"
        if not root.exists():
            return "No datasets found."
        datasets = []
        for d in sorted(root.iterdir()):
            info_path = d / "meta" / "info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            datasets.append({
                "name": d.name,
                "episodes": info.get("total_episodes", 0),
                "frames": info.get("total_frames", 0),
                "fps": info.get("fps", 0),
            })
        if not datasets:
            return "No datasets found."
        return json.dumps(datasets, indent=2, ensure_ascii=False)

    def list_policies(self) -> str:
        from roboclaw.embodied.setup import ensure_setup

        setup = ensure_setup()
        root = Path(setup.get("policies", {}).get("root", ""))
        if not root.exists():
            return "No policies found."
        policies = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            last = d / "checkpoints" / "last" / "pretrained_model"
            if not last.exists():
                continue
            entry = {"name": d.name, "checkpoint": str(last)}
            tcfg = last / "train_config.json"
            if tcfg.exists():
                try:
                    cfg = json.loads(tcfg.read_text())
                except (json.JSONDecodeError, OSError):
                    cfg = {}
                entry["dataset"] = cfg.get("dataset", {}).get("repo_id", "")
                entry["steps"] = cfg.get("steps", 0)
            policies.append(entry)
        if not policies:
            return "No policies found."
        return json.dumps(policies, indent=2, ensure_ascii=False)

    def preview_cameras(self) -> str | list:
        from roboclaw.embodied.scan import capture_camera_frames, scan_cameras

        scanned_cameras = scan_cameras()
        if not scanned_cameras:
            return "No cameras detected."
        previews = capture_camera_frames(scanned_cameras, _camera_previews_dir())
        if not previews:
            return "No camera previews captured."
        return _previews_to_multimodal(previews, scanned_cameras)

    def get_hardware_status(self, setup: dict | None = None) -> dict[str, Any]:
        if setup is None:
            setup = load_setup()
        arms = setup.get("arms", [])
        cameras = setup.get("cameras", [])
        arm_statuses = [check_arm_status(a) for a in arms]
        camera_statuses = [check_camera_status(c) for c in cameras]
        ready, missing = _compute_readiness(arms, arm_statuses, camera_statuses)
        return {
            "ready": ready,
            "missing": missing,
            "arms": [s.to_dict() for s in arm_statuses],
            "cameras": [s.to_dict() for s in camera_statuses],
            "session_busy": self._parent.session.busy,
        }

    def read_servo_positions(self) -> dict[str, Any]:
        if self._parent.busy:
            return {"error": "busy", "arms": {}}
        from roboclaw.embodied.motors import read_servo_positions
        return read_servo_positions()
