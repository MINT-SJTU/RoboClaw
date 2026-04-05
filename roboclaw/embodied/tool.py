"""Embodied tool groups - bridges agent to the embodied robotics layer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.embodied.embodiment.arm.registry import all_arm_types
from roboclaw.embodied.embodiment.hand.registry import all_hand_types
from roboclaw.embodied.manifest.helpers import (
    _probe_hand_slave_id,
    _resolve_serial_interface,
    arm_display_name,
)

_MANIFEST_ACTIONS = [
    "status",
    "bind_arm",
    "unbind_arm",
    "rename_arm",
    "bind_camera",
    "unbind_camera",
    "rename_camera",
    "bind_hand",
    "unbind_hand",
    "rename_hand",
    "describe",
]
_SETUP_ACTIONS = ["scan", "identify", "preview_cameras"]
_DOCTOR_ACTIONS = ["check"]
_CALIBRATION_ACTIONS = ["calibrate"]
_TELEOP_ACTIONS = ["teleoperate"]
_RECORD_ACTIONS = ["record"]
_REPLAY_ACTIONS = ["replay"]
_TRAIN_ACTIONS = ["train", "job_status", "list_datasets", "list_policies"]
_INFER_ACTIONS = ["run_policy"]
_EMBODIMENT_CONTROL_ACTIONS = ["hand_open", "hand_close", "hand_pose", "hand_status"]

_TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "manifest": {
        "description": "Inspect and edit the embodied hardware manifest.",
        "actions": _MANIFEST_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _MANIFEST_ACTIONS,
                    "description": "The manifest action to perform.",
                },
                "alias": {
                    "type": "string",
                    "description": "Existing arm or hand alias to update or remove.",
                },
                "arm_type": {
                    "type": "string",
                    "enum": list(all_arm_types()),
                    "description": "Arm hardware type for bind_arm.",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path for bind_arm or bind_hand.",
                },
                "new_alias": {
                    "type": "string",
                    "description": "New alias for rename_arm, rename_camera, or rename_hand.",
                },
                "camera_name": {
                    "type": "string",
                    "description": "Camera alias for bind_camera, unbind_camera, or rename_camera.",
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Index into the detected camera list for bind_camera.",
                },
                "target_action": {
                    "type": "string",
                    "description": "Action name to describe.",
                },
                "hand_type": {
                    "type": "string",
                    "enum": list(all_hand_types()),
                    "description": "Hand hardware type for bind_hand.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "setup": {
        "description": "Scan hardware, identify arms, and preview cameras.",
        "actions": _SETUP_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _SETUP_ACTIONS,
                    "description": "The setup action to perform.",
                },
                "model": {
                    "type": "string",
                    "description": "Robot model name to narrow scan protocol.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "doctor": {
        "description": "Check embodied environment health and summarize the current setup.",
        "actions": _DOCTOR_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _DOCTOR_ACTIONS,
                    "description": "The doctor action to perform.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "calibration": {
        "description": "Calibrate one or more configured arms.",
        "actions": _CALIBRATION_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _CALIBRATION_ACTIONS,
                    "description": "The calibration action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from status).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "teleop": {
        "description": "Run live teleoperation.",
        "actions": _TELEOP_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _TELEOP_ACTIONS,
                    "description": "The teleop action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from status).",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for teleoperation.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "record": {
        "description": "Record a dataset from one or two leader/follower pairs.",
        "actions": _RECORD_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _RECORD_ACTIONS,
                    "description": "The recording action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from status).",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for record.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for recording.",
                },
                "use_cameras": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether recording should include configured cameras.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for recording.",
                },
                "episode_time_s": {
                    "type": "integer",
                    "description": "Duration per episode in seconds.",
                },
                "reset_time_s": {
                    "type": "integer",
                    "description": "Duration of reset period between episodes in seconds.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "replay": {
        "description": "Replay a recorded dataset episode on follower arms.",
        "actions": _REPLAY_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _REPLAY_ACTIONS,
                    "description": "The replay action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from status).",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for replay.",
                },
                "episode": {
                    "type": "integer",
                    "description": "Episode index to replay.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for replay.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "train": {
        "description": "Train ACT, inspect training jobs, and list datasets or policies.",
        "actions": _TRAIN_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _TRAIN_ACTIONS,
                    "description": "The training action to perform.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for training.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of training steps.",
                },
                "device": {
                    "type": "string",
                    "description": "Device for training.",
                },
                "job_id": {
                    "type": "string",
                    "description": "ID of a background training job.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "infer": {
        "description": "Run a trained policy on the configured follower arms.",
        "actions": _INFER_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _INFER_ACTIONS,
                    "description": "The inference action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from status).",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Optional evaluation dataset slug.",
                },
                "source_dataset": {
                    "type": "string",
                    "description": "Dataset name to resolve the default checkpoint from.",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Path to a trained policy checkpoint.",
                },
                "task": {
                    "type": "string",
                    "description": "Task label for policy rollout.",
                },
                "use_cameras": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether inference should include configured cameras.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of rollout episodes to run.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodiment_control": {
        "description": "Control a dexterous hand: open, close, pose, and query status.",
        "actions": _EMBODIMENT_CONTROL_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _EMBODIMENT_CONTROL_ACTIONS,
                    "description": "The embodiment-control action to perform.",
                },
                "hand_name": {
                    "type": "string",
                    "description": "Hand alias. Uses first configured hand if omitted.",
                },
                "positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "6 finger positions 0-1000 for hand_pose.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

_ACTION_DESCRIPTIONS = {
    "scan": "Scan for serial ports with motors and available cameras.",
    "check": "Check LeRobot availability and show the current embodied setup.",
    "identify": "Launch the interactive arm-identification flow for detected serial ports.",
    "describe": "Explain adjustable parameters for a target embodied action.",
    "calibrate": "Calibrate one or more configured arms. If arms is omitted, calibrate every uncalibrated arm.",
    "teleoperate": "Run live teleoperation. Select arms with a comma-separated port list.",
    "record": "Record a dataset with one follower/leader pair or two pairs for bimanual capture.",
    "replay": "Replay a recorded dataset episode on one or two follower arms.",
    "train": "Start ACT training for a recorded dataset as a detached job.",
    "run_policy": "Run a trained policy rollout with one or two follower arms.",
    "job_status": "Inspect the status and recent logs for a detached training job.",
    "status": "Show hardware status: configured arms/cameras, connectivity, calibration, readiness.",
    "bind_arm": "Create or update one configured arm alias.",
    "rename_arm": "Rename an existing configured arm alias.",
    "unbind_arm": "Remove one configured arm alias.",
    "bind_camera": "Assign a scanned camera to a stable camera name.",
    "rename_camera": "Rename an existing configured camera alias.",
    "preview_cameras": "Capture one preview image for each scanned camera.",
    "unbind_camera": "Remove a configured camera.",
    "bind_hand": "Create or update one configured hand alias.",
    "unbind_hand": "Remove a configured hand alias.",
    "rename_hand": "Rename an existing configured hand alias.",
    "hand_open": "Open all fingers of a dexterous hand.",
    "hand_close": "Close all fingers of a dexterous hand.",
    "hand_pose": "Set individual finger positions on a dexterous hand.",
    "hand_status": "Read current finger angles and forces from a dexterous hand.",
    "list_datasets": "List recorded datasets with episode counts.",
    "list_policies": "List trained policy checkpoints.",
}


class EmbodiedToolGroup(Tool):
    """A single embodied tool group with group-local dispatch."""

    def __init__(self, group_name: str, spec: dict[str, Any], tty_handoff: Any = None):
        self._group_name = group_name
        self._spec = spec
        self._tty_handoff = tty_handoff
        self.embodied_service = None

    @property
    def name(self) -> str:
        return self._group_name

    @property
    def description(self) -> str:
        return self._spec["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self._spec["parameters"]

    async def execute(self, **kwargs: Any) -> str | list:
        action = kwargs.get("action", "")
        if action not in self._spec["actions"]:
            return f"Unknown action '{action}' for tool {self._group_name}."
        if self._group_name == "manifest":
            return await self._execute_manifest(kwargs)
        if self._group_name == "setup":
            return await self._execute_setup(kwargs)
        if self._group_name == "doctor":
            return await self._execute_doctor(kwargs)
        if self._group_name == "calibration":
            return await self._execute_calibration(kwargs)
        if self._group_name == "teleop":
            return await self._execute_teleop(kwargs)
        if self._group_name == "record":
            return await self._execute_record(kwargs)
        if self._group_name == "replay":
            return await self._execute_replay(kwargs)
        if self._group_name == "train":
            return await self._execute_train(kwargs)
        if self._group_name == "infer":
            return await self._execute_infer(kwargs)
        if self._group_name == "embodiment_control":
            return await self._execute_embodiment_control(kwargs)
        return f"Unknown tool group: {self._group_name}"

    async def _execute_manifest(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        if action == "status":
            return svc.get_manifest_summary()
        if action == "describe":
            target_action = kwargs.get("target_action", "")
            if not target_action:
                return json.dumps(_ACTION_DESCRIPTIONS, indent=2, ensure_ascii=False)
            if target_action not in _ACTION_DESCRIPTIONS:
                return f"Unknown target_action: {target_action}"
            return f"{target_action}: {_ACTION_DESCRIPTIONS[target_action]}"
        if action == "bind_arm":
            alias = kwargs.get("alias", "")
            arm_type = kwargs.get("arm_type", "")
            port = kwargs.get("port", "")
            if not all([alias, arm_type, port]):
                return "bind_arm requires alias, arm_type, and port."
            interface = _resolve_serial_interface(port)
            binding = svc.bind_arm(alias, arm_type, interface)
            display = arm_display_name(binding)
            return f"Arm '{display}' configured.\n{json.dumps(binding.to_dict(), indent=2)}"
        if action == "unbind_arm":
            alias = kwargs.get("alias", "")
            if not alias:
                return "unbind_arm requires alias."
            svc.unbind_arm(alias)
            return f"Arm '{alias}' removed."
        if action == "rename_arm":
            alias = kwargs.get("alias", "")
            new_alias = kwargs.get("new_alias", "")
            if not alias or not new_alias:
                return "rename_arm requires alias and new_alias."
            binding = svc.rename_arm(alias, new_alias)
            return (
                f"Arm renamed from '{alias}' to '{new_alias}'.\n"
                f"{json.dumps(binding.to_dict(), indent=2)}"
            )
        if action == "bind_camera":
            camera_name = kwargs.get("camera_name", "")
            camera_index = kwargs.get("camera_index")
            if not camera_name or camera_index is None:
                return "bind_camera requires camera_name and camera_index."
            from roboclaw.embodied.hardware.scan import scan_cameras

            scanned = scan_cameras()
            if camera_index < 0 or camera_index >= len(scanned):
                return f"camera_index {camera_index} out of range. Found {len(scanned)} camera(s)."
            interface = scanned[camera_index]
            if not interface.address:
                return f"Scanned camera at index {camera_index} has no usable path."
            binding = svc.bind_camera(camera_name, interface)
            return f"Camera '{camera_name}' configured.\n{json.dumps(binding.to_dict(), indent=2)}"
        if action == "unbind_camera":
            camera_name = kwargs.get("camera_name", "")
            if not camera_name:
                return "unbind_camera requires camera_name."
            svc.unbind_camera(camera_name)
            return f"Camera '{camera_name}' removed."
        if action == "rename_camera":
            camera_name = kwargs.get("camera_name", "")
            new_alias = kwargs.get("new_alias", "")
            if not camera_name or not new_alias:
                return "rename_camera requires camera_name and new_alias."
            binding = svc.rename_camera(camera_name, new_alias)
            return (
                f"Camera renamed from '{camera_name}' to '{new_alias}'.\n"
                f"{json.dumps(binding.to_dict(), indent=2)}"
            )
        if action == "bind_hand":
            alias = kwargs.get("alias", "")
            hand_type = kwargs.get("hand_type", "")
            port = kwargs.get("port", "")
            if not all([alias, hand_type, port]):
                return "bind_hand requires alias, hand_type, and port."
            interface = _resolve_serial_interface(port)
            slave_id = _probe_hand_slave_id(hand_type, interface.address)
            binding = svc.bind_hand(alias, hand_type, interface, slave_id)
            return f"Hand '{alias}' configured.\n{json.dumps(binding.to_dict(), indent=2)}"
        if action == "unbind_hand":
            alias = kwargs.get("alias", "")
            if not alias:
                return "unbind_hand requires alias."
            svc.unbind_hand(alias)
            return f"Hand '{alias}' removed."
        # rename_hand (final action in manifest group)
        alias = kwargs.get("alias", "")
        new_alias = kwargs.get("new_alias", "")
        if not alias or not new_alias:
            return "rename_hand requires alias and new_alias."
        binding = svc.rename_hand(alias, new_alias)
        return (
            f"Hand renamed from '{alias}' to '{new_alias}'.\n"
            f"{json.dumps(binding.to_dict(), indent=2)}"
        )

    async def _execute_setup(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        if action == "scan":
            result = await asyncio.to_thread(svc.setup.run_full_scan, kwargs.get("model", ""))
            return _format_scan(result)
        if action == "preview_cameras":
            return svc.setup.preview_cameras()
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.setup.identify(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_doctor(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.doctor.check(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_calibration(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.calibration_session.calibrate(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_teleop(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.teleop.teleoperate(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_record(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.record.record(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_replay(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.replay.replay(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_train(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        return await _run_with_manifest(
            svc,
            lambda manifest: _run_train_action(svc.train, action, manifest, kwargs, self._tty_handoff),
        )

    async def _execute_infer(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_manifest(
            svc,
            lambda manifest: svc.infer.run_policy(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_embodiment_control(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        return await _run_with_manifest(
            svc,
            lambda manifest: _run_hand_action(svc.hand, action, manifest, kwargs, self._tty_handoff),
        )


def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all tool groups."""
    return [EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff) for name, spec in _TOOL_GROUPS.items()]


def _format_scan(result: dict[str, Any]) -> str:
    ports = result["ports"]
    cameras = result["cameras"]
    lines = [f"Found {len(ports)} serial port(s) and {len(cameras)} camera(s)."]
    if ports:
        lines.append("\nPorts:")
        for port in ports:
            port_id = port.by_id or port.dev or "?"
            lines.append(f"  - {port_id}  ({len(port.motor_ids)} motors)")
    if cameras:
        lines.append("\nCameras:")
        for camera in cameras:
            lines.append(f"  - {camera.dev or '?'}  ({camera.width}x{camera.height} @ {camera.fps}fps)")
    return "\n".join(lines)


def _get_service(service: Any) -> Any:
    if service is not None:
        return service
    from roboclaw.embodied.service import EmbodiedService

    return EmbodiedService()


async def _run_with_manifest(service: Any, func: Any) -> str | list:
    from roboclaw.embodied.engine.helpers import ActionError
    from roboclaw.embodied.manifest.helpers import ensure_manifest

    manifest = ensure_manifest()
    service.manifest = manifest
    try:
        return await func(manifest)
    except ActionError as exc:
        return str(exc)


async def _run_train_action(
    train: Any,
    action: str,
    manifest: Any,
    kwargs: dict[str, Any],
    tty_handoff: Any,
) -> str:
    if action == "train":
        return await train.train(manifest, kwargs, tty_handoff)
    if action == "job_status":
        return await train.job_status(manifest, kwargs, tty_handoff)
    if action == "list_datasets":
        return train.list_datasets(manifest)
    return train.list_policies(manifest)


async def _run_hand_action(
    hand: Any,
    action: str,
    manifest: Any,
    kwargs: dict[str, Any],
    tty_handoff: Any,
) -> str:
    if action == "hand_open":
        return await hand.open_hand(manifest, kwargs, tty_handoff)
    if action == "hand_close":
        return await hand.close_hand(manifest, kwargs, tty_handoff)
    if action == "hand_pose":
        return await hand.set_pose(manifest, kwargs, tty_handoff)
    return await hand.get_status(manifest, kwargs, tty_handoff)
