"""Embodied tool groups - bridges agent to the embodied robotics layer."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool

_SETUP_ACTIONS = ["identify", "modify", "preview_cameras"]
_DOCTOR_ACTIONS = ["check"]
_CALIBRATION_ACTIONS = ["calibrate"]
_TELEOP_ACTIONS = ["teleoperate"]
_RECORD_ACTIONS = ["record"]
_REPLAY_ACTIONS = ["replay"]
_TRAIN_ACTIONS = ["train", "job_status", "list_datasets", "list_policies"]
_INFER_ACTIONS = ["run_policy"]
_HUB_ACTIONS = ["push_dataset", "pull_dataset", "push_policy", "pull_policy"]
_PERCEPTION_ACTIONS = ["scene_understand", "object_detect", "what_changed"]
_LANGUAGE_PROP = {"type": "string", "description": "User's language code (en, zh)."}

_TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "setup": {
        "description": "Hardware discovery, identification, and configuration management.",
        "actions": _SETUP_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _SETUP_ACTIONS,
                    "description": "The setup action to perform.",
                },
                "language": _LANGUAGE_PROP,
                "model": {
                    "type": "string",
                    "description": "Embodiment model name (e.g. so101, koch, inspire_rh56). Optional — the interactive flow handles selection.",
                },
                "target": {
                    "type": "string",
                    "enum": ["arm", "camera", "hand"],
                    "description": "Target type for modify.",
                },
                "operation": {
                    "type": "string",
                    "enum": ["rename", "unbind", "bind"],
                    "description": "Operation to perform for modify.",
                },
                "alias": {
                    "type": "string",
                    "description": "Existing alias to modify.",
                },
                "new_alias": {
                    "type": "string",
                    "description": "New alias for rename operation.",
                },
                "arm_type": {
                    "type": "string",
                    "description": "Arm type for bind_arm (e.g., 'koch_leader').",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path for bind_arm (by-id path from scan).",
                },
                "dev": {
                    "type": "string",
                    "description": "Camera device path for bind_camera (e.g., '/dev/video4').",
                },
                "side": {
                    "type": "string",
                    "enum": ["left", "right", ""],
                    "description": "Which arm the camera is mounted on (left/right for bimanual; omit for single-arm).",
                },
            },
            "required": ["action"],
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
                "language": _LANGUAGE_PROP,
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
    "hub": {
        "description": "Upload/download datasets and policies to/from HuggingFace Hub.",
        "actions": _HUB_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _HUB_ACTIONS,
                    "description": "The hub action to perform.",
                },
                "language": _LANGUAGE_PROP,
                "repo_id": {
                    "type": "string",
                    "description": "HuggingFace repo ID (e.g. username/dataset-name).",
                },
                "name": {
                    "type": "string",
                    "description": "Local dataset or policy name.",
                },
                "token": {
                    "type": "string",
                    "description": "HuggingFace token (optional, falls back to HF_TOKEN env or cached login).",
                },
                "private": {
                    "type": "boolean",
                    "description": "Whether to create a private repo on push.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "perception": {
        "description": "Understand what the robot's cameras see using VLM \u2014 scene description, object detection, and change detection.",
        "actions": _PERCEPTION_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _PERCEPTION_ACTIONS,
                    "description": "The perception action to perform.",
                },
                "language": _LANGUAGE_PROP,
                "camera_alias": {
                    "type": "string",
                    "description": "Camera alias (e.g. 'front', 'wrist').",
                },
                "question": {
                    "type": "string",
                    "description": "Free-text question for scene_understand.",
                },
                "object_name": {
                    "type": "string",
                    "description": "Target object name for object_detect.",
                },
                "model": {
                    "type": "string",
                    "description": "Override VLM model (e.g. 'anthropic/claude-sonnet-4-5'). Optional.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
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
        if self._group_name == "hub":
            return await self._execute_hub(kwargs)
        if self._group_name == "perception":
            return await self._execute_perception(kwargs)
        return f"Unknown tool group: {self._group_name}"

    async def _execute_setup(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        if action == "identify":
            return await _run_with_service(
                svc,
                lambda _: svc.setup.run_identify(kwargs, self._tty_handoff),
            )
        if action == "preview_cameras":
            return await _run_with_service(svc, lambda _: svc.setup.preview_cameras())
        # modify
        return await _run_with_service(
            svc,
            lambda _: _run_modify(svc, kwargs),
        )

    async def _execute_doctor(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda manifest: svc.doctor.check(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_calibration(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda manifest: svc.calibration.calibrate(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_teleop(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda manifest: svc.teleop.teleoperate(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_record(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda manifest: svc.record.record(manifest, kwargs, self._tty_handoff),
        )

    async def _execute_replay(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda _: svc.run_replay(
                dataset_name=kwargs.get("dataset_name", "default"),
                episode=kwargs.get("episode", 0),
                fps=kwargs.get("fps", 30),
                arms=kwargs.get("arms", ""),
                tty_handoff=self._tty_handoff,
            ),
        )

    async def _execute_train(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        return await _run_with_service(
            svc,
            lambda manifest: _run_train_action(svc.train, action, manifest, kwargs, self._tty_handoff),
        )

    async def _execute_infer(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        return await _run_with_service(
            svc,
            lambda _: svc.run_inference(
                checkpoint_path=kwargs.get("checkpoint_path", ""),
                source_dataset=kwargs.get("source_dataset", ""),
                dataset_name=kwargs.get("dataset_name", ""),
                task=kwargs.get("task", "eval"),
                num_episodes=kwargs.get("num_episodes", 1),
                episode_time_s=kwargs.get("episode_time_s", 60),
                arms=kwargs.get("arms", ""),
                use_cameras=kwargs.get("use_cameras", True),
                tty_handoff=self._tty_handoff,
            ),
        )

    async def _execute_hub(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        return await _run_with_service(
            svc,
            lambda manifest: _run_hub_action(svc.hub, action, manifest, kwargs, self._tty_handoff),
        )


    async def _execute_perception(self, kwargs: dict[str, Any]) -> str | list:
        action = kwargs.get("action", "")
        from roboclaw.embodied.perception import VLM, grab_all_frames, grab_frame

        try:
            vlm = VLM()
        except Exception as exc:
            return f"Perception module initialization failed: {exc}"

        if action == "scene_understand":
            camera_alias = kwargs.get("camera_alias", "")
            question = kwargs.get("question", "Describe what you see in this image.")
            model = kwargs.get("model") or None
            if camera_alias:
                result = await vlm.describe(camera_alias, question=question, model=model)
                return f"[{camera_alias}]: {result.text}"
            # No alias -> describe all
            results = await vlm.describe_all(question=question, model=model)
            if not results:
                return "No cameras configured."
            lines = [f"[{alias}]: {desc.text}" for alias, desc in results.items()]
            return "\n".join(lines)

        if action == "object_detect":
            camera_alias = kwargs.get("camera_alias", "")
            object_name = kwargs.get("object_name", "")
            model = kwargs.get("model") or None
            if not camera_alias or not object_name:
                return "camera_alias and object_name are required for object_detect."
            result = await vlm.detect_objects(camera_alias, object_name, model=model)
            return f"[{camera_alias}] {object_name}: {result.text}"

        if action == "what_changed":
            camera_alias = kwargs.get("camera_alias", "")
            if not camera_alias:
                return "camera_alias is required for what_changed."
            import os, tempfile
            import numpy as np
            tmp_before = os.path.join(tempfile.gettempdir(), f"roboclaw_before_{camera_alias}.npy")
            after_frame = grab_frame(camera_alias)
            if after_frame is None:
                return f"Camera '{camera_alias}' not found."
            if os.path.exists(tmp_before):
                before_frame = np.load(tmp_before)
                np.save(tmp_before, after_frame)
                result = await vlm.what_changed(
                    camera_alias,
                    before_frame,
                    after_frame,
                    model=kwargs.get("model"),
                )
                return f"Changes on [{camera_alias}]: {result.text}"
            else:
                np.save(tmp_before, after_frame)
                return "No previous frame saved. Captured current frame as 'before'. Call again after the action to see what changed."

        return f"Unknown perception action: {action}"


def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all tool groups."""
    return [EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff) for name, spec in _TOOL_GROUPS.items()]


def _find_camera(dev: str) -> tuple[Any, str]:
    """Scan and find a camera by device path. Returns (camera_or_None, available_list_str)."""
    from roboclaw.embodied.embodiment.hardware.scan import scan_cameras
    cameras = scan_cameras()
    matched = next((c for c in cameras if c.dev == dev), None)
    avail = ", ".join(c.dev for c in cameras) if cameras else "none"
    return matched, avail


def _find_serial_port(port: str) -> tuple[Any, str]:
    """Scan and find a serial port by by-id or dev path. Returns (port_or_None, available_list_str)."""
    from roboclaw.embodied.embodiment.hardware.scan import scan_serial_ports
    ports = scan_serial_ports()
    matched = next((p for p in ports if p.by_id == port or p.dev == port), None)
    avail = ", ".join(p.by_id or p.dev for p in ports) if ports else "none"
    return matched, avail


_MODIFY_DISPATCH = {
    ("unbind", "arm"): "unbind_arm",
    ("unbind", "camera"): "unbind_camera",
    ("unbind", "hand"): "unbind_hand",
    ("rename", "arm"): "rename_arm",
    ("rename", "camera"): "rename_camera",
    ("bind", "camera"): "bind_camera",
    ("bind", "arm"): "bind_arm",

    ("rename", "hand"): "rename_hand",
}


async def _run_modify(svc: Any, kwargs: dict[str, Any]) -> str:
    target = kwargs.get("target", "")
    operation = kwargs.get("operation", "")
    alias = kwargs.get("alias", "")
    if not all([target, operation, alias]):
        return "modify requires target, operation, and alias."

    method_name = _MODIFY_DISPATCH.get((operation, target))
    if method_name is None:
        return f"Unknown operation/target: {operation}/{target}"

    if operation == "unbind":
        getattr(svc, method_name)(alias)
        return f"{target.title()} '{alias}' removed."

    if operation == "bind" and target == "camera":
        dev = kwargs.get("dev", "")
        side = kwargs.get("side", "")
        if not dev:
            return "bind camera requires dev (e.g., '/dev/video4')."
        from roboclaw.embodied.embodiment.manifest.binding import validate_camera_side
        try:
            validate_camera_side(side)
        except ValueError as exc:
            return str(exc)
        matched, avail = _find_camera(dev)
        if matched is None:
            return f"Camera '{dev}' not found. Available: {avail}"
        result = svc.bind_camera(alias, matched, side)
        data = result.to_dict() if hasattr(result, "to_dict") else str(result)
        return f"Camera '{alias}' bound to {dev}.\n{json.dumps(data, indent=2) if isinstance(data, dict) else data}"

    if operation == "bind" and target == "arm":
        arm_type = kwargs.get("arm_type", "")
        port = kwargs.get("port", "")
        if not arm_type:
            from roboclaw.embodied.embodiment.arm.registry import all_arm_types
            return f"bind arm requires arm_type. Valid: {list(all_arm_types())}"
        if not port:
            return "bind arm requires port (by-id path from scan results)."
        matched, avail = _find_serial_port(port)
        if matched is None:
            return f"Port '{port}' not found. Available: {avail}"
        result = svc.bind_arm(alias, arm_type, matched)
        data = result.to_dict() if hasattr(result, "to_dict") else str(result)
        return f"Arm '{alias}' ({arm_type}) bound to {port}.\n{json.dumps(data, indent=2) if isinstance(data, dict) else data}"

    # rename
    new_alias = kwargs.get("new_alias", "")
    if not new_alias:
        return "rename requires new_alias."
    result = getattr(svc, method_name)(alias, new_alias)
    data = result.to_dict() if hasattr(result, "to_dict") else result
    return f"{target.title()} renamed '{alias}' → '{new_alias}'.\n{json.dumps(data, indent=2)}"


def _get_service(service: Any) -> Any:
    if service is not None:
        return service
    from roboclaw.embodied.service import EmbodiedService

    return EmbodiedService()


async def _run_with_service(service: Any, func: Any) -> str | list:
    from roboclaw.embodied.command import ActionError

    try:
        return await func(service.manifest)
    except ActionError as exc:
        return str(exc)


async def _run_hub_action(
    hub: Any,
    action: str,
    manifest: Any,
    kwargs: dict[str, Any],
    tty_handoff: Any,
) -> str:
    method = getattr(hub, action, None)
    if method is None:
        return f"Unknown hub action: {action}"
    return await method(manifest, kwargs, tty_handoff)


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
