"""Embodied tool groups - bridges agent to the embodied robotics layer."""

from __future__ import annotations

import asyncio
from typing import Any

from roboclaw.agent.tools.base import Tool

_SETUP_ACTIONS = [
    "setup_show", "scan", "set_arm", "remove_arm", "rename_arm",
    "set_camera", "preview_cameras", "remove_camera", "describe", "doctor",
    "set_hand", "remove_hand",
]

_TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "embodied_setup": {
        "description": (
            "Configure robot hardware: show setup, scan for ports/cameras, "
            "add/remove/rename arms, add/remove cameras, describe actions, "
            "check environment. Also manages dexterous hands: set_hand, remove_hand."
        ),
        "actions": _SETUP_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _SETUP_ACTIONS,
                    "description": "The action to perform.",
                },
                "alias": {
                    "type": "string",
                    "description": "Arm alias for set_arm, rename_arm, remove_arm, or hand alias for set_hand/remove_hand.",
                },
                "arm_type": {
                    "type": "string",
                    "enum": ["so101_follower", "so101_leader", "koch_follower", "koch_leader"],
                    "description": "Arm hardware type for set_arm.",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path for set_arm or set_hand.",
                },
                "new_alias": {
                    "type": "string",
                    "description": "New arm alias for rename_arm.",
                },
                "camera_name": {
                    "type": "string",
                    "description": "Camera name like front or side.",
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Index into the live-detected camera list for set_camera.",
                },
                "target_action": {
                    "type": "string",
                    "description": "Action name to describe.",
                },
                "hand_type": {
                    "type": "string",
                    "enum": ["inspire_rh56", "revo2"],
                    "description": "Hand hardware type for set_hand.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_hardware": {
        "description": "Hardware identification and calibration for configured arms.",
        "actions": ["identify", "calibrate"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["identify", "calibrate"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_control": {
        "description": (
            "Teleoperate or record datasets. "
            "With checkpoint_path, record runs policy inference instead of teleop recording."
        ),
        "actions": ["teleoperate", "record"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["teleoperate", "record"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
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
                    "description": "Whether recording or policy should include configured cameras.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record or run.",
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
                "checkpoint_path": {
                    "type": "string",
                    "description": "Path to a trained policy checkpoint (turns record into policy inference).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_replay": {
        "description": "Replay a recorded dataset episode on follower arms.",
        "actions": ["replay"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["replay"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
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
    "embodied_train": {
        "description": "Train a policy on a recorded dataset, check training job status, or list datasets/policies.",
        "actions": ["train", "job_status", "list_datasets", "list_policies"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["train", "job_status", "list_datasets", "list_policies"],
                    "description": "The action to perform.",
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
                    "description": "Device for training (default: cuda).",
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
    "embodied_hand": {
        "description": "Control a dexterous hand: open, close, set finger pose, read status.",
        "actions": ["hand_open", "hand_close", "hand_pose", "hand_status"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["hand_open", "hand_close", "hand_pose", "hand_status"],
                    "description": "The hand action to perform.",
                },
                "hand_name": {
                    "type": "string",
                    "description": "Hand alias. Uses first configured hand if omitted.",
                },
                "positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "6 finger positions 0-1000 [little, ring, middle, index, thumb_bend, thumb_rotation] for hand_pose.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


class EmbodiedToolGroup(Tool):
    """A single tool group that dispatches to shared action functions."""

    def __init__(self, group_name: str, spec: dict[str, Any], tty_handoff: Any = None):
        self._group_name = group_name
        self._spec = spec
        self._tty_handoff = tty_handoff
        self.embodied_service = None  # Set after construction for lazy binding

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
        return await _dispatch(action, kwargs, self._tty_handoff, self.embodied_service)


def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all 6 groups."""
    return [
        EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff)
        for name, spec in _TOOL_GROUPS.items()
    ]


def _format_scan(result: dict[str, Any]) -> str:
    """Format scan results for CLI display."""
    ports = result["ports"]
    cameras = result["cameras"]
    lines = [f"Found {len(ports)} serial port(s) and {len(cameras)} camera(s)."]
    if ports:
        lines.append("\nPorts:")
        for p in ports:
            port_id = p.get("by_id") or p.get("dev", "?")
            motors = p.get("motor_ids", [])
            lines.append(f"  - {port_id}  ({len(motors)} motors)")
    if cameras:
        lines.append("\nCameras:")
        for c in cameras:
            dev = c.get("dev", "?")
            w, h = c.get("width", "?"), c.get("height", "?")
            fps = c.get("fps", "?")
            lines.append(f"  - {dev}  ({w}x{h} @ {fps}fps)")
    return "\n".join(lines)


def _get_service(service: Any) -> Any:
    """Return the provided service or create a default one."""
    if service is not None:
        return service
    from roboclaw.embodied.service import EmbodiedService
    return EmbodiedService()


async def _dispatch(
    action: str, kwargs: dict[str, Any], tty_handoff: Any, service: Any = None,
) -> str | list:
    svc = _get_service(service)

    # Config operations — no setup needed
    if action == "setup_show":
        return svc.queries.get_setup()
    if action == "scan":
        result = await asyncio.to_thread(svc.scanning.run_full_scan)
        return _format_scan(result)
    if action == "describe":
        return svc.queries.describe_actions(kwargs.get("target_action", ""))
    if action == "set_arm":
        return svc.config.set_arm(kwargs["alias"], kwargs["arm_type"], kwargs["port"])
    if action == "rename_arm":
        return svc.config.rename_arm(kwargs["alias"], kwargs["new_alias"])
    if action == "remove_arm":
        return svc.config.remove_arm(kwargs["alias"])
    if action == "set_camera":
        return svc.config.set_camera(kwargs.get("camera_name", ""), kwargs.get("camera_index", 0))
    if action == "preview_cameras":
        return svc.queries.preview_cameras()
    if action == "remove_camera":
        return svc.config.remove_camera(kwargs.get("camera_name", ""))
    if action == "set_hand":
        return svc.config.set_hand(kwargs["alias"], kwargs.get("hand_type", ""), kwargs.get("port", ""))
    if action == "remove_hand":
        return svc.config.remove_hand(kwargs["alias"])
    if action == "list_datasets":
        return svc.queries.list_datasets()
    if action == "list_policies":
        return svc.queries.list_policies()

    # Operations requiring setup — ActionError is a user-facing error
    # raised by helpers (e.g. missing arm), converted to a plain string.
    return await _dispatch_with_setup(action, kwargs, tty_handoff, svc)


async def _dispatch_with_setup(
    action: str, kwargs: dict[str, Any], tty_handoff: Any, svc: Any,
) -> str | list:
    from roboclaw.embodied.ops.helpers import ActionError
    from roboclaw.embodied.setup import ensure_setup

    setup = ensure_setup()

    try:
        return await _run_action(action, kwargs, tty_handoff, svc, setup)
    except ActionError as exc:
        return str(exc)


async def _run_action(
    action: str, kwargs: dict[str, Any], tty_handoff: Any, svc: Any, setup: dict,
) -> str | list:
    if action == "doctor":
        return await svc.run_doctor(setup, kwargs, tty_handoff)

    # Record with checkpoint_path => run policy (no CLI session needed)
    if action == "record" and kwargs.get("checkpoint_path"):
        return await svc.run_policy(setup, kwargs, tty_handoff)

    # Early dataset name validation for record
    if action == "record":
        dataset_name = kwargs.get("dataset_name")
        if dataset_name:
            from roboclaw.embodied.ops.helpers import _validate_dataset_name
            error = _validate_dataset_name(dataset_name)
            if error:
                return error

    if action in ("teleoperate", "record"):
        from roboclaw.embodied.adapters.cli import run_cli_session
        return await run_cli_session(svc, action, setup, kwargs, tty_handoff)

    if action == "calibrate":
        return await svc.run_calibrate(setup, kwargs, tty_handoff)
    if action == "identify":
        return await svc.run_identify(setup, kwargs, tty_handoff)
    if action == "replay":
        return await svc.run_replay(setup, kwargs, tty_handoff)
    if action == "train":
        return await svc.start_training(setup, kwargs, tty_handoff)
    if action == "job_status":
        return await svc.get_job_status(setup, kwargs, tty_handoff)

    if action == "hand_open":
        return await svc.hand_open(setup, kwargs, tty_handoff)
    if action == "hand_close":
        return await svc.hand_close(setup, kwargs, tty_handoff)
    if action == "hand_pose":
        return await svc.hand_pose(setup, kwargs, tty_handoff)
    if action == "hand_status":
        return await svc.hand_status(setup, kwargs, tty_handoff)

    return f"Unknown action: {action}"
