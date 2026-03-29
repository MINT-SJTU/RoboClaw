"""Embodied tool groups - bridges agent to the embodied robotics layer."""

from __future__ import annotations

from typing import Any

from roboclaw.agent.tools.base import Tool

_SETUP_ACTIONS = [
    "setup_show", "set_arm", "remove_arm", "rename_arm",
    "set_camera", "preview_cameras", "remove_camera", "describe", "doctor",
    "set_hand", "remove_hand",
]

_TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "embodied_setup": {
        "description": (
            "Configure robot hardware: show setup, add/remove/rename arms, "
            "add/remove cameras, describe actions, check environment. "
            "Also manages dexterous hands: set_hand, remove_hand."
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
                    "enum": ["so101_follower", "so101_leader"],
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
        return await _dispatch(action, kwargs, self._tty_handoff)


def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all 6 groups."""
    return [
        EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff)
        for name, spec in _TOOL_GROUPS.items()
    ]


_NO_SETUP_ACTIONS = frozenset({
    "setup_show", "describe", "set_arm", "rename_arm",
    "remove_arm", "set_camera", "preview_cameras", "remove_camera",
    "set_hand", "remove_hand",
    "list_datasets", "list_policies",
})


async def _dispatch(action: str, kwargs: dict[str, Any], tty_handoff: Any) -> str | list:
    from roboclaw.embodied.ops.configure import SYNC_DISPATCH
    from roboclaw.embodied.ops.execute import ASYNC_DISPATCH
    from roboclaw.embodied.ops.helpers import ActionError

    if action in _NO_SETUP_ACTIONS:
        return SYNC_DISPATCH[action](kwargs)

    from roboclaw.embodied.setup import ensure_setup

    setup = ensure_setup()
    try:
        return await ASYNC_DISPATCH[action](setup, kwargs, tty_handoff)
    except ActionError as exc:
        return str(exc)
