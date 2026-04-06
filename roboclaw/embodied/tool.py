"""Embodied tool groups - bridges agent to the embodied robotics layer."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool

_SETUP_ACTIONS = ["identify", "modify"]
_DOCTOR_ACTIONS = ["check"]
_CALIBRATION_ACTIONS = ["calibrate"]
_TELEOP_ACTIONS = ["teleoperate"]
_RECORD_ACTIONS = ["record"]
_REPLAY_ACTIONS = ["replay"]
_TRAIN_ACTIONS = ["train", "job_status", "list_datasets", "list_policies"]
_INFER_ACTIONS = ["run_policy"]
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
                    "enum": ["rename", "unbind"],
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
        return f"Unknown tool group: {self._group_name}"

    async def _execute_setup(self, kwargs: dict[str, Any]) -> str | list:
        svc = _get_service(self.embodied_service)
        action = kwargs["action"]
        if action == "identify":
            return await _run_with_service(
                svc,
                lambda _: svc.setup.run_identify(kwargs, self._tty_handoff),
            )
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
            lambda manifest: svc.calibration_session.calibrate(manifest, kwargs, self._tty_handoff),
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
            lambda manifest: svc.replay.replay(manifest, kwargs, self._tty_handoff),
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
            lambda manifest: svc.infer.run_policy(manifest, kwargs, self._tty_handoff),
        )


def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all tool groups."""
    return [EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff) for name, spec in _TOOL_GROUPS.items()]


_MODIFY_DISPATCH = {
    ("unbind", "arm"): "unbind_arm",
    ("unbind", "camera"): "unbind_camera",
    ("unbind", "hand"): "unbind_hand",
    ("rename", "arm"): "rename_arm",
    ("rename", "camera"): "rename_camera",
    ("rebind", "arm"): "rebind_arm",
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

    new_alias = kwargs.get("new_alias", "")

    if operation == "rebind":
        new_type = kwargs.get("new_type", "")
        if not new_type:
            return "rebind requires new_type (e.g., 'koch_follower')."
        if not new_alias:
            new_alias = alias
        result = svc.rebind_arm(alias, new_alias, new_type)
        return f"Arm rebound: '{alias}' → '{new_alias}' ({new_type}).\n{json.dumps(result, indent=2)}"

    # rename
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
    from roboclaw.embodied.engine.helpers import ActionError

    try:
        return await func(service.manifest)
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
