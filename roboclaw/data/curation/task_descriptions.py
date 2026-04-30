from __future__ import annotations

from typing import Any

TASK_DESCRIPTION_KEYS = (
    "task",
    "tasks",
    "task_label",
    "instruction",
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
)

TASK_DESCRIPTION_NEEDLES = ("task", "instruction", "language")
TASK_IDENTITY_KEYS = {"task_index", "task_id", "task_uid"}


def payload_has_task_description(payload: dict[str, Any]) -> bool:
    """Return whether a metadata row contains human-readable task text."""
    for key in TASK_DESCRIPTION_KEYS:
        if value_has_task_description(payload.get(key)):
            return True

    for key, value in payload.items():
        if not is_task_description_key(key):
            continue
        if value_has_task_description(value):
            return True
    return False


def value_has_task_description(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(value_has_task_description(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(value_has_task_description(item) for item in value)
    return False


def is_task_description_key(key: Any) -> bool:
    key_lower = str(key).lower()
    if key_lower in TASK_IDENTITY_KEYS:
        return False
    return any(needle in key_lower for needle in TASK_DESCRIPTION_NEEDLES)
