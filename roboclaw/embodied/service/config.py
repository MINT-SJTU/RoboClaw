"""Configuration sub-service: CRUD for arms, cameras, and hands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.manifest.helpers import (
    _probe_hand_slave_id,
    _resolve_serial_interface,
    arm_display_name,
    find_arm,
    find_camera,
    find_hand,
)

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


class ConfigService:
    """Manages setup configuration: add/remove/rename arms, cameras, hands."""

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    def set_arm(self, alias: str, arm_type: str, port: str) -> str:
        if not all([alias, arm_type, port]):
            return "set_arm requires alias, arm_type, and port."
        interface = _resolve_serial_interface(port)
        self._parent.manifest.set_arm(alias, arm_type, interface)
        arm = self._parent.manifest.find_arm(alias)
        display = arm_display_name(arm)
        return f"Arm '{display}' configured.\n{json.dumps(arm, indent=2)}"

    def rename_arm(self, old_alias: str, new_alias: str) -> str:
        if not old_alias or not new_alias:
            return "rename_arm requires alias and new_alias."
        self._parent.manifest.rename_arm(old_alias, new_alias)
        arm = self._parent.manifest.find_arm(new_alias)
        return f"Arm renamed from '{old_alias}' to '{new_alias}'.\n{json.dumps(arm, indent=2)}"

    def remove_arm(self, alias: str) -> str:
        if not alias:
            return "remove_arm requires alias."
        if self._parent.embodiment_busy:
            return f"Cannot remove arm while embodiment is busy: {self._parent.busy_reason}"
        self._parent.manifest.remove_arm(alias)
        return f"Arm '{alias}' removed."

    def set_camera(self, camera_name: str, camera_index: int) -> str:
        if not camera_name or camera_index is None:
            return "set_camera requires camera_name and camera_index."
        from roboclaw.embodied.hardware.scan import scan_cameras

        scanned = scan_cameras()
        if camera_index < 0 or camera_index >= len(scanned):
            return (
                f"camera_index {camera_index} out of range. "
                f"Found {len(scanned)} camera(s)."
            )
        interface = scanned[camera_index]
        if not interface.address:
            return f"Scanned camera at index {camera_index} has no usable path."
        self._parent.manifest.set_camera(camera_name, interface)
        cam = self._parent.manifest.find_camera(camera_name)
        return f"Camera '{camera_name}' configured.\n{json.dumps(cam, indent=2)}"

    def remove_camera(self, camera_name: str) -> str:
        if not camera_name:
            return "remove_camera requires camera_name."
        if self._parent.embodiment_busy:
            return f"Cannot remove camera while embodiment is busy: {self._parent.busy_reason}"
        self._parent.manifest.remove_camera(camera_name)
        return f"Camera '{camera_name}' removed."

    def set_hand(self, alias: str, hand_type: str, port: str) -> str:
        if not all([alias, hand_type, port]):
            return "set_hand requires alias, hand_type, and port."
        interface = _resolve_serial_interface(port)
        slave_id = _probe_hand_slave_id(hand_type, interface.address)
        self._parent.manifest.set_hand(alias, hand_type, interface, slave_id)
        hand = self._parent.manifest.find_hand(alias)
        return f"Hand '{alias}' configured.\n{json.dumps(hand, indent=2)}"

    def remove_hand(self, alias: str) -> str:
        if not alias:
            return "remove_hand requires alias."
        self._parent.manifest.remove_hand(alias)
        return f"Hand '{alias}' removed."
