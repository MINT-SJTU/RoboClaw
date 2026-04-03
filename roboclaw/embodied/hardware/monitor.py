"""Background hardware health checker.

Periodically checks that configured arms and cameras are reachable,
emits events when faults appear or resolve.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.embodied.events import EventBus, FaultDetectedEvent, FaultResolvedEvent

_CHECK_INTERVAL_SECONDS = 5


class FaultType(str, Enum):
    ARM_DISCONNECTED = "arm_disconnected"
    ARM_TIMEOUT = "arm_timeout"
    ARM_NOT_CALIBRATED = "arm_not_calibrated"
    CAMERA_DISCONNECTED = "camera_disconnected"
    CAMERA_FRAME_DROP = "camera_frame_drop"
    RECORD_CRASHED = "record_crashed"


@dataclass
class HardwareFault:
    fault_type: FaultType
    device_alias: str
    message: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fault_type"] = self.fault_type.value
        return d


@dataclass
class ArmStatus:
    """Connectivity and calibration status for a single arm."""

    alias: str
    arm_type: str
    role: str  # "follower" | "leader" | ""
    connected: bool
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias, "type": self.arm_type, "role": self.role,
            "connected": self.connected, "calibrated": self.calibrated,
        }


@dataclass
class CameraStatus:
    """Connectivity status for a single camera."""

    alias: str
    connected: bool
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_arm_status(arm: dict[str, Any]) -> ArmStatus:
    """Check a single arm's connectivity and calibration state."""
    alias = arm.get("alias", "unknown")
    port = arm.get("port", "")
    connected = bool(port and Path(port).exists())
    calibrated = bool(arm.get("calibrated", False))
    arm_type = arm.get("type", "")
    role = "follower" if "follower" in arm_type else "leader" if "leader" in arm_type else ""
    return ArmStatus(alias=alias, arm_type=arm_type, role=role, connected=connected, calibrated=calibrated)


def check_camera_status(cam: dict[str, Any]) -> CameraStatus:
    """Check a single camera's connectivity."""
    alias = cam.get("alias", "unknown")
    port = cam.get("port", "")
    connected = bool(port and Path(port).exists())
    return CameraStatus(
        alias=alias, connected=connected,
        width=cam.get("width", 640), height=cam.get("height", 480),
    )


def _fault_key(fault: HardwareFault) -> str:
    """Unique key for deduplicating active faults."""
    return f"{fault.fault_type.value}:{fault.device_alias}"


class HardwareMonitor:
    """Periodically checks hardware health and emits fault events."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        manifest: "Manifest | None" = None,
    ) -> None:
        self._bus = event_bus
        self._manifest = manifest
        self._active_faults: dict[str, HardwareFault] = {}
        self._recording_active = False
        self._stop_event = asyncio.Event()

    @property
    def active_faults(self) -> list[HardwareFault]:
        return list(self._active_faults.values())

    def set_recording_active(self, active: bool) -> None:
        self._recording_active = active

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Main loop: check hardware every N seconds until stopped."""
        logger.info("Hardware monitor started")
        while not self._stop_event.is_set():
            await self._tick()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=_CHECK_INTERVAL_SECONDS
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # normal interval elapsed
        logger.info("Hardware monitor stopped")

    async def _tick(self) -> None:
        """Run one check cycle, diff against active faults, emit events."""
        current_faults = self.check_hardware()
        current_keys = {_fault_key(f): f for f in current_faults}

        # Detect new faults
        for key, fault in current_keys.items():
            if key not in self._active_faults:
                self._active_faults[key] = fault
                logger.warning("Hardware fault detected: {} — {}", key, fault.message)
                if self._bus is not None:
                    await self._bus.emit(FaultDetectedEvent(
                        fault_type=fault.fault_type.value,
                        device_alias=fault.device_alias,
                        message=fault.message,
                    ))

        # Detect resolved faults
        resolved_keys = set(self._active_faults.keys()) - set(current_keys.keys())
        for key in resolved_keys:
            resolved_fault = self._active_faults.pop(key)
            logger.info("Hardware fault resolved: {}", key)
            if self._bus is not None:
                await self._bus.emit(FaultResolvedEvent(
                    fault_type=resolved_fault.fault_type.value,
                    device_alias=resolved_fault.device_alias,
                ))

    def check_hardware(self) -> list[HardwareFault]:
        """Check all configured devices and return current faults."""
        if self._manifest is not None:
            manifest = self._manifest.snapshot
        else:
            from roboclaw.embodied.manifest.helpers import load_manifest
            manifest = load_manifest()
        now = time.time()
        faults: list[HardwareFault] = []
        _check_arms(manifest.get("arms", []), now, faults)
        _check_cameras(manifest.get("cameras", []), now, faults, self._recording_active)
        return faults


def _check_arms(
    arms: list[dict[str, Any]], now: float, faults: list[HardwareFault],
) -> None:
    """Check arm connectivity and calibration state."""
    for arm in arms:
        status = check_arm_status(arm)
        if arm.get("port") and not status.connected:
            faults.append(HardwareFault(
                fault_type=FaultType.ARM_DISCONNECTED,
                device_alias=status.alias,
                message=f"Arm '{status.alias}' USB port not found",
                timestamp=now,
            ))
            continue
        if not status.calibrated:
            faults.append(HardwareFault(
                fault_type=FaultType.ARM_NOT_CALIBRATED,
                device_alias=status.alias,
                message=f"Arm '{status.alias}' is not calibrated",
                timestamp=now,
            ))


def _check_cameras(
    cameras: list[dict[str, Any]],
    now: float,
    faults: list[HardwareFault],
    recording_active: bool,
) -> None:
    """Check camera connectivity (skip during active recording)."""
    if recording_active:
        return
    for cam in cameras:
        status = check_camera_status(cam)
        if cam.get("port") and not status.connected:
            faults.append(HardwareFault(
                fault_type=FaultType.CAMERA_DISCONNECTED,
                device_alias=status.alias,
                message=f"Camera '{status.alias}' device not found",
                timestamp=now,
            ))


