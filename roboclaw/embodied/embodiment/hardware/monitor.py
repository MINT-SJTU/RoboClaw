"""Background hardware health checker.

Periodically checks that configured arms and cameras are reachable,
emits events when faults appear or resolve.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.embodied.board.board import Board
    from roboclaw.embodied.embodiment.manifest import Manifest

from loguru import logger

from roboclaw.embodied.board.channels import CH_FAULT_DETECTED, CH_FAULT_RESOLVED
from roboclaw.embodied.embodiment.manifest.binding import ArmBinding, CameraBinding

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


def check_arm_status(arm: ArmBinding) -> ArmStatus:
    """Check a single arm's connectivity and calibration state."""
    return ArmStatus(
        alias=arm.alias,
        arm_type=arm.arm_type,
        role=arm.role.value,
        connected=arm.connected,
        calibrated=arm.calibrated,
    )


def check_camera_status(cam: CameraBinding) -> CameraStatus:
    """Check a single camera's connectivity."""
    port = (cam.port or "").strip()
    connected = cam.connected
    if port and not connected and port.isdigit():
        connected = port in scan_connected_camera_ports()
    return CameraStatus(
        alias=cam.alias,
        connected=connected,
        width=cam.interface.width,
        height=cam.interface.height,
    )


def scan_connected_camera_ports() -> set[str]:
    """Return currently discoverable camera identifiers from the active scan."""
    try:
        from roboclaw.embodied.embodiment.hardware.scan import scan_cameras
    except Exception:
        return set()

    ports: set[str] = set()
    try:
        for cam in scan_cameras():
            for value in (cam.dev, cam.by_id, cam.by_path):
                if value:
                    ports.add(str(value))
    except Exception:
        return set()
    return ports


def _fault_key(fault: HardwareFault) -> str:
    """Unique key for deduplicating active faults."""
    return f"{fault.fault_type.value}:{fault.device_alias}"


class HardwareMonitor:
    """Periodically checks hardware health and emits fault events."""

    def __init__(
        self,
        board: "Board | None" = None,
        manifest: "Manifest | None" = None,
    ) -> None:
        self._board = board
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
                if self._board is not None:
                    await self._board.emit(CH_FAULT_DETECTED, {
                        "fault_type": fault.fault_type.value,
                        "device_alias": fault.device_alias,
                        "message": fault.message,
                        "timestamp": fault.timestamp,
                    })

        # Detect resolved faults
        resolved_keys = set(self._active_faults.keys()) - set(current_keys.keys())
        for key in resolved_keys:
            resolved_fault = self._active_faults.pop(key)
            logger.info("Hardware fault resolved: {}", key)
            if self._board is not None:
                await self._board.emit(CH_FAULT_RESOLVED, {
                    "fault_type": resolved_fault.fault_type.value,
                    "device_alias": resolved_fault.device_alias,
                    "timestamp": time.time(),
                })

    def check_hardware(self) -> list[HardwareFault]:
        """Check all configured devices and return current faults."""
        if self._manifest is not None:
            arms = self._manifest.arms
            cameras = self._manifest.cameras
        else:
            from roboclaw.embodied.embodiment.manifest import Manifest
            manifest = Manifest()
            arms = manifest.arms
            cameras = manifest.cameras
        now = time.time()
        faults: list[HardwareFault] = []
        _check_arms(arms, now, faults)
        _check_cameras(cameras, now, faults, self._recording_active)
        return faults


def _check_arms(
    arms: list[ArmBinding], now: float, faults: list[HardwareFault],
) -> None:
    """Check arm connectivity and calibration state."""
    for arm in arms:
        status = check_arm_status(arm)
        if arm.port and not status.connected:
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
    cameras: list[CameraBinding],
    now: float,
    faults: list[HardwareFault],
    recording_active: bool,
) -> None:
    """Check camera connectivity (skip during active recording)."""
    if recording_active:
        return
    for cam in cameras:
        status = check_camera_status(cam)
        if cam.port and not status.connected:
            faults.append(HardwareFault(
                fault_type=FaultType.CAMERA_DISCONNECTED,
                device_alias=status.alias,
                message=f"Camera '{status.alias}' device not found",
                timestamp=now,
            ))
