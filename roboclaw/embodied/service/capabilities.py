"""Hardware capability snapshots for control modes."""

from __future__ import annotations

from dataclasses import dataclass

from roboclaw.embodied.command.helpers import ActionError, group_arms, resolve_bimanual_pair
from roboclaw.embodied.embodiment.hardware.monitor import ArmStatus, CameraStatus
from roboclaw.embodied.embodiment.manifest.binding import ArmBinding


@dataclass(frozen=True)
class OperationCapability:
    """Readiness for one control operation."""

    ready: bool
    missing: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ready": self.ready, "missing": list(self.missing)}


@dataclass(frozen=True)
class ControlCapabilities:
    """Capability map exposed to the dashboard."""

    teleop: OperationCapability
    record: OperationCapability
    record_without_cameras: OperationCapability
    replay: OperationCapability
    infer: OperationCapability
    infer_without_cameras: OperationCapability

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "teleop": self.teleop.to_dict(),
            "record": self.record.to_dict(),
            "record_without_cameras": self.record_without_cameras.to_dict(),
            "replay": self.replay.to_dict(),
            "infer": self.infer.to_dict(),
            "infer_without_cameras": self.infer_without_cameras.to_dict(),
        }


@dataclass(frozen=True)
class HardwareSnapshot:
    """Full dashboard-facing hardware status."""

    ready: bool
    missing: list[str]
    arms: list[dict[str, object]]
    cameras: list[dict[str, object]]
    session_busy: bool
    capabilities: ControlCapabilities

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "missing": list(self.missing),
            "arms": list(self.arms),
            "cameras": list(self.cameras),
            "session_busy": self.session_busy,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass(frozen=True)
class OperationRequirements:
    """Resources required by one control mode."""

    require_leaders: bool = False
    require_cameras: bool = False


class HardwareContext:
    """Evaluates capability readiness for the configured arms and cameras.

    Owns the precomputed arm grouping plus status lookup so the six
    per-operation checks and the global status share a single pass.
    """

    def __init__(
        self,
        arms: list[ArmBinding],
        arm_statuses: list[ArmStatus],
        camera_statuses: list[CameraStatus],
    ) -> None:
        grouped = group_arms(arms)
        self.followers: list[ArmBinding] = grouped["followers"]
        self.leaders: list[ArmBinding] = grouped["leaders"]
        self.arm_statuses = arm_statuses
        self.camera_statuses = camera_statuses
        self._status_by_alias = {status.alias: status for status in arm_statuses}

    def snapshot(self, *, session_busy: bool) -> HardwareSnapshot:
        """Assemble the full dashboard-facing status."""
        global_missing = self.global_missing()
        return HardwareSnapshot(
            ready=not global_missing,
            missing=global_missing,
            arms=[status.to_dict() for status in self.arm_statuses],
            cameras=[status.to_dict() for status in self.camera_statuses],
            session_busy=session_busy,
            capabilities=ControlCapabilities(
                teleop=self.evaluate(OperationRequirements(require_leaders=True)),
                record=self.evaluate(
                    OperationRequirements(require_leaders=True, require_cameras=True),
                ),
                record_without_cameras=self.evaluate(
                    OperationRequirements(require_leaders=True),
                ),
                replay=self.evaluate(OperationRequirements()),
                infer=self.evaluate(OperationRequirements(require_cameras=True)),
                infer_without_cameras=self.evaluate(OperationRequirements()),
            ),
        )

    def evaluate(self, requirements: OperationRequirements) -> OperationCapability:
        missing = self._role_missing(self.followers, "follower", "followers")
        if requirements.require_leaders:
            missing.extend(self._role_missing(self.leaders, "leader", "leaders"))
            missing.extend(self._pair_count_missing())
        if requirements.require_cameras:
            missing.extend(self._required_cameras_missing())
        return OperationCapability(ready=not missing, missing=missing)

    def global_missing(self) -> list[str]:
        missing: list[str] = []
        missing.extend(self._role_missing(self.followers, "follower", "followers"))
        missing.extend(self._role_missing(self.leaders, "leader", "leaders"))
        missing.extend(self._disconnected_cameras())
        missing.extend(self._pair_count_missing())
        return missing

    def _role_missing(
        self,
        arms: list[ArmBinding],
        singular_name: str,
        plural_name: str,
    ) -> list[str]:
        if not arms:
            return [f"No {singular_name} arm configured"]

        missing: list[str] = []
        for arm in arms:
            status = self._status_by_alias[arm.alias]
            if not status.connected:
                missing.append(f"Arm '{status.alias}' is disconnected")
                continue
            if not status.calibrated:
                missing.append(f"Arm '{status.alias}' is not calibrated")
        missing.extend(_pairing_missing(arms, plural_name))
        return missing

    def _required_cameras_missing(self) -> list[str]:
        if not self.camera_statuses:
            return ["No cameras configured"]
        return self._disconnected_cameras()

    def _disconnected_cameras(self) -> list[str]:
        return [
            f"Camera '{status.alias}' is disconnected"
            for status in self.camera_statuses
            if not status.connected
        ]

    def _pair_count_missing(self) -> list[str]:
        if self.followers and self.leaders and len(self.followers) != len(self.leaders):
            return [
                f"Follower/leader count mismatch: {len(self.followers)} vs {len(self.leaders)}"
            ]
        return []


def build_hardware_snapshot(
    arms: list[ArmBinding],
    arm_statuses: list[ArmStatus],
    camera_statuses: list[CameraStatus],
    *,
    session_busy: bool,
) -> HardwareSnapshot:
    """Assemble global status plus per-operation capability states."""
    return HardwareContext(arms, arm_statuses, camera_statuses).snapshot(
        session_busy=session_busy,
    )


def _pairing_missing(arms: list[ArmBinding], role_name: str) -> list[str]:
    if len(arms) <= 1:
        return []
    if len(arms) != 2:
        return [f"Unsupported {role_name} arm count: {len(arms)}"]
    try:
        resolve_bimanual_pair(arms, role_name)
    except ActionError as exc:
        return [str(exc)]
    return []
