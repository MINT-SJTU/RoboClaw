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


@dataclass(frozen=True)
class _HardwareContext:
    """Precomputed arm grouping and status lookup, shared across evaluations."""

    followers: list[ArmBinding]
    leaders: list[ArmBinding]
    status_by_alias: dict[str, ArmStatus]
    camera_statuses: list[CameraStatus]

    @classmethod
    def build(
        cls,
        arms: list[ArmBinding],
        arm_statuses: list[ArmStatus],
        camera_statuses: list[CameraStatus],
    ) -> "_HardwareContext":
        grouped = group_arms(arms)
        return cls(
            followers=grouped["followers"],
            leaders=grouped["leaders"],
            status_by_alias={status.alias: status for status in arm_statuses},
            camera_statuses=camera_statuses,
        )


def build_hardware_snapshot(
    arms: list[ArmBinding],
    arm_statuses: list[ArmStatus],
    camera_statuses: list[CameraStatus],
    *,
    session_busy: bool,
) -> HardwareSnapshot:
    """Assemble global status plus per-operation capability states."""

    ctx = _HardwareContext.build(arms, arm_statuses, camera_statuses)
    capabilities = ControlCapabilities(
        teleop=_evaluate_operation(ctx, OperationRequirements(require_leaders=True)),
        record=_evaluate_operation(
            ctx, OperationRequirements(require_leaders=True, require_cameras=True),
        ),
        record_without_cameras=_evaluate_operation(
            ctx, OperationRequirements(require_leaders=True),
        ),
        replay=_evaluate_operation(ctx, OperationRequirements()),
        infer=_evaluate_operation(ctx, OperationRequirements(require_cameras=True)),
        infer_without_cameras=_evaluate_operation(ctx, OperationRequirements()),
    )
    global_missing = _global_missing(ctx)
    return HardwareSnapshot(
        ready=not global_missing,
        missing=global_missing,
        arms=[status.to_dict() for status in arm_statuses],
        cameras=[status.to_dict() for status in camera_statuses],
        session_busy=session_busy,
        capabilities=capabilities,
    )


def _evaluate_operation(
    ctx: _HardwareContext,
    requirements: OperationRequirements,
) -> OperationCapability:
    missing = _arm_role_missing(ctx.followers, ctx.status_by_alias, "follower", "followers")
    if requirements.require_leaders:
        missing.extend(_arm_role_missing(ctx.leaders, ctx.status_by_alias, "leader", "leaders"))
        if ctx.followers and ctx.leaders and len(ctx.followers) != len(ctx.leaders):
            missing.append(
                f"Follower/leader count mismatch: {len(ctx.followers)} vs {len(ctx.leaders)}"
            )

    if requirements.require_cameras:
        missing.extend(_camera_missing(ctx.camera_statuses))

    return OperationCapability(ready=not missing, missing=missing)


def _arm_role_missing(
    arms: list[ArmBinding],
    status_by_alias: dict[str, ArmStatus],
    singular_name: str,
    plural_name: str,
) -> list[str]:
    if not arms:
        return [f"No {singular_name} arm configured"]

    missing: list[str] = []
    for arm in arms:
        status = status_by_alias[arm.alias]
        if not status.connected:
            missing.append(f"Arm '{status.alias}' is disconnected")
            continue
        if not status.calibrated:
            missing.append(f"Arm '{status.alias}' is not calibrated")
    missing.extend(_pairing_missing(arms, plural_name))
    return missing


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


def _camera_missing(camera_statuses: list[CameraStatus]) -> list[str]:
    if not camera_statuses:
        return ["No cameras configured"]
    return [
        f"Camera '{status.alias}' is disconnected"
        for status in camera_statuses
        if not status.connected
    ]


def _global_missing(ctx: _HardwareContext) -> list[str]:
    missing: list[str] = []
    missing.extend(_arm_role_missing(ctx.followers, ctx.status_by_alias, "follower", "followers"))
    missing.extend(_arm_role_missing(ctx.leaders, ctx.status_by_alias, "leader", "leaders"))
    missing.extend(
        f"Camera '{status.alias}' is disconnected"
        for status in ctx.camera_statuses
        if not status.connected
    )
    if ctx.followers and ctx.leaders and len(ctx.followers) != len(ctx.leaders):
        missing.append(
            f"Follower/leader count mismatch: {len(ctx.followers)} vs {len(ctx.leaders)}"
        )
    return missing
