"""Structured state for embodied setup onboarding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


SETUP_STATE_KEY = "embodied_onboarding"


class SetupStage(StrEnum):
    """High-level onboarding stage for one setup."""

    IDENTIFY_SETUP_SCOPE = "identify_setup_scope"
    CONFIRM_CONNECTED = "confirm_connected"
    PROBE_LOCAL_ENVIRONMENT = "probe_local_environment"
    RESOLVE_PREREQUISITES = "resolve_prerequisites"
    INSTALL_PREREQUISITES = "install_prerequisites"
    VALIDATE_PREREQUISITES = "validate_prerequisites"
    MATERIALIZE_ASSEMBLY = "materialize_assembly"
    MATERIALIZE_DEPLOYMENT_ADAPTER = "materialize_deployment_adapter"
    VALIDATE_SETUP = "validate_setup"
    HANDOFF_READY = "handoff_ready"


class SetupStatus(StrEnum):
    """Lifecycle status for one setup."""

    BOOTSTRAPPING = "bootstrapping"
    READY = "ready"
    REFINING = "refining"


@dataclass(frozen=True)
class SetupOnboardingState:
    """JSON-serializable setup onboarding state."""

    setup_id: str
    intake_slug: str
    assembly_id: str
    deployment_id: str
    adapter_id: str
    stage: SetupStage = SetupStage.IDENTIFY_SETUP_SCOPE
    status: SetupStatus = SetupStatus.BOOTSTRAPPING
    robot_attachments: list[dict[str, Any]] = field(default_factory=list)
    sensor_attachments: list[dict[str, Any]] = field(default_factory=list)
    execution_targets: list[dict[str, Any]] = field(default_factory=list)
    detected_facts: dict[str, Any] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    generated_assets: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status == SetupStatus.READY and self.stage == SetupStage.HANDOFF_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "intake_slug": self.intake_slug,
            "assembly_id": self.assembly_id,
            "deployment_id": self.deployment_id,
            "adapter_id": self.adapter_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "robot_attachments": self.robot_attachments,
            "sensor_attachments": self.sensor_attachments,
            "execution_targets": self.execution_targets,
            "detected_facts": self.detected_facts,
            "missing_facts": self.missing_facts,
            "generated_assets": self.generated_assets,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SetupOnboardingState":
        return cls(
            setup_id=data["setup_id"],
            intake_slug=data["intake_slug"],
            assembly_id=data["assembly_id"],
            deployment_id=data["deployment_id"],
            adapter_id=data["adapter_id"],
            stage=SetupStage(data.get("stage", SetupStage.IDENTIFY_SETUP_SCOPE.value)),
            status=SetupStatus(data.get("status", SetupStatus.BOOTSTRAPPING.value)),
            robot_attachments=list(data.get("robot_attachments", [])),
            sensor_attachments=list(data.get("sensor_attachments", [])),
            execution_targets=list(data.get("execution_targets", [])),
            detected_facts=dict(data.get("detected_facts", {})),
            missing_facts=list(data.get("missing_facts", [])),
            generated_assets=dict(data.get("generated_assets", {})),
            notes=list(data.get("notes", [])),
        )
