"""Typed training contracts inspired by LlamaFactory's explicit argument schemas."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping

_log = logging.getLogger(__name__)

TrainingMode = Literal["local", "cloud"]
PolicySource = Literal["local", "cloud"]

_JOB_STATUS_KEYS = (
    "job_id",
    "status",
    "running",
    "pid",
    "log_path",
    "log_tail",
    "task_name",
    "checkpoint_path",
    "dataset_path",
    "provider",
)


@dataclass(frozen=True)
class TrainingStartSpec:
    dataset_name: str
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"
    username: str = ""
    provider: str = ""
    workflow: str = ""
    params: Mapping[str, Any] | None = None
    sku_id: str = ""
    image_id: str = ""
    task_name: str = ""
    wait_for_submit: bool = True


@dataclass(frozen=True)
class TrainingPlanSpec:
    username: str = ""
    message: str = ""
    workflow: str = ""
    params: Mapping[str, Any] | None = None
    provider: str = ""
    sku_id: str = ""
    image_id: str = ""


@dataclass(frozen=True)
class TrainingStopSpec:
    job_id: str
    username: str = ""


@dataclass(frozen=True)
class TrainingJobStatus:
    job_id: str = ""
    status: str = "idle"
    running: bool = False
    message: str = ""
    mode: TrainingMode = "local"
    pid: int | None = None
    log_path: str = ""
    log_tail: str = ""
    task_name: str = ""
    checkpoint_path: str = ""
    dataset_path: str = ""
    provider: str = ""
    error: str = ""
    failure_remediation: Mapping[str, Any] | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        mode: TrainingMode,
        message: str | None = None,
    ) -> "TrainingJobStatus":
        return cls(
            job_id=_as_text(payload.get("job_id")),
            status=_as_text(payload.get("status")) or "idle",
            running=_as_bool(payload.get("running")),
            message=message if message is not None else format_status_message(payload),
            mode=mode,
            pid=_as_int_or_none(payload.get("pid")),
            log_path=_as_text(payload.get("log_path")),
            log_tail=_as_text(payload.get("log_tail")),
            task_name=_as_text(payload.get("task_name")),
            checkpoint_path=_as_text(payload.get("checkpoint_path")),
            dataset_path=_as_text(payload.get("dataset_path")),
            provider=_as_text(payload.get("provider")),
            error=_as_text(payload.get("error")),
            failure_remediation=_as_mapping_or_none(payload.get("failureRemediation") or payload.get("failure_remediation")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "running": self.running,
            "message": self.message,
            "mode": self.mode,
            "pid": self.pid,
            "log_path": self.log_path,
            "log_tail": self.log_tail,
            "task_name": self.task_name,
            "checkpoint_path": self.checkpoint_path,
            "dataset_path": self.dataset_path,
            "provider": self.provider,
            "error": self.error,
            "failureRemediation": dict(self.failure_remediation or {}),
        }


@dataclass(frozen=True)
class TrainingPolicyEntry:
    name: str
    checkpoint: str
    dataset: str = ""
    steps: int | None = None
    source: PolicySource = "local"
    deployable: bool = True
    provider: str = ""
    status: str = ""
    job_id: str = ""
    task_name: str = ""
    updated_at: str = ""

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        source: PolicySource,
        deployable: bool | None = None,
    ) -> "TrainingPolicyEntry":
        return cls(
            name=_as_text(payload.get("name")),
            checkpoint=_as_text(payload.get("checkpoint")),
            dataset=_as_text(payload.get("dataset")),
            steps=_as_int_or_none(payload.get("steps")),
            source=source,
            deployable=deployable if deployable is not None else _as_bool(payload.get("deployable"), True),
            provider=_as_text(payload.get("provider")),
            status=_as_text(payload.get("status")),
            job_id=_as_text(payload.get("job_id")),
            task_name=_as_text(payload.get("task_name")),
            updated_at=_as_text(payload.get("updated_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "steps": self.steps,
            "source": self.source,
            "deployable": self.deployable,
            "provider": self.provider,
            "status": self.status,
            "job_id": self.job_id,
            "task_name": self.task_name,
            "updated_at": self.updated_at,
        }


def format_status_message(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("message")
    if isinstance(explicit, str) and explicit.strip():
        return explicit

    lines: list[str] = []
    seen: set[str] = set()
    for key in _JOB_STATUS_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            continue
        seen.add(key)
        lines.append(f"{key}: {value}")

    for key, value in payload.items():
        if key in seen or key == "message" or value is None:
            continue
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "running"}:
            return True
        if lowered in {"false", "0", "no", "idle", "finished", "stopped", "missing"}:
            return False
    return bool(value)


def _as_mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        _log.warning("Invalid integer value in training status payload: %r (%s)", value, exc)
        return None
