"""Shared types for local and remote training backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TrainingProvider(str, Enum):
    """Supported training execution backends."""

    LOCAL = "local"
    ALIYUN = "aliyun"
    AUTODL = "autodl"


class TrainingJobState(str, Enum):
    """Coarse-grained lifecycle states for training jobs."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    MISSING = "missing"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TrainingJobState.IDLE,
            TrainingJobState.FINISHED,
            TrainingJobState.SUCCEEDED,
            TrainingJobState.FAILED,
            TrainingJobState.STOPPED,
            TrainingJobState.MISSING,
        }


@dataclass(frozen=True)
class JobResources:
    """Generic resource request shared by remote and local backends."""

    gpu_count: int = 1
    gpu_type: str = "A100"
    cpu_cores: int = 16
    memory_gb: int = 128
    node_count: int = 1
    image: str = ""
    ecs_spec: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "node_count": self.node_count,
            "image": self.image,
            "ecs_spec": self.ecs_spec,
        }


@dataclass(frozen=True)
class TrainingRequest:
    """Provider-agnostic training request assembled by TrainSession."""

    provider: TrainingProvider
    dataset_name: str
    dataset_repo_id: str
    dataset_local_path: Path
    train_argv: tuple[str, ...]
    output_dir: Path
    policy_type: str
    steps: int
    device: str
    job_name: str
    code_dir: Path
    entrypoint: str = ""
    resources: JobResources = field(default_factory=JobResources)
    env: dict[str, str] = field(default_factory=dict)
    wait: bool = False
    timeout_s: float | None = None
    poll_interval_s: float = 30.0
    auto_collect: bool = True
    remote_workdir: str = ""


@dataclass(frozen=True)
class TrainingSubmitResult:
    """Submit result returned by a backend."""

    job_id: str
    provider: TrainingProvider
    message: str
    remote_job_id: str = ""
    log_path: str = ""
    provider_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingJobRecord:
    """Persistent metadata for a training job across providers."""

    job_id: str
    provider: TrainingProvider
    created_at: float
    job_name: str
    dataset_name: str
    dataset_repo_id: str
    output_dir: str
    log_path: str = ""
    remote_job_id: str = ""
    provider_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider.value,
            "created_at": self.created_at,
            "job_name": self.job_name,
            "dataset_name": self.dataset_name,
            "dataset_repo_id": self.dataset_repo_id,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "remote_job_id": self.remote_job_id,
            "provider_data": self.provider_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJobRecord":
        return cls(
            job_id=str(data.get("job_id", "")),
            provider=TrainingProvider(str(data.get("provider", TrainingProvider.LOCAL.value))),
            created_at=float(data.get("created_at", 0.0)),
            job_name=str(data.get("job_name", "")),
            dataset_name=str(data.get("dataset_name", "")),
            dataset_repo_id=str(data.get("dataset_repo_id", "")),
            output_dir=str(data.get("output_dir", "")),
            log_path=str(data.get("log_path", "")),
            remote_job_id=str(data.get("remote_job_id", "")),
            provider_data=dict(data.get("provider_data", {})),
        )


@dataclass(frozen=True)
class TrainingJobStatus:
    """Normalized status payload for local and remote jobs."""

    job_id: str
    provider: TrainingProvider
    state: TrainingJobState
    running: bool
    message: str = ""
    remote_job_id: str = ""
    log_path: str = ""
    log_tail: str = ""
    output_dir: str = ""
    updated_at: float | None = None
    provider_data: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state.is_terminal

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "provider": self.provider.value,
            "status": self.state.value,
            "running": self.running,
            "terminal": self.terminal,
            "message": self.message,
            "remote_job_id": self.remote_job_id,
            "log_path": self.log_path,
            "log_tail": self.log_tail,
            "output_dir": self.output_dir,
            "updated_at": self.updated_at,
            "provider_data": self.provider_data,
        }
