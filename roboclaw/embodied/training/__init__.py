"""Training backends for local and remote execution providers."""

from roboclaw.embodied.training.backend import BaseTrainingBackend, build_training_backend
from roboclaw.embodied.training.local import LocalTrainingBackend
from roboclaw.embodied.training.registry import TrainingJobStore
from roboclaw.embodied.training.types import (
    JobResources,
    TrainingJobRecord,
    TrainingJobState,
    TrainingJobStatus,
    TrainingProvider,
    TrainingRequest,
    TrainingSubmitResult,
)

__all__ = [
    "BaseTrainingBackend",
    "JobResources",
    "LocalTrainingBackend",
    "TrainingJobRecord",
    "TrainingJobState",
    "TrainingJobStatus",
    "TrainingJobStore",
    "TrainingProvider",
    "TrainingRequest",
    "TrainingSubmitResult",
    "build_training_backend",
]
