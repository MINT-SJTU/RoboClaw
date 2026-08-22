"""Provider registry and abstract backend behavior for training jobs."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from roboclaw.embodied.command import ActionError
from roboclaw.embodied.training.types import (
    TrainingJobRecord,
    TrainingJobStatus,
    TrainingProvider,
    TrainingRequest,
    TrainingSubmitResult,
)


class TrainingBackend(Protocol):
    """Protocol implemented by each training execution backend."""

    provider: TrainingProvider

    async def submit(self, request: TrainingRequest) -> TrainingSubmitResult:
        ...

    async def status(self, record: TrainingJobRecord) -> TrainingJobStatus:
        ...

    async def stop(self, record: TrainingJobRecord) -> TrainingJobStatus:
        ...

    async def collect(
        self,
        record: TrainingJobRecord,
        *,
        output_dir: str | None = None,
    ) -> list[str]:
        ...

    async def wait(
        self,
        record: TrainingJobRecord,
        *,
        poll_interval_s: float,
        timeout_s: float | None,
    ) -> TrainingJobStatus:
        ...


class BaseTrainingBackend:
    """Default wait/collect behavior shared by provider implementations."""

    provider: TrainingProvider

    async def collect(
        self,
        record: TrainingJobRecord,
        *,
        output_dir: str | None = None,
    ) -> list[str]:
        return []

    async def wait(
        self,
        record: TrainingJobRecord,
        *,
        poll_interval_s: float,
        timeout_s: float | None,
    ) -> TrainingJobStatus:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            status = await self.status(record)
            if status.terminal:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Training job {record.job_id} did not finish within {timeout_s}s."
                )
            await asyncio.sleep(max(poll_interval_s, 0.1))


def build_training_backend(provider: TrainingProvider) -> TrainingBackend:
    """Instantiate the backend for *provider*."""
    if provider is TrainingProvider.LOCAL:
        from roboclaw.embodied.training.local import LocalTrainingBackend

        return LocalTrainingBackend()
    if provider is TrainingProvider.ALIYUN:
        from roboclaw.embodied.training.cloud.aliyun import AliyunTrainingBackend

        return AliyunTrainingBackend()
    if provider is TrainingProvider.AUTODL:
        from roboclaw.embodied.training.cloud.autodl import AutoDLTrainingBackend

        return AutoDLTrainingBackend()
    raise ActionError(f"Unsupported training provider: {provider.value}")

