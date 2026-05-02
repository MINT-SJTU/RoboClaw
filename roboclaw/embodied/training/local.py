"""Local detached subprocess backend for policy training."""

from __future__ import annotations

from pathlib import Path

from roboclaw.embodied.command import logs_dir
from roboclaw.embodied.executor import SubprocessExecutor
from roboclaw.embodied.training.backend import BaseTrainingBackend
from roboclaw.embodied.training.types import (
    TrainingJobRecord,
    TrainingJobState,
    TrainingJobStatus,
    TrainingProvider,
    TrainingRequest,
    TrainingSubmitResult,
)


class LocalTrainingBackend(BaseTrainingBackend):
    """Run training locally via the existing detached subprocess executor."""

    provider = TrainingProvider.LOCAL

    async def submit(self, request: TrainingRequest) -> TrainingSubmitResult:
        job_id = await SubprocessExecutor().run_detached(argv=list(request.train_argv), log_dir=logs_dir())
        log_path = str(logs_dir() / f"{job_id}.log")
        return TrainingSubmitResult(
            job_id=job_id,
            provider=self.provider,
            message=f"Training started. Job ID: {job_id}",
            remote_job_id=job_id,
            log_path=log_path,
        )

    async def status(self, record: TrainingJobRecord) -> TrainingJobStatus:
        payload = await SubprocessExecutor().job_status(job_id=record.job_id, log_dir=logs_dir())
        raw_status = str(payload.get("status", "unknown"))
        running = bool(payload.get("running", False))
        if raw_status == "running":
            state = TrainingJobState.RUNNING
        elif raw_status == "finished":
            state = TrainingJobState.FINISHED
        elif raw_status == "missing":
            state = TrainingJobState.MISSING
        elif raw_status == "stopped":
            state = TrainingJobState.STOPPED
        else:
            state = TrainingJobState.UNKNOWN
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=state,
            running=running,
            message=raw_status,
            remote_job_id=record.remote_job_id or record.job_id,
            log_path=str(payload.get("log_path", record.log_path)),
            log_tail=str(payload.get("log_tail", "")),
            output_dir=record.output_dir,
            provider_data={"pid": payload.get("pid")},
        )

    async def stop(self, record: TrainingJobRecord) -> TrainingJobStatus:
        payload = await SubprocessExecutor().stop_job(job_id=record.job_id, log_dir=logs_dir())
        running = bool(payload.get("running", False))
        raw_status = str(payload.get("status", "unknown"))
        state = TrainingJobState.RUNNING if running else TrainingJobState.STOPPED
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=state,
            running=running,
            message=raw_status,
            remote_job_id=record.remote_job_id or record.job_id,
            log_path=str(payload.get("log_path", record.log_path)),
            log_tail=str(payload.get("log_tail", "")),
            output_dir=record.output_dir,
            provider_data={"pid": payload.get("pid")},
        )

    async def collect(
        self,
        record: TrainingJobRecord,
        *,
        output_dir: str | None = None,
    ) -> list[str]:
        target = Path(output_dir or record.output_dir)
        if target.exists():
            return [str(target)]
        return []

