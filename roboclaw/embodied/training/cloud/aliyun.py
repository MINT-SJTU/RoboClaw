"""Alibaba Cloud PAI-DLC backend for remote policy training."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from roboclaw.embodied.training.backend import BaseTrainingBackend
from roboclaw.embodied.training.common import (
    make_tarball,
    remote_entrypoint_for_request,
    timestamp_slug,
)
from roboclaw.embodied.training.types import (
    JobResources,
    TrainingJobRecord,
    TrainingJobState,
    TrainingJobStatus,
    TrainingProvider,
    TrainingRequest,
    TrainingSubmitResult,
)

logger = logging.getLogger(__name__)

_REMOTE_BASE_DIR = "/workspace/roboclaw_training"
_REMOTE_DATASET_DIR = f"{_REMOTE_BASE_DIR}/dataset"
_REMOTE_OUTPUT_DIR = f"{_REMOTE_BASE_DIR}/outputs"
_REMOTE_CODE_DIR = f"{_REMOTE_BASE_DIR}/code"


class JobStatus(str, Enum):
    """Subset of PAI-DLC job states used by RoboClaw."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @classmethod
    def from_raw(cls, raw: str) -> "JobStatus":
        raw_lower = (raw or "").lower()
        if raw_lower in {"succeeded"}:
            return cls.SUCCEEDED
        if raw_lower in {"failed", "stopped", "stopping"}:
            return cls.FAILED
        if raw_lower in {"running"}:
            return cls.RUNNING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED}


@dataclass
class AliyunTrainingConfig:
    """Credentials and workspace settings for PAI-DLC + OSS."""

    access_key_id: str
    access_key_secret: str
    region_id: str
    workspace_id: str
    oss_bucket: str
    oss_endpoint: str
    oss_prefix: str = "roboclaw-training"
    resource_group_id: str | None = None
    pai_endpoint: str | None = None

    def dlc_endpoint(self) -> str:
        if self.pai_endpoint:
            return self.pai_endpoint
        return f"pai-dlc.{self.region_id}.aliyuncs.com"

    @classmethod
    def from_env(cls, prefix: str = "ROBOCLAW_ALIYUN_") -> "AliyunTrainingConfig":
        def _req(key: str) -> str:
            name = f"{prefix}{key}"
            value = os.environ.get(name)
            if not value:
                raise ValueError(f"Missing required env var: {name}")
            return value

        return cls(
            access_key_id=_req("ACCESS_KEY_ID"),
            access_key_secret=_req("ACCESS_KEY_SECRET"),
            region_id=_req("REGION_ID"),
            workspace_id=_req("WORKSPACE_ID"),
            oss_bucket=_req("OSS_BUCKET"),
            oss_endpoint=_req("OSS_ENDPOINT"),
            oss_prefix=os.environ.get(f"{prefix}OSS_PREFIX", "roboclaw-training"),
            resource_group_id=os.environ.get(f"{prefix}RESOURCE_GROUP_ID"),
            pai_endpoint=os.environ.get(f"{prefix}PAI_ENDPOINT"),
        )


class AliyunCloudTrainer:
    """Thin wrapper around PAI-DLC + OSS SDKs."""

    def __init__(
        self,
        config: AliyunTrainingConfig,
        *,
        dlc_client: Any = None,
        oss_bucket: Any = None,
    ) -> None:
        self.config = config
        self._dlc = dlc_client if dlc_client is not None else self._build_dlc_client()
        self._oss = oss_bucket if oss_bucket is not None else self._build_oss_bucket()

    def _build_dlc_client(self):
        from alibabacloud_pai_dlc20201203.client import Client as DlcClient
        from alibabacloud_tea_openapi.models import Config as OpenApiConfig

        cfg = OpenApiConfig(
            access_key_id=self.config.access_key_id,
            access_key_secret=self.config.access_key_secret,
            region_id=self.config.region_id,
            endpoint=self.config.dlc_endpoint(),
        )
        return DlcClient(cfg)

    def _build_oss_bucket(self):
        import oss2

        auth = oss2.Auth(self.config.access_key_id, self.config.access_key_secret)
        return oss2.Bucket(auth, self.config.oss_endpoint, self.config.oss_bucket)

    def submit_job(
        self,
        *,
        job_name: str,
        code_dir: str | Path,
        dataset_dir: str | Path,
        entrypoint: str,
        resources: JobResources | None = None,
        env: dict[str, str] | None = None,
        job_id_prefix: str = "rc",
    ) -> str:
        """Stage code + dataset to OSS and submit a PAI-DLC job."""
        resources = resources or JobResources()
        code_dir = Path(code_dir).resolve()
        dataset_dir = Path(dataset_dir).resolve()
        if not code_dir.is_dir():
            raise FileNotFoundError(f"code_dir does not exist: {code_dir}")
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

        stamp = timestamp_slug()
        uid = uuid.uuid4().hex[:8]
        code_key = f"{self.config.oss_prefix}/code/{job_id_prefix}-{stamp}-{uid}.tar.gz"
        dataset_key = f"{self.config.oss_prefix}/datasets/{job_id_prefix}-{stamp}-{uid}.tar.gz"
        output_prefix = f"{self.config.oss_prefix}/outputs/{job_id_prefix}-{stamp}-{uid}/"

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            code_archive = tmp_dir / "code.tar.gz"
            dataset_archive = tmp_dir / "dataset.tar.gz"
            make_tarball(code_dir, code_archive)
            make_tarball(dataset_dir, dataset_archive)
            self._oss.put_object_from_file(code_key, str(code_archive))
            self._oss.put_object_from_file(dataset_key, str(dataset_archive))

        from alibabacloud_pai_dlc20201203 import models as dlc_models

        full_command = _compose_command(
            code_oss_uri=f"oss://{self.config.oss_bucket}/{code_key}",
            dataset_oss_uri=f"oss://{self.config.oss_bucket}/{dataset_key}",
            output_oss_uri=f"oss://{self.config.oss_bucket}/{output_prefix}",
            user_entrypoint=entrypoint,
        )
        image = resources.image or (
            "registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:"
            "2.4.0-gpu-py310-cu121-ubuntu22.04"
        )
        ecs_spec = self._resolve_ecs_spec(resources)
        job_spec = dlc_models.JobSpec(
            type="Worker",
            image=image,
            pod_count=resources.node_count,
            ecs_spec=ecs_spec,
        )
        request = dlc_models.CreateJobRequest(
            display_name=job_name,
            job_type="PyTorchJob",
            job_specs=[job_spec],
            workspace_id=self.config.workspace_id,
            resource_id=self.config.resource_group_id,
            user_command=full_command,
            envs={
                **(env or {}),
                "ROBOCLAW_OUTPUT_OSS_URI": f"oss://{self.config.oss_bucket}/{output_prefix}",
            },
        )
        try:
            response = self._dlc.create_job(request)
        except Exception as exc:
            raise ValueError(_format_submit_error(exc)) from exc
        job_id = str(response.body.job_id)
        marker_key = f"{self.config.oss_prefix}/markers/{job_id}.txt"
        self._oss.put_object(marker_key, output_prefix.encode("utf-8"))
        logger.info("Submitted Aliyun PAI-DLC job {} ({})", job_id, job_name)
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        request = dlc_models.GetJobRequest()
        response = self._dlc.get_job(job_id, request)
        raw = getattr(response.body, "status", None) or ""
        return JobStatus.from_raw(str(raw))

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 30.0,
        timeout: float | None = None,
    ) -> JobStatus:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.get_job_status(job_id)
            if status.is_terminal:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} did not finish within {timeout}s.")
            time.sleep(max(poll_interval, 0.1))

    def download_artifacts(self, job_id: str, local_dir: str | Path) -> list[Path]:
        """Download all output artifacts for a remote job."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = self._lookup_output_prefix(job_id)
        written: list[Path] = []
        for obj_key in self._iter_oss_keys(output_prefix):
            relative = obj_key[len(output_prefix):].lstrip("/")
            if not relative:
                continue
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self._oss.get_object_to_file(obj_key, str(target))
            written.append(target)
        return written

    def cancel_job(self, job_id: str) -> None:
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        request = dlc_models.StopJobRequest()
        self._dlc.stop_job(job_id, request)

    def _lookup_output_prefix(self, job_id: str) -> str:
        marker_key = f"{self.config.oss_prefix}/markers/{job_id}.txt"
        stream = self._oss.get_object(marker_key)
        return stream.read().decode("utf-8")

    def _iter_oss_keys(self, prefix: str) -> Iterable[str]:
        import oss2

        for obj in oss2.ObjectIterator(self._oss, prefix=prefix):
            yield obj.key

    def _list_ecs_specs(self) -> list[Any]:
        from alibabacloud_pai_dlc20201203 import models as dlc_models

        specs: list[Any] = []
        page_number = 1
        page_size = 100
        while True:
            request = dlc_models.ListEcsSpecsRequest(page_number=page_number, page_size=page_size)
            response = self._dlc.list_ecs_specs(request)
            page_specs = list(getattr(response.body, "ecs_specs", None) or [])
            if not page_specs:
                break
            specs.extend(page_specs)
            total_count = int(getattr(response.body, "total_count", len(specs)) or len(specs))
            if len(specs) >= total_count or len(page_specs) < page_size:
                break
            page_number += 1
        return specs

    def _resolve_ecs_spec(self, resources: JobResources) -> str:
        if resources.ecs_spec.strip():
            return resources.ecs_spec.strip()

        requested_gpu_count = max(int(resources.gpu_count), 0)
        requested_gpu_type = resources.gpu_type.strip().upper() if requested_gpu_count else ""
        requested_cpu = max(int(resources.cpu_cores), 0)
        requested_memory = max(int(resources.memory_gb), 0)

        specs = [spec for spec in self._list_ecs_specs() if getattr(spec, "is_available", True)]
        if requested_gpu_count:
            specs = [spec for spec in specs if int(getattr(spec, "gpu", 0) or 0) == requested_gpu_count]
        else:
            specs = [spec for spec in specs if int(getattr(spec, "gpu", 0) or 0) == 0]
        if not specs:
            raise ValueError(
                f"No available Aliyun ECS specs found for gpu_count={requested_gpu_count} "
                f"in region {self.config.region_id}."
            )

        if requested_gpu_type:
            exact_gpu_type = [
                spec for spec in specs
                if str(getattr(spec, "gpu_type", "") or "").upper() == requested_gpu_type
            ]
            if not exact_gpu_type:
                available = sorted({
                    str(getattr(spec, "gpu_type", "") or "").upper()
                    for spec in specs
                    if str(getattr(spec, "gpu_type", "") or "").strip()
                })
                detail = ", ".join(available) if available else "none"
                raise ValueError(
                    f"No available Aliyun ECS spec matches gpu_type '{resources.gpu_type}' "
                    f"with gpu_count={requested_gpu_count}. Available gpu types: {detail}."
                )
            specs = exact_gpu_type

        sized = [
            spec for spec in specs
            if int(getattr(spec, "cpu", 0) or 0) >= requested_cpu
            and int(getattr(spec, "memory", 0) or 0) >= requested_memory
        ]
        if not sized:
            options = ", ".join(_describe_ecs_spec(spec) for spec in specs[:5])
            raise ValueError(
                "No available Aliyun ECS spec satisfies "
                f"cpu>={requested_cpu} and memory>={requested_memory}Gi "
                f"for gpu_type='{resources.gpu_type or 'CPU'}', gpu_count={requested_gpu_count}. "
                f"Closest available options: {options}."
            )

        sized.sort(
            key=lambda spec: (
                int(getattr(spec, "cpu", 0) or 0),
                int(getattr(spec, "memory", 0) or 0),
                str(getattr(spec, "instance_type", "") or ""),
            )
        )
        return str(getattr(sized[0], "instance_type", "") or "")


class AliyunTrainingBackend(BaseTrainingBackend):
    """Provider adapter that wires generic TrainSession requests to PAI-DLC."""

    provider = TrainingProvider.ALIYUN

    def __init__(
        self,
        config: AliyunTrainingConfig | None = None,
        trainer: AliyunCloudTrainer | None = None,
    ) -> None:
        self._config = config
        self._trainer = trainer

    def _trainer_client(self) -> AliyunCloudTrainer:
        if self._trainer is not None:
            return self._trainer
        config = self._config or AliyunTrainingConfig.from_env()
        self._trainer = AliyunCloudTrainer(config)
        return self._trainer

    async def submit(self, request: TrainingRequest) -> TrainingSubmitResult:
        trainer = self._trainer_client()
        entrypoint = remote_entrypoint_for_request(
            request,
            dataset_root=_REMOTE_DATASET_DIR,
            output_dir=_REMOTE_OUTPUT_DIR,
        )
        local_job_id = uuid.uuid4().hex[:12]
        remote_job_id = await asyncio.to_thread(
            trainer.submit_job,
            job_name=request.job_name,
            code_dir=request.code_dir,
            dataset_dir=request.dataset_local_path,
            entrypoint=entrypoint,
            resources=request.resources,
            env=request.env,
            job_id_prefix=local_job_id,
        )
        return TrainingSubmitResult(
            job_id=local_job_id,
            provider=self.provider,
            message=(
                f"Aliyun training submitted. Job ID: {local_job_id}\n"
                f"Provider job ID: {remote_job_id}"
            ),
            remote_job_id=remote_job_id,
        )

    async def status(self, record: TrainingJobRecord) -> TrainingJobStatus:
        remote_job_id = record.remote_job_id
        if not remote_job_id:
            return TrainingJobStatus(
                job_id=record.job_id,
                provider=self.provider,
                state=TrainingJobState.MISSING,
                running=False,
                message="Missing Aliyun provider job id.",
                output_dir=record.output_dir,
            )
        job_status = await asyncio.to_thread(self._trainer_client().get_job_status, remote_job_id)
        state = _generic_state(job_status)
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=state,
            running=state is TrainingJobState.RUNNING or state is TrainingJobState.QUEUED,
            message=job_status.value,
            remote_job_id=remote_job_id,
            output_dir=record.output_dir,
        )

    async def stop(self, record: TrainingJobRecord) -> TrainingJobStatus:
        remote_job_id = record.remote_job_id
        if not remote_job_id:
            return TrainingJobStatus(
                job_id=record.job_id,
                provider=self.provider,
                state=TrainingJobState.MISSING,
                running=False,
                message="Missing Aliyun provider job id.",
                output_dir=record.output_dir,
            )
        await asyncio.to_thread(self._trainer_client().cancel_job, remote_job_id)
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=TrainingJobState.STOPPED,
            running=False,
            message="stop_requested",
            remote_job_id=remote_job_id,
            output_dir=record.output_dir,
        )

    async def collect(
        self,
        record: TrainingJobRecord,
        *,
        output_dir: str | None = None,
    ) -> list[str]:
        remote_job_id = record.remote_job_id
        if not remote_job_id:
            return []
        target = Path(output_dir or record.output_dir)
        written = await asyncio.to_thread(
            self._trainer_client().download_artifacts,
            remote_job_id,
            target,
        )
        return [str(path) for path in written]


def _resource_config(resources: JobResources) -> dict[str, str]:
    config = {
        "cpu": str(resources.cpu_cores),
        "memory": f"{resources.memory_gb}Gi",
    }
    if resources.gpu_count > 0:
        config["gpu"] = str(resources.gpu_count)
    if resources.gpu_count > 0 and resources.gpu_type.strip():
        config["gputype"] = resources.gpu_type
    return config


def _describe_ecs_spec(spec: Any) -> str:
    return (
        f"{getattr(spec, 'instance_type', '')}"
        f"(gpu_type={getattr(spec, 'gpu_type', '') or 'CPU'}, "
        f"gpu={getattr(spec, 'gpu', 0) or 0}, "
        f"cpu={getattr(spec, 'cpu', 0) or 0}, "
        f"memory={getattr(spec, 'memory', 0) or 0}Gi)"
    )


def _format_submit_error(exc: Exception) -> str:
    message = str(exc).strip()
    if "EcsSpec of JobSpec(Worker) must be present" in message:
        return (
            "Aliyun training submission failed because the request is missing a valid ecs_spec. "
            "Choose an available GPU type for your region, such as A10 or V100, or set ecs_spec explicitly."
        )
    return f"Aliyun training submission failed: {message}"


def _generic_state(status: JobStatus) -> TrainingJobState:
    if status is JobStatus.RUNNING:
        return TrainingJobState.RUNNING
    if status is JobStatus.SUCCEEDED:
        return TrainingJobState.SUCCEEDED
    if status is JobStatus.FAILED:
        return TrainingJobState.FAILED
    return TrainingJobState.QUEUED


def _compose_command(
    *,
    code_oss_uri: str,
    dataset_oss_uri: str,
    output_oss_uri: str,
    user_entrypoint: str,
) -> str:
    """Build the container command for the PAI-DLC worker."""
    return (
        "set -e && "
        "OSSUTIL_BIN=$(command -v ossutil64 || command -v ossutil || true) && "
        'if [ -z "$OSSUTIL_BIN" ]; then echo "Missing ossutil binary (expected ossutil64 or ossutil)." >&2; exit 127; fi && '
        f"mkdir -p {_REMOTE_CODE_DIR} {_REMOTE_DATASET_DIR} {_REMOTE_OUTPUT_DIR}/logs && "
        f'"$OSSUTIL_BIN" cp -f {code_oss_uri} {_REMOTE_BASE_DIR}/code.tar.gz && '
        f'"$OSSUTIL_BIN" cp -f {dataset_oss_uri} {_REMOTE_BASE_DIR}/dataset.tar.gz && '
        f"tar -xzf {_REMOTE_BASE_DIR}/code.tar.gz -C {_REMOTE_CODE_DIR} && "
        f"tar -xzf {_REMOTE_BASE_DIR}/dataset.tar.gz -C {_REMOTE_DATASET_DIR} && "
        f"cd {_REMOTE_CODE_DIR} && "
        "set +e && "
        f"( {user_entrypoint} ) > {_REMOTE_OUTPUT_DIR}/logs/train.log 2>&1; "
        "rc=$?; "
        f"echo $rc > {_REMOTE_OUTPUT_DIR}/logs/exit_code; "
        f'"$OSSUTIL_BIN" cp -r -f {_REMOTE_OUTPUT_DIR}/ {output_oss_uri}; '
        "exit $rc"
    )
