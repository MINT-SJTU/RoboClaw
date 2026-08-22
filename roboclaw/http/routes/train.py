"""Training routes — policy training lifecycle."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from roboclaw.embodied.command import ActionError
from roboclaw.embodied.service import EmbodiedService


class TrainStartRequest(BaseModel):
    dataset_name: str
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"
    provider: str = "local"
    preset: str = ""
    job_name: str = ""
    code_dir: str = ""
    entrypoint: str = ""
    gpu_count: int = 1
    gpu_type: str = "A100"
    cpu_cores: int = 16
    memory_gb: int = 128
    node_count: int = 1
    image: str = ""
    ecs_spec: str = ""
    wait: bool = False
    timeout_s: float | None = None
    poll_interval_s: float = 30.0
    auto_collect: bool = True
    remote_workdir: str = ""
    env: dict[str, str] = Field(default_factory=dict)


class TrainStopRequest(BaseModel):
    job_id: str


class TrainCollectRequest(BaseModel):
    job_id: str
    output_dir: str = ""


def register_train_routes(app: FastAPI, service: EmbodiedService) -> None:

    def _bad_request(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/train/start")
    async def train_start(body: TrainStartRequest) -> dict[str, Any]:
        try:
            result = await service.train.train(
                manifest=service.manifest,
                kwargs={
                    "dataset_name": body.dataset_name,
                    "policy_type": body.policy_type,
                    "steps": body.steps,
                    "device": body.device,
                    "provider": body.provider,
                    "preset": body.preset,
                    "job_name": body.job_name,
                    "code_dir": body.code_dir,
                    "entrypoint": body.entrypoint,
                    "gpu_count": body.gpu_count,
                    "gpu_type": body.gpu_type,
                    "cpu_cores": body.cpu_cores,
                    "memory_gb": body.memory_gb,
                    "node_count": body.node_count,
                    "image": body.image,
                    "ecs_spec": body.ecs_spec,
                    "wait": body.wait,
                    "timeout_s": body.timeout_s,
                    "poll_interval_s": body.poll_interval_s,
                    "auto_collect": body.auto_collect,
                    "remote_workdir": body.remote_workdir,
                    "env": body.env,
                },
                tty_handoff=None,
            )
        except (ActionError, ValueError) as exc:
            raise _bad_request(exc) from exc
        job_id = _extract_message_value(result, "Job ID")
        provider_job_id = _extract_message_value(result, "Provider job ID")
        return {"message": result, "job_id": job_id, "provider_job_id": provider_job_id}

    @app.post("/api/train/stop")
    async def train_stop(body: TrainStopRequest) -> dict[str, Any]:
        try:
            result = await service.train.stop_job(
                manifest=service.manifest,
                kwargs={"job_id": body.job_id},
                tty_handoff=None,
            )
        except (ActionError, ValueError) as exc:
            raise _bad_request(exc) from exc
        return {"message": result}

    @app.get("/api/train/current")
    async def train_current() -> dict[str, Any]:
        return await service.train.current_job(
            manifest=service.manifest,
            kwargs={},
            tty_handoff=None,
        )

    @app.get("/api/train/capabilities")
    async def train_capabilities() -> dict[str, Any]:
        return await asyncio.to_thread(service.train.capabilities)

    @app.get("/api/train/status/{job_id}")
    async def train_status(job_id: str) -> dict[str, Any]:
        try:
            payload = await service.train.job_status_payload(
                manifest=service.manifest,
                kwargs={"job_id": job_id},
                tty_handoff=None,
            )
        except (ActionError, ValueError) as exc:
            raise _bad_request(exc) from exc
        result = "\n".join(f"{key}: {value}" for key, value in payload.items())
        return {"message": result, **payload}

    @app.post("/api/train/collect")
    async def train_collect(body: TrainCollectRequest) -> dict[str, Any]:
        try:
            result = await service.train.collect_job(
                manifest=service.manifest,
                kwargs={"job_id": body.job_id, "output_dir": body.output_dir},
                tty_handoff=None,
            )
        except (ActionError, ValueError) as exc:
            raise _bad_request(exc) from exc
        return {"message": result}

    @app.get("/api/train/curve/{job_id}")
    async def train_curve(job_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(service.train.curve_data, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/train/datasets")
    async def train_datasets() -> dict[str, Any]:
        result = service.train.list_datasets(service.manifest)
        return {"message": result}

    @app.get("/api/train/policies")
    async def train_policies() -> dict[str, Any]:
        result = service.train.list_policies(service.manifest)
        return {"message": result}


def _extract_message_value(message: str, label: str) -> str:
    pattern = rf"(?:^|\n)[^\n]*{re.escape(label)}:\s*([^\n]+)"
    match = re.search(pattern, message)
    return match.group(1).strip() if match else ""
