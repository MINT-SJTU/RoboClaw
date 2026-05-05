"""Training routes — policy training lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.service import EmbodiedService


class TrainStartRequest(BaseModel):
    dataset_name: str
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"


class TrainStopRequest(BaseModel):
    job_id: str


def register_train_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.post("/api/train/start")
    async def train_start(body: TrainStartRequest) -> dict[str, Any]:
        try:
            return await service.start_training(
                dataset_name=body.dataset_name,
                policy_type=body.policy_type,
                steps=body.steps,
                device=body.device,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/train/stop")
    async def train_stop(body: TrainStopRequest) -> dict[str, Any]:
        result = await service.train.stop_job(
            manifest=service.manifest,
            kwargs={"job_id": body.job_id},
            tty_handoff=None,
        )
        return {"message": result}

    @app.get("/api/train/current")
    async def train_current() -> dict[str, Any]:
        return await service.train.current_job(
            manifest=service.manifest,
            kwargs={},
            tty_handoff=None,
        )

    @app.get("/api/train/status/{job_id}")
    async def train_status(job_id: str) -> dict[str, Any]:
        result = await service.train.job_status(
            manifest=service.manifest,
            kwargs={"job_id": job_id},
            tty_handoff=None,
        )
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
