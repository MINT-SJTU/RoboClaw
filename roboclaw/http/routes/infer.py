"""Inference routes — trained policy rollout."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from roboclaw.embodied.service import EmbodiedService


class InferStartRequest(BaseModel):
    checkpoint_path: str = ""
    dataset_name: str = ""
    task: str = "eval"
    num_episodes: int = 1
    episode_time_s: int = 60


def register_infer_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.post("/api/infer/start")
    async def infer_start(body: InferStartRequest) -> dict[str, Any]:
        await service.start_inference(
            checkpoint_path=body.checkpoint_path,
            dataset_name=body.dataset_name,
            task=body.task,
            num_episodes=body.num_episodes,
            episode_time_s=body.episode_time_s,
        )
        return {"status": "inferring"}

    @app.post("/api/infer/stop")
    async def infer_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}
