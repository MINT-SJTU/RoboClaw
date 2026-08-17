"""Replay routes — dataset playback on follower arms."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.service import EmbodiedService


class ReplayStartRequest(BaseModel):
    dataset_name: str
    episode: int = 0
    fps: int = 30


def register_replay_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.post("/api/replay/start")
    async def replay_start(body: ReplayStartRequest) -> dict[str, Any]:
        try:
            await service.start_replay(
                dataset_name=body.dataset_name,
                episode=body.episode,
                fps=body.fps,
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "replaying"}

    @app.post("/api/replay/stop")
    async def replay_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}
