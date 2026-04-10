"""Session lifecycle routes — teleop / record / episode control."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.service import EmbodiedService


class RecordStartRequest(BaseModel):
    task: str
    num_episodes: int = 10
    fps: int = 30
    episode_time_s: int = 300
    reset_time_s: int = 10
    dataset_name: str = ""
    use_cameras: bool = True
    arms: str = ""


class TeleopStartRequest(BaseModel):
    fps: int = 30
    arms: str = ""


def register_session_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.get("/api/session/status")
    async def session_status() -> dict[str, Any]:
        return service.get_status()

    @app.post("/api/session/dismiss-error")
    async def dismiss_error() -> dict[str, str]:
        await service.dismiss_error()
        return {"status": "idle"}

    @app.post("/api/teleop/start")
    async def teleop_start(body: TeleopStartRequest | None = None) -> dict[str, str]:
        fps = body.fps if body else 30
        arms = body.arms if body else ""
        await service.start_teleop(fps=fps, arms=arms)
        return {"status": "teleoperating"}

    @app.post("/api/teleop/stop")
    async def teleop_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}

    @app.post("/api/record/start")
    async def record_start(body: RecordStartRequest) -> dict[str, Any]:
        dataset_name = await service.start_recording(
            task=body.task,
            num_episodes=body.num_episodes,
            fps=body.fps,
            episode_time_s=body.episode_time_s,
            reset_time_s=body.reset_time_s,
            dataset_name=body.dataset_name,
            use_cameras=body.use_cameras,
            arms=body.arms,
        )
        return {"status": "recording", "dataset_name": dataset_name}

    @app.post("/api/record/stop")
    async def record_stop() -> dict[str, str]:
        await service.stop()
        return {"status": "idle"}

    @app.post("/api/record/episode/save")
    async def episode_save() -> dict[str, str]:
        try:
            await service.save_episode()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "episode_saved"}

    @app.post("/api/record/episode/discard")
    async def episode_discard() -> dict[str, str]:
        try:
            await service.discard_episode()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "episode_discarded"}

    @app.post("/api/record/episode/skip-reset")
    async def episode_skip_reset() -> dict[str, str]:
        try:
            await service.skip_reset()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"status": "reset_skipped"}
