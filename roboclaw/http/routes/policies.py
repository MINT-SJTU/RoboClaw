"""Policy list route."""

from __future__ import annotations

from fastapi import FastAPI

from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingService


def register_policy_routes(app: FastAPI, service: EmbodiedService) -> None:
    training = TrainingService(service)

    @app.get("/api/policies")
    async def policies_list_route(username: str = "") -> list[dict]:
        try:
            entries = await training.list_policies(username=username)
        except Exception:
            entries = service.train.list_policy_entries(service.manifest)
        return [entry.to_dict() for entry in entries]
