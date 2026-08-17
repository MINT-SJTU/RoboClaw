"""HuggingFace Hub push/pull routes for datasets and policies."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from roboclaw.embodied.service import EmbodiedService


class HubPushRequest(BaseModel):
    repo_id: str  # e.g. "username/dataset-name"
    token: str = ""
    private: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.repo_id.strip() or "/" not in self.repo_id:
            raise ValueError("repo_id must be in 'namespace/name' format")


class HubPullRequest(BaseModel):
    repo_id: str
    token: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.repo_id.strip() or "/" not in self.repo_id:
            raise ValueError("repo_id must be in 'namespace/name' format")


class DatasetHubPushRequest(HubPushRequest):
    dataset_id: str

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")


class DatasetHubPullRequest(HubPullRequest):
    dataset_id: str = ""


class PolicyHubPushRequest(HubPushRequest):
    name: str

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.name.strip():
            raise ValueError("name must not be empty")


class PolicyHubPullRequest(HubPullRequest):
    name: str = ""


def register_hub_routes(app: FastAPI, service: EmbodiedService) -> None:

    async def _call(method, body) -> dict[str, Any]:
        try:
            result = await method(service.manifest, body.model_dump(), None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"message": result}

    @app.post("/api/hub/datasets/push")
    async def hub_datasets_push(body: DatasetHubPushRequest) -> dict[str, Any]:
        return await _call(service.hub.push_dataset, body)

    @app.post("/api/hub/datasets/pull")
    async def hub_datasets_pull(body: DatasetHubPullRequest) -> dict[str, Any]:
        return await _call(service.hub.pull_dataset, body)

    @app.post("/api/hub/policies/push")
    async def hub_policies_push(body: PolicyHubPushRequest) -> dict[str, Any]:
        return await _call(service.hub.push_policy, body)

    @app.post("/api/hub/policies/pull")
    async def hub_policies_pull(body: PolicyHubPullRequest) -> dict[str, Any]:
        return await _call(service.hub.pull_policy, body)
