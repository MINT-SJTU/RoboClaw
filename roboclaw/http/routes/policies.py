"""Policy list route."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI

from roboclaw.embodied.service import EmbodiedService
from roboclaw.http.dashboard_policies import list_policies


def _policies_root(service: EmbodiedService) -> Path:
    root = service.manifest.snapshot.get("policies", {}).get("root", "")
    if root:
        return Path(root).expanduser()
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "policies"


def register_policy_routes(app: FastAPI, service: EmbodiedService) -> None:

    @app.get("/api/policies")
    async def policies_list_route() -> list[dict]:
        return await asyncio.to_thread(list_policies, _policies_root(service))
