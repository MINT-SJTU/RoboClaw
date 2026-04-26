from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from roboclaw.http import runtime as runtime_mod
from roboclaw.http import server


class _FakeChannelManager:
    def get_channel(self, _name: str) -> None:
        return None


class _FakeRuntime:
    channel_manager = _FakeChannelManager()
    embodied_service = None
    hw_monitor = None

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def test_create_app_registers_explorer_routes_without_embodied_service(
    monkeypatch,
    tmp_path,
) -> None:
    config = SimpleNamespace(
        workspace_path=tmp_path,
        channels=SimpleNamespace(
            web={
                "host": "127.0.0.1",
                "port": 8765,
                "cors_origins": [],
            },
        ),
    )

    monkeypatch.setattr(server, "load_runtime_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(server, "sync_workspace_templates", lambda _workspace: None)
    monkeypatch.setattr(
        runtime_mod.WebRuntime,
        "build",
        staticmethod(lambda *_args, **_kwargs: _FakeRuntime()),
    )

    app = server.create_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/explorer/datasets", params={"source": "local"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), list)
