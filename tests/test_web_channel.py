"""Optional tests for the web channel transport."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.bus.queue import MessageBus
from roboclaw.channels.web import WebChannel
from roboclaw.session.manager import SessionManager


def test_web_channel_health_and_session_routes(tmp_path) -> None:
    channel = WebChannel(
        SimpleNamespace(allow_from=["*"], cors_origins=["http://localhost:5173"]),
        MessageBus(),
        session_manager=SessionManager(tmp_path),
    )
    app = FastAPI()
    channel.register_routes(app)
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "channel": "web"}

    session = client.get("/api/chat/sessions/demo")
    assert session.status_code == 200
    assert session.json() == {"chat_id": "demo", "messages": []}
