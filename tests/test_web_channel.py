"""Tests for the web channel transport and chat upload routes."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.bus.queue import MessageBus
from roboclaw.channels.web import WebChannel
from roboclaw.http.routes.chat_uploads import register_chat_upload_routes
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


def test_chat_upload_image_and_fetch(tmp_path) -> None:
    app = FastAPI()
    with patch("roboclaw.http.routes.chat_uploads.get_workspace_path", return_value=tmp_path):
        register_chat_upload_routes(app)
        client = TestClient(app)

        raw = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

        upload = client.post(
            "/api/chat/uploads/image",
            json={
                "chat_id": "demo",
                "filename": "sample.png",
                "data_url": data_url,
            },
        )
        assert upload.status_code == 200
        payload = upload.json()
        assert payload["name"] == "sample.png"
        assert payload["mime_type"] == "image/png"
        assert "media_path" not in payload
        assert payload["preview_url"].endswith(".png")

        uploaded_path = tmp_path / "chat_uploads" / "demo"
        assert uploaded_path.exists()
        assert len(list(uploaded_path.iterdir())) == 1

        preview = client.get(payload["preview_url"])
        assert preview.status_code == 200
        assert preview.content == raw


def test_chat_upload_rejects_non_image_bytes(tmp_path) -> None:
    app = FastAPI()
    with patch("roboclaw.http.routes.chat_uploads.get_workspace_path", return_value=tmp_path):
        register_chat_upload_routes(app)
        client = TestClient(app)

        raw = b"not-an-image"
        data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

        upload = client.post(
            "/api/chat/uploads/image",
            json={"chat_id": "demo", "filename": "fake.png", "data_url": data_url},
        )
        assert upload.status_code == 400
        assert "Unsupported" in upload.json()["detail"]


def test_web_channel_session_history_returns_empty_metadata(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("web:demo")
    session.messages.append({
        "id": "user-1",
        "role": "user",
        "content": "hello",
        "timestamp": "2026-04-02T00:00:00",
    })
    manager.save(session)

    channel = WebChannel(
        SimpleNamespace(allow_from=["*"], cors_origins=["http://localhost:5173"]),
        MessageBus(),
        session_manager=manager,
    )
    app = FastAPI()
    channel.register_routes(app)
    client = TestClient(app)

    response = client.get("/api/chat/sessions/demo")
    assert response.status_code == 200
    msg = response.json()["messages"][0]
    assert msg["content"] == "hello"
    assert msg["metadata"] == {}
