"""Tests for the Web provider settings API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from roboclaw.config.loader import save_config, set_config_path
from roboclaw.config.schema import Config
from roboclaw.http.server import create_app


def test_provider_status_and_save_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    status = client.get("/api/system/provider-status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["active_provider_configured"] is False
    assert payload["custom_provider"]["configured"] is False

    save = client.post(
        "/api/system/provider-config",
        json={
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "sk-test",
        },
    )
    assert save.status_code == 200
    saved = save.json()
    assert saved["status"] == "ok"
    assert saved["custom_provider"]["configured"] is True
    assert saved["default_provider"] == "custom"
    assert saved["custom_provider"]["has_api_key"] is True
    assert saved["custom_provider"]["masked_api_key"] == "已保存"


def test_provider_save_auto_discovers_model(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    async def _fake_discover(api_base: str, api_key: str | None) -> str | None:
        assert api_base == "http://127.0.0.1:8000/v1"
        assert api_key == "sk-test"
        return "gpt-4.1-mini"

    monkeypatch.setattr("roboclaw.http.server._discover_custom_model", _fake_discover)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "sk-test",
        },
    )
    assert save.status_code == 200
    saved = save.json()
    assert saved["default_model"] == "gpt-4.1-mini"
    assert saved["custom_provider"]["masked_api_key"] == "已保存" or saved["custom_provider"]["masked_api_key"].startswith("sk-te")


def test_provider_save_uses_explicit_model(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    async def _unexpected_discover(api_base: str, api_key: str | None) -> str | None:
        raise AssertionError("explicit model should skip auto-discovery")

    monkeypatch.setattr("roboclaw.http.server._discover_custom_model", _unexpected_discover)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "sk-test",
            "model": "openai/gpt-4.1-mini",
        },
    )
    assert save.status_code == 200
    saved = save.json()
    assert saved["default_model"] == "openai/gpt-4.1-mini"


def test_provider_models_discovers_from_payload(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    async def _fake_discover(api_base: str, api_key: str | None, extra_headers=None) -> tuple[list[str], str]:
        assert api_base == "http://127.0.0.1:8000/v1"
        assert api_key == "sk-test"
        assert extra_headers is None
        return ["gpt-4.1-mini", "deepseek-chat"], ""

    monkeypatch.setattr("roboclaw.http.server._discover_provider_models", _fake_discover)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post(
        "/api/system/provider-models",
        json={
            "provider": "custom",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "sk-test",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"models": ["gpt-4.1-mini", "deepseek-chat"], "error": ""}


def test_provider_models_rejects_unknown_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post("/api/system/provider-models", json={"provider": "missing"})
    assert response.status_code == 400
