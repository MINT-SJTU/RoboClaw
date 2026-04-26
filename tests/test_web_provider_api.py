"""Tests for the Web provider settings API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from roboclaw.config.loader import save_config, set_config_path
from roboclaw.config.schema import Config
from roboclaw.http.server import create_app
from roboclaw.providers.base import LLMResponse


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
    assert saved["custom_provider"]["default_model"] == "gpt-4o-mini"
    assert "gpt-5.5" in saved["custom_provider"]["model_presets"]
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


def test_provider_save_rejects_invalid_config_without_persisting(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "openai"
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.openai.api_key = "sk-existing"
    save_config(config, config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "provider": "custom",
            "model": "custom/local-model",
            "api_base": "",
            "api_key": "",
        },
    )

    assert save.status_code == 400
    assert save.json()["detail"]["code"] == "provider_configuration_error"

    saved_config = Config.model_validate_json(config_path.read_text())
    assert saved_config.agents.defaults.provider == "openai"
    assert saved_config.agents.defaults.model == "gpt-4o-mini"
    assert saved_config.providers.openai.api_key == "sk-existing"
    assert saved_config.providers.custom.api_base is None


def test_provider_status_treats_oauth_provider_as_configured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "openai_codex"
    config.agents.defaults.model = "openai-codex/gpt-5.1-codex"
    save_config(config, config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    status = client.get("/api/system/provider-status")

    assert status.status_code == 200
    payload = status.json()
    assert payload["active_provider"] == "openai_codex"
    assert payload["active_provider_configured"] is True


def test_provider_save_rejects_codex_base_for_custom_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "openai"
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.openai.api_key = "sk-existing"
    save_config(config, config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "provider": "custom",
            "model": "gpt-5.2",
            "api_base": "https://right.codes/codex/v1",
            "api_key": "sk-test",
        },
    )

    assert save.status_code == 400
    detail = save.json()["detail"]
    assert detail["code"] == "provider_configuration_error"
    assert "Codex endpoint" in detail["message"]

    saved_config = Config.model_validate_json(config_path.read_text())
    assert saved_config.agents.defaults.provider == "openai"
    assert saved_config.providers.custom.api_base is None


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


def test_provider_test_returns_provider_permission_error(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://127.0.0.1:8000/v1"
    config.providers.custom.api_key = "sk-existing"
    save_config(config, config_path)
    set_config_path(config_path)

    class PermissionDeniedProvider:
        async def chat_with_retry(self, **kwargs: Any) -> LLMResponse:
            assert kwargs["messages"][0]["content"] == "测试输入"
            return LLMResponse(
                content="Error: Error code: 403 - {'error': 'API Key 不允许访问该渠道，请前往令牌管理界面修改令牌权限'}",
                finish_reason="error",
            )

    monkeypatch.setattr("roboclaw.http.server.build_provider", lambda _config: PermissionDeniedProvider())

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post(
        "/api/system/provider-test",
        json={
            "provider": "custom",
            "model": "gpt-4o-mini",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "sk-test",
            "input": "测试输入",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["finish_reason"] == "error"
    assert "API Key 不允许访问该渠道" in payload["error"]


def test_provider_test_rejects_codex_base_for_custom_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "gpt-4o-mini"
    config.providers.custom.api_base = "http://127.0.0.1:8000/v1"
    config.providers.custom.api_key = "sk-existing"
    save_config(config, config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post(
        "/api/system/provider-test",
        json={
            "provider": "custom",
            "model": "gpt-5.2",
            "api_base": "https://right.codes/codex/v1",
            "api_key": "sk-test",
            "input": "测试输入",
        },
    )

    assert response.status_code == 400
    assert "Codex endpoint" in response.json()["detail"]["message"]
