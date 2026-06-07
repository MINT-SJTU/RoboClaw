"""Tests for the Web provider settings API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from roboclaw.config.loader import save_config, set_config_path
from roboclaw.config.schema import Config
from roboclaw.http.runtime import _provider_default_model
from roboclaw.http.server import create_app
from roboclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest


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


def test_provider_save_custom_does_not_keep_codex_model_when_discovery_fails(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "openai_codex"
    config.agents.defaults.model = "openai-codex/gpt-5.5"
    save_config(config, config_path)
    set_config_path(config_path)

    async def _fake_discover(api_base: str, api_key: str | None) -> str | None:
        return None

    monkeypatch.setattr("roboclaw.http.server._discover_custom_model", _fake_discover)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "provider": "custom",
            "api_base": "https://gateway.example/v1",
            "api_key": "sk-test",
        },
    )

    assert save.status_code == 200
    saved = save.json()
    assert saved["default_provider"] == "custom"
    assert saved["default_model"] == "claude-sonnet-4-5-20250929"


def test_provider_save_rejects_shell_text_as_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={
            "provider": "custom",
            "api_base": "https://gateway.example/v1",
            "api_key": "rm -rf /tmp/foo > /tmp/log 2>&1",
        },
    )

    assert save.status_code == 200
    saved = save.json()
    assert saved["status"] == "error"
    assert "API key" in saved["message"]


def test_provider_status_marks_suspicious_saved_key_unconfigured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.providers.custom.api_base = "https://gateway.example/v1"
    config.providers.custom.api_key = "rm -rf /tmp/foo > /tmp/log 2>&1"
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "claude-test"
    save_config(config, config_path)
    set_config_path(config_path)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    status = client.get("/api/system/provider-status")

    assert status.status_code == 200
    payload = status.json()
    assert payload["custom_provider"]["configured"] is False
    assert payload["custom_provider"]["masked_api_key"] == "疑似误填，需重新保存"
    assert payload["custom_provider"]["api_key_warning"]


def test_provider_save_openai_codex_uses_oauth_default_model(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    set_config_path(config_path)

    monkeypatch.setattr(
        "roboclaw.http.server._is_oauth_provider_configured",
        lambda provider_name: provider_name == "openai_codex",
    )

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    save = client.post(
        "/api/system/provider-config",
        json={"provider": "openai_codex"},
    )

    assert save.status_code == 200
    saved = save.json()
    assert saved["status"] == "ok"
    assert saved["default_provider"] == "openai_codex"
    assert saved["active_provider"] == "openai_codex"
    assert saved["default_model"] == "openai-codex/gpt-5.5"
    assert saved["active_provider_configured"] is True
    codex = next(item for item in saved["providers"] if item["name"] == "openai_codex")
    assert codex["oauth"] is True
    assert codex["configured"] is True


def test_provider_default_model_uses_provider_normalized_model() -> None:
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.3-codex"

    class Provider:
        def get_default_model(self) -> str:
            return "openai-codex/gpt-5.5"

    assert _provider_default_model(Provider(), config) == "openai-codex/gpt-5.5"


def test_provider_test_reports_agent_capability(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "claude-test"
    config.providers.custom.api_base = "https://gateway.example/v1"
    save_config(config, config_path)
    set_config_path(config_path)

    class ToolProvider(LLMProvider):
        async def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            if tools:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCallRequest(id="abc123xyz", name="report_provider_status", arguments={"ok": True})],
                    finish_reason="tool_calls",
                )
            return LLMResponse(content="OK")

        def get_default_model(self) -> str:
            return "claude-test"

    monkeypatch.setattr("roboclaw.http.server.build_provider", lambda config: ToolProvider())

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post("/api/system/provider-test", json={"provider": "custom", "model": "claude-test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["capability"] == "agent"
    assert payload["text"]["ok"] is True
    assert payload["tools"]["ok"] is True


def test_provider_test_auto_discovers_model_when_model_is_blank(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "openai-codex/gpt-5.5"
    config.providers.custom.api_base = "https://gateway.example/v1"
    config.providers.custom.api_key = "sk-test"
    save_config(config, config_path)
    set_config_path(config_path)

    async def _fake_discover(api_base: str, api_key: str | None) -> str | None:
        assert api_base == "https://gateway.example/v1"
        assert api_key == "sk-test"
        return "claude-auto"

    class TextProvider(LLMProvider):
        async def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            return LLMResponse(content="OK")

        def get_default_model(self) -> str:
            return "claude-auto"

    def _fake_build_provider(config: Config) -> LLMProvider:
        assert config.agents.defaults.model == "claude-auto"
        return TextProvider()

    monkeypatch.setattr("roboclaw.http.server._discover_custom_model", _fake_discover)
    monkeypatch.setattr("roboclaw.http.server.build_provider", _fake_build_provider)

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post(
        "/api/system/provider-test",
        json={"provider": "custom", "model": "", "api_base": "https://gateway.example/v1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "claude-auto"


def test_provider_test_classifies_insufficient_balance(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "claude-test"
    config.providers.custom.api_base = "https://gateway.example/v1"
    save_config(config, config_path)
    set_config_path(config_path)

    class BalanceProvider(LLMProvider):
        async def chat(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            return LLMResponse(
                content="Error: Error code: 403 - {'error': {'message': 'Insufficient balance'}}",
                finish_reason="error",
            )

        def get_default_model(self) -> str:
            return "claude-test"

    monkeypatch.setattr("roboclaw.http.server.build_provider", lambda config: BalanceProvider())

    app = create_app(config_path=str(config_path), workspace=str(tmp_path / "workspace"))
    client = TestClient(app)

    response = client.post("/api/system/provider-test", json={"provider": "custom", "model": "claude-test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["capability"] == "unavailable"
    assert payload["text"]["errorCode"] == "insufficient_balance"
    assert "AI provider" in payload["recommendation"]
