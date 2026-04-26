"""Tests for provider factory helpers used by the Web settings flow."""

from __future__ import annotations

import pytest

from roboclaw.config.loader import load_config
from roboclaw.config.schema import Config
from roboclaw.providers.custom_provider import CustomProvider
from roboclaw.providers.factory import ProviderConfigurationError, UnconfiguredProvider, build_provider


def test_build_provider_requires_configuration() -> None:
    config = Config()

    with pytest.raises(ProviderConfigurationError):
        build_provider(config)


def test_build_provider_supports_custom_api_base() -> None:
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "custom/local-model"
    config.providers.custom.api_base = "http://127.0.0.1:8000/v1"

    provider = build_provider(config)

    assert isinstance(provider, CustomProvider)


def test_build_provider_custom_requires_api_base() -> None:
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "custom/local-model"

    with pytest.raises(ProviderConfigurationError):
        build_provider(config)


def test_build_provider_rejects_codex_base_for_custom_provider() -> None:
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "gpt-5.2"
    config.providers.custom.api_base = "https://right.codes/codex/v1"
    config.providers.custom.api_key = "sk-test"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        build_provider(config)

    assert "Codex endpoint" in str(exc_info.value)
    assert "OpenAI Codex" in exc_info.value.hint


def test_load_config_migrates_custom_codex_base_to_openai_codex(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "agents": {"defaults": {"provider": "custom", "model": "gpt-5.2"}},
          "providers": {"custom": {"apiBase": "https://right.codes/codex/v1", "apiKey": "sk-test"}}
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.agents.defaults.provider == "openai_codex"
    assert config.agents.defaults.model == "openai-codex/gpt-5.1-codex"


@pytest.mark.asyncio
async def test_unconfigured_provider_returns_helpful_error() -> None:
    provider = UnconfiguredProvider()
    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.finish_reason == "error"
    assert response.content is not None
    assert "No provider configured" in response.content
