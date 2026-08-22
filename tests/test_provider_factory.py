"""Tests for provider factory helpers used by the Web settings flow."""

from __future__ import annotations

import asyncio

import pytest

from roboclaw.config.schema import Config
from roboclaw.providers.custom_provider import CustomProvider
from roboclaw.providers.factory import (
    ProviderConfigurationError,
    UnconfiguredProvider,
    build_provider,
)


def test_build_provider_requires_configuration() -> None:
    config = Config()

    with pytest.raises(ProviderConfigurationError):
        build_provider(config)


def test_build_provider_supports_custom_api_base() -> None:
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "custom/local-model"
    config.providers.custom.api_base = "http://127.0.0.1:8000/v1"
    config.providers.custom.extra_headers = {"APP-Code": "abc"}

    provider = build_provider(config)

    assert isinstance(provider, CustomProvider)
    assert provider.extra_headers == {"APP-Code": "abc"}


def test_build_provider_custom_requires_api_base() -> None:
    config = Config()
    config.agents.defaults.provider = "custom"
    config.agents.defaults.model = "custom/local-model"

    with pytest.raises(ProviderConfigurationError):
        build_provider(config)


@pytest.mark.asyncio
async def test_unconfigured_provider_returns_helpful_error() -> None:
    provider = UnconfiguredProvider()
    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.finish_reason == "error"
    assert response.content is not None
    assert "No provider configured" in response.content


def test_custom_provider_uses_timeout_and_extra_headers(monkeypatch) -> None:
    captured = {}
    http_client_kwargs = {}

    class FakeDefaultAsyncHttpxClient:
        def __init__(self, **kwargs):
            http_client_kwargs.update(kwargs)

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("roboclaw.providers.custom_provider.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("roboclaw.providers.custom_provider.DefaultAsyncHttpxClient", FakeDefaultAsyncHttpxClient)

    CustomProvider(
        api_key="key",
        api_base="https://example.test/v1",
        default_model="model",
        extra_headers={"APP-Code": "abc"},
        timeout=12.5,
    )

    assert captured["api_key"] == "key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["timeout"] == 12.5
    assert captured["default_headers"]["APP-Code"] == "abc"
    assert captured["default_headers"]["x-session-affinity"]
    assert captured["http_client"] is not None
    assert http_client_kwargs == {"timeout": 12.5, "trust_env": False}


def test_custom_provider_ignores_invalid_env_proxy(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:10808")

    provider = CustomProvider(
        api_key="key",
        api_base="https://example.test/v1",
        default_model="model",
    )

    assert provider.get_default_model() == "model"
    asyncio.run(provider._client.close())


def test_custom_provider_builds_compatible_retry_without_reasoning_effort() -> None:
    provider = CustomProvider(default_model="model")
    kwargs = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    }

    retry = provider._compatible_retry_kwargs(
        kwargs,
        Exception("unsupported parameter: reasoning_effort"),
    )

    assert retry is not None
    assert "reasoning_effort" not in retry
    assert kwargs["reasoning_effort"] == "high"
