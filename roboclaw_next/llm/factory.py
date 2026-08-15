"""Provider factory for RoboClaw Next."""

from __future__ import annotations

import os

from roboclaw_next.llm.openai_compatible import LLMProvider, OpenAICompatibleProvider
from roboclaw_next.llm.types import GenerationConfig, ProviderConfig, ProviderName


DEFAULT_MODELS: dict[ProviderName, str] = {
    "openai": "gpt-4.1",
    "deepseek": "deepseek-chat",
}


DEFAULT_BASE_URLS: dict[ProviderName, str | None] = {
    "openai": None,
    "deepseek": "https://api.deepseek.com",
}


API_KEY_ENV: dict[ProviderName, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


BASE_URL_ENV: dict[ProviderName, str] = {
    "openai": "OPENAI_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
}


def create_llm_provider(
    provider: ProviderName,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> LLMProvider:
    """Create an OpenAI or DeepSeek provider.

    Values passed explicitly win; otherwise environment variables and provider
    defaults are used.
    """

    resolved_api_key = api_key or os.environ.get(API_KEY_ENV[provider], "")
    if not resolved_api_key:
        raise ValueError(f"{provider} provider requires {API_KEY_ENV[provider]}.")

    config = ProviderConfig(
        provider=provider,
        api_key=resolved_api_key,
        model=model or os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider],
        base_url=base_url or os.environ.get(BASE_URL_ENV[provider]) or DEFAULT_BASE_URLS[provider],
        generation=GenerationConfig(temperature=temperature, max_tokens=max_tokens),
    )
    return OpenAICompatibleProvider(config)
