"""Provider construction helpers shared by CLI and Web runtime."""

from __future__ import annotations

import importlib
import os
from typing import Any

from roboclaw.config.schema import Config
from roboclaw.providers.base import GenerationSettings, LLMProvider, LLMResponse


class ProviderConfigurationError(RuntimeError):
    """Raised when the configured provider cannot be used."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def is_codex_api_base(api_base: str | None) -> bool:
    """Detect Codex Responses API endpoints (e.g. .../codex/v1)."""
    if not api_base:
        return False
    normalized = api_base.rstrip("/").lower()
    return normalized.endswith("/codex") or "/codex/" in normalized


def build_provider(config: Config) -> LLMProvider:
    """Create the active provider from config or raise ProviderConfigurationError."""
    stub_module = os.environ.get("ROBOCLAW_STUB_LLM")
    if stub_module and os.environ.get("ROBOCLAW_STUB"):
        mod = importlib.import_module(stub_module)
        return mod.create_provider(config)

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    provider_config = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        from roboclaw.providers.openai_codex_provider import OpenAICodexProvider
        provider = OpenAICodexProvider(default_model=model)
    elif provider_name == "custom":
        if not provider_config or not provider_config.api_base:
            raise ProviderConfigurationError(
                "Custom provider requires api_base.",
                "Set the global base URL in the Web Settings page or in providers.custom.api_base.",
            )
        if is_codex_api_base(provider_config.api_base):
            if not provider_config.api_key:
                raise ProviderConfigurationError(
                    "Custom Codex provider requires an API key.",
                    "Paste a token that is allowed to access the selected Codex channel/model.",
                )
            from roboclaw.providers.custom_codex_provider import CustomCodexProvider
            provider = CustomCodexProvider(
                api_key=provider_config.api_key,
                api_base=provider_config.api_base,
                default_model=model,
                extra_headers=provider_config.extra_headers,
            )
        else:
            from roboclaw.providers.custom_provider import CustomProvider
            provider = CustomProvider(
                api_key=provider_config.api_key or "no-key",
                api_base=provider_config.api_base,
                default_model=model,
                extra_headers=provider_config.extra_headers,
            )
    elif provider_name == "azure_openai":
        if not provider_config or not provider_config.api_key or not provider_config.api_base:
            raise ProviderConfigurationError(
                "Azure OpenAI requires api_key and api_base.",
                "Set them in ~/.roboclaw/config.json under providers.azure_openai section.",
            )
        from roboclaw.providers.azure_openai_provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=model,
        )
    else:
        from roboclaw.providers.registry import find_by_name
        spec = find_by_name(provider_name)
        if (
            not model.startswith("bedrock/")
            and not (provider_config and provider_config.api_key)
            and not (spec and (spec.is_oauth or spec.is_local))
        ):
            raise ProviderConfigurationError(
                "No API key configured.",
                "Set one in ~/.roboclaw/config.json under providers section.",
            )
        from roboclaw.providers.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider(
            api_key=provider_config.api_key if provider_config else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=provider_config.extra_headers if provider_config else None,
            provider_name=provider_name,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


class UnconfiguredProvider(LLMProvider):
    """Placeholder when no provider is configured. Replaced on settings save."""

    def __init__(self, message: str = ""):
        super().__init__(api_key=None, api_base=None)
        self._message = message or (
            "No provider configured. Please open Settings and save your API configuration."
        )

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._message, finish_reason="error")

    def get_default_model(self) -> str:
        return "unconfigured"
