"""LLM provider abstraction module.

Keep optional provider implementations lazily imported so a runtime that only
uses one provider does not need every provider-specific dependency installed.
"""

from roboclaw.providers.base import LLMProvider, LLMResponse

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LiteLLMProvider",
    "OpenAICodexProvider",
    "AzureOpenAIProvider",
]


def __getattr__(name: str):
    if name == "LiteLLMProvider":
        from roboclaw.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider
    if name == "OpenAICodexProvider":
        from roboclaw.providers.openai_codex_provider import OpenAICodexProvider

        return OpenAICodexProvider
    if name == "AzureOpenAIProvider":
        from roboclaw.providers.azure_openai_provider import AzureOpenAIProvider

        return AzureOpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
