"""Tests for custom Codex Responses API gateways."""

from __future__ import annotations

from typing import Any

import pytest

from roboclaw.providers.custom_codex_provider import CustomCodexProvider, _responses_url
from roboclaw.providers.openai_codex_provider import _friendly_error


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://right.codes/codex/v1", "https://right.codes/codex/v1/responses"),
        ("https://right.codes/codex/v1/", "https://right.codes/codex/v1/responses"),
        ("https://right.codes/codex/v1/responses", "https://right.codes/codex/v1/responses"),
    ],
)
def test_responses_url(api_base: str, expected: str) -> None:
    assert _responses_url(api_base) == expected


@pytest.mark.asyncio
async def test_custom_codex_provider_uses_responses_api(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_request(url: str, headers: dict[str, str], body: dict[str, Any], verify: bool):
        seen.update(url=url, headers=headers, body=body, verify=verify)
        return "OK", [], "stop"

    monkeypatch.setattr("roboclaw.providers.custom_codex_provider._request_codex", _fake_request)

    provider = CustomCodexProvider(
        api_key="sk-test",
        api_base="https://right.codes/codex/v1/",
        default_model="openai-codex/gpt-5.3-codex",
        extra_headers={"APP-Code": "ROBOCLAW"},
    )

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert response.content == "OK"
    assert seen["url"] == "https://right.codes/codex/v1/responses"
    assert seen["body"]["model"] == "gpt-5.3-codex"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["headers"]["OpenAI-Beta"] == "responses=experimental"
    assert seen["headers"]["APP-Code"] == "ROBOCLAW"


def test_codex_permission_error_is_actionable() -> None:
    message = _friendly_error(
        403,
        '{"error":"API Key \\u4e0d\\u5141\\u8bb8\\u8bbf\\u95ee\\u8be5\\u6e20\\u9053"}',
    )

    assert "Provider refused the request (403)" in message
    assert "API Key" in message
    assert "selected Codex channel or model" in message
