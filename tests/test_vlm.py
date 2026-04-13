"""Tests for VLM perception helpers."""

from __future__ import annotations

import numpy as np
import pytest

from roboclaw.embodied.perception.vlm import VLM
from roboclaw.providers.base import LLMResponse


class _FakeProvider:
    def __init__(self) -> None:
        self.default_model = "openai-codex/gpt-5.3-codex"
        self.calls: list[dict] = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(content="scene ok")


def test_vlm_builds_provider_from_current_config(monkeypatch) -> None:
    fake_provider = object()
    fake_config = object()

    monkeypatch.setattr("roboclaw.config.loader.load_config", lambda: fake_config)
    monkeypatch.setattr("roboclaw.providers.factory.build_provider", lambda config: fake_provider)

    vlm = VLM()

    assert vlm._provider is fake_provider


@pytest.mark.asyncio
async def test_vlm_describe_uses_async_provider(monkeypatch) -> None:
    provider = _FakeProvider()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "roboclaw.embodied.perception.vlm.camera_configs",
        lambda: {"wrist": {"port": "0"}},
    )
    monkeypatch.setattr("roboclaw.embodied.perception.vlm.grab_frame", lambda alias: frame)
    monkeypatch.setattr("roboclaw.embodied.perception.vlm.frame_to_base64", lambda _: "abc123")

    result = await VLM(provider=provider).describe("wrist", question="Describe the scene")

    assert result.text == "scene ok"
    assert provider.calls[0]["model"] == "openai-codex/gpt-5.3-codex"
    assert provider.calls[0]["max_tokens"] == 1024
    content = provider.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["text"] == "Describe the scene"


@pytest.mark.asyncio
async def test_vlm_describe_reports_capture_failure_for_configured_camera(monkeypatch) -> None:
    provider = _FakeProvider()

    monkeypatch.setattr(
        "roboclaw.embodied.perception.vlm.camera_configs",
        lambda: {"overhead": {"port": "2"}},
    )
    monkeypatch.setattr("roboclaw.embodied.perception.vlm.grab_frame", lambda alias: None)

    result = await VLM(provider=provider).describe("overhead")

    assert "frame capture failed" in result.text
    assert provider.calls == []


@pytest.mark.asyncio
async def test_vlm_describe_all_reports_capture_failure_per_camera(monkeypatch) -> None:
    provider = _FakeProvider()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "roboclaw.embodied.perception.vlm.camera_configs",
        lambda: {"overhead": {"port": "2"}, "wrist": {"port": "0"}},
    )
    monkeypatch.setattr(
        "roboclaw.embodied.perception.vlm.grab_frame",
        lambda alias: frame if alias == "wrist" else None,
    )
    monkeypatch.setattr("roboclaw.embodied.perception.vlm.frame_to_base64", lambda _: "abc123")

    results = await VLM(provider=provider).describe_all()

    assert "frame capture failed" in results["overhead"].text
    assert results["wrist"].text == "scene ok"
