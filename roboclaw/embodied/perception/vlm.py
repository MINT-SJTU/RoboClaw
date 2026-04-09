"""VLM scene understanding via LiteLLM multi-modal chat."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from roboclaw.embodied.perception.camera_grabber import camera_configs, frame_to_base64, grab_frame
from roboclaw.providers.base import LLMResponse


@dataclass
class SceneDescription:
    """Structured result from scene understanding."""
    text: str
    raw_response: str | None = None


class VLM:
    """Lightweight VLM caller using the agent's LiteLLM provider."""

    def __init__(self, provider: Any | None = None):
        if provider is None:
            from roboclaw.config.loader import load_config
            from roboclaw.providers.factory import build_provider

            provider = build_provider(load_config())
        self._provider = provider

    def _default_model(self) -> str:
        default_model = getattr(self._provider, "default_model", None)
        if default_model:
            return default_model
        getter = getattr(self._provider, "get_default_model", None)
        if callable(getter):
            return getter()
        return "anthropic/claude-sonnet-4-5"

    def _has_vision(self, model: str) -> bool:
        """Check if model supports vision by name convention."""
        vision_models = {
            "claude", "gpt-4o", "gpt-4-turbo", "gpt-4v",
            "gemini", "qwen-vl", "qwen2-vl", "qwen2.5-vl",
            "qvq", "internvl", "glm-4v", "minimax",
        }
        return any(tag in model.lower() for tag in vision_models)

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        return await self._provider.chat_with_retry(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _scene_description(response: LLMResponse) -> SceneDescription:
        return SceneDescription(
            text=response.content or "",
            raw_response=str(response),
        )

    @staticmethod
    def _capture_failed_message(camera_alias: str) -> str:
        return (
            f"Camera '{camera_alias}' is configured but frame capture failed. "
            "Check whether another app is using it and, on macOS, whether camera permission is granted."
        )

    async def describe(
        self,
        camera_alias: str,
        question: str = "Describe what you see in this image.",
        model: str | None = None,
    ) -> SceneDescription:
        """Ask a free-text question about a camera frame."""
        configs = camera_configs()
        if camera_alias not in configs:
            return SceneDescription(text=f"No camera found with alias '{camera_alias}'.")

        frame = grab_frame(camera_alias)
        if frame is None:
            return SceneDescription(text=self._capture_failed_message(camera_alias))

        model = model or self._default_model()
        b64_img = frame_to_base64(frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        resp = await self._chat(messages, model=model, max_tokens=1024)
        return self._scene_description(resp)

    async def describe_all(
        self,
        question: str = "Describe the overall scene across all camera views.",
        model: str | None = None,
    ) -> dict[str, SceneDescription]:
        """Ask the same question about every configured camera."""
        configs = camera_configs()
        if not configs:
            return {}
        model = model or self._default_model()
        results = {}
        for alias in configs:
            frame = grab_frame(alias)
            if frame is None:
                results[alias] = SceneDescription(text=self._capture_failed_message(alias))
                continue
            b64_img = frame_to_base64(frame)
            messages = [
                {
                    "role": "user",
                "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            resp = await self._chat(messages, model=model, max_tokens=512)
            results[alias] = self._scene_description(resp)
        return results

    async def detect_objects(
        self,
        camera_alias: str,
        target: str,
        model: str | None = None,
    ) -> SceneDescription:
        """Detect whether a specific object is visible and where it is."""
        question = (
            f"Is the object '{target}' visible in this image? "
            f"If yes, describe its approximate location (left/center/right, "
            f"top/middle/bottom) and whether it appears graspable. "
            f"If no, say 'not visible'."
        )
        return await self.describe(camera_alias, question, model=model)

    async def grasp_check(
        self,
        camera_alias: str,
        object_name: str,
        model: str | None = None,
    ) -> SceneDescription:
        """Check if an object is in a good state for grasping."""
        question = (
            f"Is the object '{object_name}' currently visible and in a stable position "
            f"for a robot gripper to grasp? Is it obstructed or tangled with other objects? "
            f"Answer briefly: OK / not visible / obstructed / unstable."
        )
        return await self.describe(camera_alias, question, model=model)

    async def what_changed(
        self,
        camera_alias: str,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        model: str | None = None,
    ) -> SceneDescription:
        """Compare two frames and describe what changed."""
        model = model or self._default_model()
        before_b64 = frame_to_base64(before_frame)
        after_b64 = frame_to_base64(after_frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (before action):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{before_b64}"}},
                    {"type": "text", "text": "Image 2 (after action):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{after_b64}"}},
                    {"type": "text", "text": "Describe what changed between these two images."},
                ],
            }
        ]
        resp = await self._chat(messages, model=model, max_tokens=512)
        return self._scene_description(resp)
