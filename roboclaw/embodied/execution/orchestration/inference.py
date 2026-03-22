"""Thin inference orchestration for embodied policy runs."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from roboclaw.embodied.execution.orchestration.runtime.model import RuntimeStatus

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class InferenceSession:
    checkpoint_path: str
    setup_id: str
    running: bool = False
    steps_completed: int = 0
    stop_event: asyncio.Event | None = None
    task: asyncio.Task[Any] | None = None


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    steps_completed: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)


_INFERENCE_SESSIONS: dict[str, InferenceSession] = {}


class _ZeroPolicy:
    def predict(self, values: dict[str, Any]) -> dict[str, float]:
        return {key: 0.0 for key, value in values.items() if isinstance(value, (int, float))}


def _load_policy(checkpoint_path: str) -> Any:
    torch = importlib.import_module("torch")
    from roboclaw.embodied.learning.policies.act import ACTConfig, ACTPolicy
    model = ACTPolicy(ACTConfig())
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


async def start_inference(
    session: InferenceSession,
    executor: Any,
    context: Any,
    on_progress: ProgressCallback | None = None,
) -> InferenceResult:
    if context.runtime.status != RuntimeStatus.READY:
        connected = await executor.execute_connect(context, on_progress=on_progress)
        if not connected.ok:
            return InferenceResult(False, session.steps_completed, connected.message, dict(connected.details))
    adapter = executor._adapter(context)
    try:
        policy = _load_policy(session.checkpoint_path)
    except Exception as exc:
        return InferenceResult(False, 0, f"Failed to load policy: {exc}", {"error": str(exc)})
    session.running = True
    session.stop_event = session.stop_event or asyncio.Event()
    context.runtime.status = RuntimeStatus.BUSY
    context.runtime.last_error = None
    try:
        while not session.stop_event.is_set():
            state = await adapter.get_state()
            primitive = await adapter.execute_primitive("policy_step", policy.predict(state.values))
            if not primitive.accepted:
                context.runtime.last_error = primitive.message or primitive.error_code
                return InferenceResult(False, session.steps_completed, primitive.message or "Inference failed.")
            session.steps_completed += 1
            if on_progress is not None and session.steps_completed % 10 == 0:
                await on_progress(f"Policy running on setup `{session.setup_id}` ({session.steps_completed} steps).")
            await asyncio.sleep(0.02)
    except Exception as exc:
        context.runtime.last_error = str(exc)
        return InferenceResult(False, session.steps_completed, "Inference failed.", {"error": str(exc)})
    finally:
        session.running = False
        context.runtime.status = RuntimeStatus.READY
    return InferenceResult(True, session.steps_completed, f"Policy stopped after {session.steps_completed} steps.")


def stop_inference(session: InferenceSession) -> InferenceResult:
    if session.stop_event is not None:
        session.stop_event.set()
    return InferenceResult(True, session.steps_completed, f"Policy stopped after {session.steps_completed} steps.")
