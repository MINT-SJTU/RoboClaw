"""Interaction protocol — dataclasses that describe how a session wants I/O.

Each session exposes ``interaction_spec()`` returning one of these specs.
The TtySession (or any future UI driver) reads the spec and drives I/O accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PassthroughSpec:
    """Subprocess directly owns the TTY (e.g. calibration, replay)."""

    argv: list[str]
    label: str


@dataclass(frozen=True)
class PollingSpec:
    """Real-time key capture + status-line display (teleop, record)."""

    label: str
    poll_interval_s: float = 0.05


@dataclass(frozen=True)
class PromptStep:
    """A single question in a multi-step prompting flow."""

    prompt_id: str
    message: str
    options: list[str] | None = None


@dataclass(frozen=True)
class PollStep:
    """A polling-wait step (e.g. motion detection) in a prompting flow."""

    prompt_id: str
    message: str
    poll_fn: Callable[[], str | None]
    timeout_s: float = 30.0
    timeout_message: str = "  Timeout -- no event detected."
    retry_prompt: str = "  Retry? (y/n): "


@dataclass(frozen=True)
class PromptingSpec:
    """Multi-step Q&A flow (setup identify)."""

    label: str
