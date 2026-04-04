"""WheeledSpec — reserved for wheeled robot types."""

from __future__ import annotations

from dataclasses import dataclass

from roboclaw.embodied.embodiment.base import EmbodimentSpec


@dataclass(frozen=True)
class WheeledSpec(EmbodimentSpec):
    """Wheeled robot — reserved for future implementation."""

    pass
