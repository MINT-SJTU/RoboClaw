"""HumanoidSpec — reserved for humanoid robot types."""

from __future__ import annotations

from dataclasses import dataclass

from roboclaw.embodied.embodiment.base import EmbodimentSpec


@dataclass(frozen=True)
class HumanoidSpec(EmbodimentSpec):
    """Humanoid robot — reserved for future implementation."""

    pass
