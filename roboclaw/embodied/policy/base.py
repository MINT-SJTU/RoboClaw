"""Base config types for trainable LeRobot policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BasePolicyConfig(ABC):
    """Train-time policy config registered against a LeRobot policy type."""

    policy_type: str = field(init=False, default="")

    @abstractmethod
    def extra_train_args(self) -> list[str]:
        """Return policy-specific ``lerobot-train`` CLI args."""

