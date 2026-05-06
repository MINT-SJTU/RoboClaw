"""GR00T-family policy training configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import policy_registry


@policy_registry.register
@dataclass(frozen=True)
class GR00TPolicyConfig(BasePolicyConfig):
    """NVIDIA GR00T N1."""

    policy_type: str = field(init=False, default="groot")

    def extra_train_args(self) -> list[str]:
        return []

