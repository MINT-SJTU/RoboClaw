"""VLA policy training configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import policy_registry


@policy_registry.register
@dataclass(frozen=True)
class SmolVLAConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="smolvla")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class XVLAPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="xvla")

    def extra_train_args(self) -> list[str]:
        return []

