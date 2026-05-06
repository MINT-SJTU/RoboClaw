"""PI-family policy training configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import policy_registry


@policy_registry.register
@dataclass(frozen=True)
class Pi0PolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="pi0")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class Pi0FastPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="pi0_fast")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class Pi05PolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="pi05")

    def extra_train_args(self) -> list[str]:
        return []

