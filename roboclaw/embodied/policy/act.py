"""ACT-family policy training configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import policy_registry


@policy_registry.register
@dataclass(frozen=True)
class ActPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="act")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class MultiTaskDiTPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="multi_task_dit")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class VQBeTPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="vqbet")

    def extra_train_args(self) -> list[str]:
        return []

