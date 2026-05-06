"""Diffusion, RL, and auxiliary policy training configs."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import policy_registry


@policy_registry.register
@dataclass(frozen=True)
class DiffusionPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="diffusion")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class TDMPCPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="tdmpc")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class SACPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="sac")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class RewardClassifierPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="reward_classifier")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class SARMPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="sarm")

    def extra_train_args(self) -> list[str]:
        return []


@policy_registry.register
@dataclass(frozen=True)
class WallXPolicyConfig(BasePolicyConfig):
    policy_type: str = field(init=False, default="wall_x")

    def extra_train_args(self) -> list[str]:
        return []

