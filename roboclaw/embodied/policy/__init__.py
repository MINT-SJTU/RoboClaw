"""Policy config registry for embodied training."""

from roboclaw.embodied.policy.base import BasePolicyConfig
from roboclaw.embodied.policy.registry import PolicyRegistry, policy_registry

from . import act as _act
from . import diffusion as _diffusion
from . import gr00t as _gr00t
from . import pi0 as _pi0
from . import smolvla as _smolvla

__all__ = [
    "BasePolicyConfig",
    "PolicyRegistry",
    "policy_registry",
]

