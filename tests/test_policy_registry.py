from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from roboclaw.embodied.policy import BasePolicyConfig, PolicyRegistry, policy_registry


def test_policy_registry_registers_custom_config() -> None:
    registry = PolicyRegistry()

    @registry.register
    @dataclass(frozen=True)
    class ExamplePolicyConfig(BasePolicyConfig):
        policy_type: str = field(init=False, default="example")

        def extra_train_args(self) -> list[str]:
            return ["--policy.example=true"]

    config = registry.get("example")

    assert isinstance(config, ExamplePolicyConfig)
    assert config.extra_train_args() == ["--policy.example=true"]
    assert registry.supported_types() == {"example"}


def test_policy_registry_returns_registered_builtin_policy() -> None:
    config = policy_registry.get("act")

    assert config.policy_type == "act"
    assert config.extra_train_args() == []
    assert "groot" in policy_registry.supported_types()


def test_policy_registry_raises_for_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported policy_type 'unknown'"):
        policy_registry.get("unknown")
