"""Registry for trainable LeRobot policy configs."""

from __future__ import annotations

from dataclasses import is_dataclass

from roboclaw.embodied.policy.base import BasePolicyConfig


class PolicyRegistry:
    """Register and resolve train-time policy configs by policy type."""

    def __init__(self) -> None:
        self._config_types: dict[str, type[BasePolicyConfig]] = {}

    def register(self, config_cls: type[BasePolicyConfig]) -> type[BasePolicyConfig]:
        """Register a config dataclass and return it for decorator usage."""
        if not issubclass(config_cls, BasePolicyConfig):
            raise TypeError("Policy config must inherit from BasePolicyConfig.")
        if not is_dataclass(config_cls):
            raise TypeError("Policy config must be a dataclass.")

        config = config_cls()
        policy_type = config.policy_type.strip()
        if not policy_type:
            raise ValueError("Policy config must declare a non-empty policy_type.")
        if policy_type in self._config_types:
            raise ValueError(f"Policy '{policy_type}' is already registered.")

        self._config_types[policy_type] = config_cls
        return config_cls

    def get(self, policy_type: str) -> BasePolicyConfig:
        """Instantiate the config registered for ``policy_type``."""
        config_cls = self._config_types.get(policy_type)
        if config_cls is None:
            allowed = ", ".join(sorted(self.supported_types()))
            raise ValueError(
                f"Unsupported policy_type '{policy_type}'. Expected one of: {allowed}."
            )
        return config_cls()

    def supported_types(self) -> set[str]:
        """Return a copy of all registered policy types."""
        return set(self._config_types)


policy_registry = PolicyRegistry()

