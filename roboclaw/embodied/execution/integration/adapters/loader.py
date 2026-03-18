"""Adapter implementation loading helpers."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.embodied.definition.components.robots.model import RobotManifest
from roboclaw.embodied.definition.systems.assemblies.model import AssemblyManifest
from roboclaw.embodied.definition.systems.deployments.model import DeploymentProfile
from roboclaw.embodied.execution.integration.adapters.model import AdapterBinding
from roboclaw.embodied.execution.integration.adapters.protocols import EmbodiedAdapter
from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget


class AdapterLoader:
    """Instantiate one adapter implementation from an entrypoint string."""

    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def load(
        self,
        *,
        binding: AdapterBinding,
        assembly: AssemblyManifest,
        deployment: DeploymentProfile,
        target: ExecutionTarget,
        robot: RobotManifest,
        profile: Any,
    ) -> EmbodiedAdapter:
        module_name, _, attr_name = binding.implementation.partition(":")
        if not module_name or not attr_name:
            raise ValueError(
                f"Adapter '{binding.id}' implementation must use 'module:attribute' format."
            )

        module = import_module(module_name)
        adapter_cls = getattr(module, attr_name)
        kwargs = {
            "binding": binding,
            "assembly": assembly,
            "deployment": deployment,
            "target": target,
            "robot": robot,
            "tools": self.tools,
            "profile": profile,
        }
        signature = inspect.signature(adapter_cls)
        accepted = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        }
        return adapter_cls(**accepted)
