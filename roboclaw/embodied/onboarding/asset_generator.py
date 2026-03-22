"""Asset rendering and writing for onboarding-generated workspace files."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.embodied.execution.integration.control_surfaces import ARM_HAND_CONTROL_SURFACE_PROFILE
from roboclaw.embodied.onboarding.helpers import (
    generated_at,
    launch_command,
    mount_frame,
    profile_id,
    resolved_ros2_distro,
    ros2_namespace,
    sensor_topic,
)
from roboclaw.embodied.onboarding.model import SetupOnboardingState


class AssetGenerator:
    """Generate onboarding workspace assets through the tool registry."""

    def __init__(self, workspace: Path, tool_registry: ToolRegistry):
        self.workspace = workspace
        self.tool_registry = tool_registry

    async def write_intake(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        intake_path = self.workspace / "embodied" / "intake" / f"{state.intake_slug}.md"
        content = self.render_intake(state)
        await self.run_tool("write_file", {"path": str(intake_path), "content": content}, on_progress=on_progress)
        assets = dict(state.generated_assets)
        assets["intake"] = str(intake_path)
        return replace(state, generated_assets=assets)

    async def write_assembly(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "assemblies" / f"{state.assembly_id}.py"
        await self.run_tool(
            "write_file",
            {"path": str(path), "content": self.render_assembly(state)},
            on_progress=on_progress,
        )
        assets = dict(state.generated_assets)
        assets["assembly"] = str(path)
        return replace(state, generated_assets=assets)

    async def write_deployment(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "deployments" / f"{state.deployment_id}.py"
        await self.run_tool(
            "write_file",
            {"path": str(path), "content": self.render_deployment(state)},
            on_progress=on_progress,
        )
        assets = dict(state.generated_assets)
        assets["deployment"] = str(path)
        return replace(state, generated_assets=assets)

    async def write_adapter(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "adapters" / f"{state.adapter_id}.py"
        await self.run_tool(
            "write_file",
            {"path": str(path), "content": self.render_adapter(state)},
            on_progress=on_progress,
        )
        assets = dict(state.generated_assets)
        assets["adapter"] = str(path)
        return replace(state, generated_assets=assets)

    async def run_tool(
        self,
        name: str,
        params: dict[str, Any],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        if on_progress is not None:
            await on_progress(self._format_tool_hint(name, params), tool_hint=True)
        logger.info("Onboarding tool call: {}({})", name, json.dumps(params, ensure_ascii=False)[:200])
        result = await self.tool_registry.execute(name, params)
        if on_progress is not None:
            summary = self._tool_result_summary(name, params, result)
            if summary:
                await on_progress(summary)
        return result

    @staticmethod
    def _format_tool_hint(name: str, params: dict[str, Any]) -> str:
        if name in {"read_file", "write_file", "list_dir"} and isinstance(params.get("path"), str):
            return f'{name}("{params["path"]}")'
        if name == "exec" and isinstance(params.get("command"), str):
            command = params["command"]
            command = command[:60] + "..." if len(command) > 60 else command
            return f'exec("{command}")'
        return name

    @staticmethod
    def _tool_result_summary(name: str, params: dict[str, Any], result: str) -> str | None:
        if result.startswith("Error"):
            return result
        if name == "write_file":
            return f"Updated {Path(str(params['path'])).name}"
        if name == "read_file":
            return f"Read {Path(str(params['path'])).name}"
        if name == "exec" and "serial" in result:
            return "Completed local device probing"
        if name == "exec":
            return "Completed local environment probing"
        return None

    def render_intake(self, state: SetupOnboardingState) -> str:
        robot_lines = "\n".join(
            f"- `{item['attachment_id']}`: `{item['robot_id']}` ({item['role']})" for item in state.robot_attachments
        ) or "- pending"
        sensor_lines = "\n".join(
            f"- `{item['attachment_id']}`: `{item['sensor_id']}` mounted as `{item['mount']}`"
            for item in state.sensor_attachments
        ) or "- none yet"
        facts = state.detected_facts
        fact_lines = [
            f"- connected: `{facts.get('connected', 'unknown')}`",
            f"- serial_device_by_id: `{facts.get('serial_device_by_id', 'unknown')}`",
            f"- serial_device_unstable: `{facts.get('serial_device_unstable', 'unknown')}`",
            f"- serial_device_unresponsive: `{facts.get('serial_device_unresponsive', 'unknown')}`",
            f"- serial_probe_error: `{facts.get('serial_probe_error', 'unknown')}`",
            f"- calibration_path: `{facts.get('calibration_path', 'unknown')}`",
            f"- calibration_missing: `{facts.get('calibration_missing', 'unknown')}`",
            f"- ros2_available: `{facts.get('ros2_available', 'unknown')}`",
            f"- ros2_distro: `{facts.get('ros2_distro', 'unknown')}`",
            f"- host_pretty_name: `{facts.get('host_pretty_name', 'unknown')}`",
            f"- host_shell: `{facts.get('host_shell', 'unknown')}`",
            f"- host_passwordless_sudo: `{facts.get('host_passwordless_sudo', 'unknown')}`",
            f"- ros2_install_profile: `{facts.get('ros2_install_profile', 'unknown')}`",
            f"- ros2_install_recipe: `{facts.get('ros2_install_recipe', 'unknown')}`",
            f"- ros2_install_step_index: `{facts.get('ros2_install_step_index', 'unknown')}`",
        ]
        generated = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(state.generated_assets.items())) or "- none yet"
        notes_lines = [f"- {note}" for note in state.notes] or ["- none"]
        return "\n".join(
            [
                f"# {state.setup_id}",
                "",
                "## Setup Scope",
                robot_lines,
                "",
                "## Sensors",
                sensor_lines,
                "",
                "## Deployment Facts",
                *fact_lines,
                "",
                "## Generated Assets",
                generated,
                "",
                "## Notes",
                *notes_lines,
                "",
            ]
        )

    def render_assembly(self, state: SetupOnboardingState) -> str:
        is_sim = state.detected_facts.get("simulation_requested") is True
        robot_blocks = "\n".join(
            [
                "\n".join(
                    [
                        "        RobotAttachment(",
                        f"            attachment_id={item['attachment_id']!r},",
                        f"            robot_id={item['robot_id']!r},",
                        f"            config=RobotConfig(instance_id={item['attachment_id']!r}, base_frame='base_link', tool_frame='tool0'),",
                        "        ),",
                    ]
                )
                for item in state.robot_attachments
            ]
        )
        sensor_blocks = "\n".join(
            [
                "\n".join(
                    [
                        "        SensorAttachment(",
                        f"            attachment_id={item['attachment_id']!r},",
                        f"            sensor_id={item['sensor_id']!r},",
                        f"            mount={item['mount']!r},",
                        f"            mount_frame={mount_frame(item['mount'])!r},",
                        "            mount_transform=Transform3D(),",
                        "        ),",
                    ]
                )
                for item in state.sensor_attachments
            ]
        )
        if not sensor_blocks:
            sensor_blocks = ""
        target_imports = (
            "from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget\n"
            "from roboclaw.embodied.definition.foundation.schema import CarrierKind, SimulatorKind, TransportKind"
            if is_sim
            else
            "from roboclaw.embodied.execution.integration.carriers.real import build_real_ros2_target\n"
            "from roboclaw.embodied.execution.integration.transports.ros2 import build_standard_ros2_contract"
        )
        target_block = (
            f"SIM_TARGET = ExecutionTarget(\n"
            "    id='sim',\n"
            "    carrier=CarrierKind.SIM,\n"
            "    transport=TransportKind.DIRECT,\n"
            f"    description={'Simulation target for ' + state.setup_id!r},\n"
            "    simulator=SimulatorKind.MUJOCO,\n"
            ")\n"
            if is_sim
            else
            f"REAL_TARGET = build_real_ros2_target(\n"
            "    target_id='real',\n"
            f"    description={'Real target for ' + state.setup_id!r},\n"
            f"    ros2=build_standard_ros2_contract({state.assembly_id!r}, 'real'),\n"
            ")\n"
        )
        target_name = "SIM_TARGET" if is_sim else "REAL_TARGET"
        target_id = "sim" if is_sim else "real"
        return "\n".join(
            [
                '"""Workspace-generated embodied assembly."""',
                "",
                "from roboclaw.embodied.definition.components.robots import RobotConfig",
                "from roboclaw.embodied.definition.systems.assemblies import (",
                "    AssemblyBlueprint,",
                "    FrameTransform,",
                "    RobotAttachment,",
                "    SensorAttachment,",
                "    Transform3D,",
                ")",
                *target_imports.splitlines(),
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                "    WorkspaceProvenance,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.ASSEMBLY,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.ASSEMBLY,",
                "    provenance=WorkspaceProvenance(",
                '        source="workspace_generated",',
                '        generator="onboarding_controller",',
                f"        generated_by={state.setup_id!r},",
                f"        generated_at={generated_at()!r},",
                "    ),",
                ")",
                "",
                *target_block.splitlines(),
                "ASSEMBLY = AssemblyBlueprint(",
                f"    id={state.assembly_id!r},",
                f"    name={f'{state.setup_id} assembly'!r},",
                f"    description={f'Workspace setup for {state.setup_id}.'!r},",
                "    robots=(",
                robot_blocks,
                "    ),",
                "    sensors=(",
                sensor_blocks,
                "    ),",
                f"    execution_targets=({target_name},),",
                f"    default_execution_target_id={target_id!r},",
                "    frame_transforms=(",
                "        FrameTransform(parent_frame='world', child_frame='base_link', transform=Transform3D()),",
                "        FrameTransform(parent_frame='base_link', child_frame='tool0', transform=Transform3D()),",
                "    ),",
                "    tools=(),",
                "    notes=('Generated by assembly-centered onboarding.',),",
                ").build()",
                "",
            ]
        )

    def render_deployment(self, state: SetupOnboardingState) -> str:
        facts = state.detected_facts
        is_sim = facts.get("simulation_requested") is True
        namespace = ros2_namespace(state)
        robot_entries = "\n".join(
            [
                "\n".join(
                    [
                        f"        {item['attachment_id']!r}: {{",
                        f"            'serial_device_by_id': {facts.get('serial_device_by_id')!r},",
                        f"            'namespace': {item['attachment_id']!r},",
                        "        },",
                    ]
                )
                for item in state.robot_attachments
            ]
        )
        if is_sim:
            robot_entries = "\n".join(f"        {item['attachment_id']!r}: {{}}" for item in state.robot_attachments)
        sensor_entries = "\n".join(
            [
                "\n".join(
                    [
                        f"        {item['attachment_id']!r}: {{",
                        "            'driver': 'ros2',",
                        f"            'topic': {sensor_topic(item)!r},",
                        "        },",
                    ]
                )
                for item in state.sensor_attachments
            ]
        )
        launch = launch_command(state)
        launch_line = f"        'launch_command': {launch!r}," if launch else None
        connection_lines = (
            [
                "        'transport': 'ros2',",
                "        'profile_id': 'mujoco_sim',",
                f"        'namespace': '/roboclaw/{state.assembly_id}/sim',",
                f"        'model_path': {facts.get('sim_model_path')!r},",
                f"        'joint_mapping': {(facts.get('sim_joint_mapping') or {})!r},",
            ]
            if is_sim
            else
            [
                "        'transport': 'ros2',",
                f"        'ros_distro': {resolved_ros2_distro(state)!r},",
                f"        'profile_id': {profile_id(state)!r},",
                f"        'namespace': {namespace!r},",
                f"        'serial_device_by_id': {facts.get('serial_device_by_id')!r},",
            ]
        )
        if launch_line is not None:
            connection_lines.append(launch_line)
        return "\n".join(
            [
                '"""Workspace-generated deployment profile."""',
                "",
                "from roboclaw.embodied.definition.systems.deployments import DeploymentProfile",
                "from roboclaw.embodied.workspace import (",
                "    WORKSPACE_SCHEMA_VERSION,",
                "    WorkspaceAssetContract,",
                "    WorkspaceAssetKind,",
                "    WorkspaceExportConvention,",
                "    WorkspaceProvenance,",
                ")",
                "",
                "WORKSPACE_ASSET = WorkspaceAssetContract(",
                "    kind=WorkspaceAssetKind.DEPLOYMENT,",
                "    schema_version=WORKSPACE_SCHEMA_VERSION,",
                "    export_convention=WorkspaceExportConvention.DEPLOYMENT,",
                "    provenance=WorkspaceProvenance(",
                '        source="workspace_generated",',
                '        generator="onboarding_controller",',
                f"        generated_by={state.setup_id!r},",
                f"        generated_at={generated_at()!r},",
                "    ),",
                ")",
                "",
                "DEPLOYMENT = DeploymentProfile(",
                f"    id={state.deployment_id!r},",
                f"    assembly_id={state.assembly_id!r},",
                f"    target_id={'sim' if is_sim else 'real'!r},",
                "    connection={",
                *connection_lines,
                "    },",
                "    robots={",
                robot_entries,
                "    },",
                "    sensors={",
                sensor_entries,
                "    },",
                "    safety_overrides={},",
                ")",
                "",
            ]
        )

    def render_adapter(self, state: SetupOnboardingState) -> str:
        is_sim = state.detected_facts.get("simulation_requested") is True
        implementation = "roboclaw.embodied.execution.integration.adapters.ros2.standard:Ros2ActionServiceAdapter"
        compatibility_lines = [
            "        VersionConstraint(",
            "            component=CompatibilityComponent.TRANSPORT,",
            "            target='ros2',",
            "            requirement='>=1.0,<2.0',",
            "        ),",
        ]
        if not is_sim:
            compatibility_lines.extend(
                [
                    "        VersionConstraint(",
                    "            component=CompatibilityComponent.CONTROL_SURFACE_PROFILE,",
                    f"            target={ARM_HAND_CONTROL_SURFACE_PROFILE.id!r},",
                    "            requirement='>=1.0,<2.0',",
                    "        ),",
                ]
            )
        adapter_lines = [
            '"""Workspace-generated adapter binding."""',
            "",
            "from roboclaw.embodied.definition.foundation.schema import TransportKind",
            "from roboclaw.embodied.execution.integration.adapters import (",
            "    AdapterBinding,",
            "    AdapterCompatibilitySpec,",
            "    CompatibilityComponent,",
            "    VersionConstraint,",
            ")",
            "from roboclaw.embodied.workspace import (",
            "    WORKSPACE_SCHEMA_VERSION,",
            "    WorkspaceAssetContract,",
            "    WorkspaceAssetKind,",
            "    WorkspaceExportConvention,",
            "    WorkspaceProvenance,",
            ")",
            "",
            "WORKSPACE_ASSET = WorkspaceAssetContract(",
            "    kind=WorkspaceAssetKind.ADAPTER,",
            "    schema_version=WORKSPACE_SCHEMA_VERSION,",
            "    export_convention=WorkspaceExportConvention.ADAPTER,",
            "    provenance=WorkspaceProvenance(",
            '        source="workspace_generated",',
            '        generator="onboarding_controller",',
            f"        generated_by={state.setup_id!r},",
            f"        generated_at={generated_at()!r},",
            "    ),",
            ")",
            "",
            "COMPATIBILITY = AdapterCompatibilitySpec(",
            "    constraints=(",
            *compatibility_lines,
            "    ),",
            ")",
            "",
            "ADAPTER = AdapterBinding(",
            f"    id={state.adapter_id!r},",
            f"    assembly_id={state.assembly_id!r},",
            f"    transport=TransportKind.{'DIRECT' if is_sim else 'ROS2'},",
            f"    implementation={implementation!r},",
            f"    supported_targets={('sim',) if is_sim else ('real',)!r},",
        ]
        if not is_sim:
            adapter_lines.append(f"    control_surface_profile_id={ARM_HAND_CONTROL_SURFACE_PROFILE.id!r},")
        adapter_lines.extend(
            [
                "    compatibility=COMPATIBILITY,",
                "    notes=('Generated by assembly-centered onboarding.',),",
                ")",
                "",
            ]
        )
        return "\n".join(adapter_lines)
