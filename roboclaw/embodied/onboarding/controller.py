"""Assembly-centered setup onboarding controller."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.catalog import build_default_catalog
from roboclaw.embodied.execution.integration.bridges import ARM_HAND_BRIDGE
from roboclaw.embodied.onboarding.model import SETUP_STATE_KEY, SetupOnboardingState, SetupStage, SetupStatus
from roboclaw.embodied.workspace import WorkspaceInspectOptions, WorkspaceLintProfile, inspect_workspace_assets
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]


class OnboardingController:
    """Handle first-run embodied setup and later setup refinements."""

    _ROBOT_ALIASES = {
        "so101": ("so101", "so 101"),
    }
    _SETUP_START_KEYWORDS = (
        "connect", "real robot", "setup", "onboard",
    )
    _SETUP_EDIT_KEYWORDS = (
        "camera", "sensor", "serial", "/dev/", "ros2", "deployment", "adapter", "installed", "replace",
    )
    _YES_WORDS = ("yes", "y", "confirmed", "ready", "installed", "connected")
    _NO_WORDS = ("no", "not yet", "not connected", "missing")
    _SERIAL_RE = re.compile(r"(/dev/[^\s,;]+)")

    def __init__(self, workspace: Path, tools: ToolRegistry):
        self.workspace = workspace
        self.tools = tools
        self.catalog = build_default_catalog()

    def should_handle(self, session: Session, content: str) -> bool:
        state = self._load_state(session)
        if state is not None:
            if not state.is_ready:
                return True
            return self._looks_like_setup_edit(content)
        return self._looks_like_setup_start(content)

    async def handle_message(
        self,
        msg: InboundMessage,
        session: Session,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        state = self._load_state(session)
        if state is None:
            state = self._new_state(msg.content)

        state, changed = self._apply_user_input(state, msg.content)
        response = await self._advance(state, msg.content, on_progress=on_progress)

        session.metadata[SETUP_STATE_KEY] = response["state"].to_dict()
        session.add_message("user", msg.content)
        session.add_message("assistant", response["content"])

        if changed or response["state"].stage != state.stage:
            logger.info(
                "Onboarding state {} -> {} ({})",
                state.stage.value,
                response["state"].stage.value,
                response["state"].status.value,
            )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=response["content"],
            metadata=msg.metadata or {},
        )

    def _load_state(self, session: Session) -> SetupOnboardingState | None:
        raw = session.metadata.get(SETUP_STATE_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            return SetupOnboardingState.from_dict(raw)
        except Exception:
            logger.exception("Failed to decode onboarding state for session {}", session.key)
            return None

    def _new_state(self, content: str) -> SetupOnboardingState:
        primary_robot = next(iter(self._extract_robot_ids(content)), "embodied_setup")
        setup_id = f"{primary_robot}_setup"
        return SetupOnboardingState(
            setup_id=setup_id,
            intake_slug=setup_id,
            assembly_id=setup_id,
            deployment_id=f"{setup_id}_real_local",
            adapter_id=f"{setup_id}_ros2_local",
            execution_targets=[{"id": "real", "carrier": "real"}],
        )

    def _looks_like_setup_start(self, content: str) -> bool:
        text = content.lower()
        if any(alias in text for aliases in self._ROBOT_ALIASES.values() for alias in aliases):
            return True
        return any(keyword in text for keyword in self._SETUP_START_KEYWORDS)

    def _looks_like_setup_edit(self, content: str) -> bool:
        text = content.lower()
        return any(keyword in text for keyword in self._SETUP_EDIT_KEYWORDS)

    def _apply_user_input(self, state: SetupOnboardingState, content: str) -> tuple[SetupOnboardingState, bool]:
        changed = False
        robots = list(state.robot_attachments)
        sensors = list(state.sensor_attachments)
        facts = dict(state.detected_facts)

        for robot_id in self._extract_robot_ids(content):
            if not any(item["robot_id"] == robot_id for item in robots):
                attachment_id = "primary" if not robots else f"robot_{len(robots) + 1}"
                robots.append({"attachment_id": attachment_id, "robot_id": robot_id, "role": "primary" if not robots else "secondary"})
                changed = True

        sensor_changes = self._extract_sensor_changes(content)
        for sensor_change in sensor_changes:
            sensors, sensor_changed = self._apply_sensor_change(sensors, sensor_change)
            changed = changed or sensor_changed

        connected = self._extract_connected_state(content)
        if connected is not None and facts.get("connected") != connected:
            facts["connected"] = connected
            changed = True

        serial_path = self._extract_serial_path(content)
        if serial_path and facts.get("serial_device") != serial_path:
            facts["serial_device"] = serial_path
            changed = True

        ros2_state = self._extract_ros2_state(content)
        if ros2_state is not None and facts.get("ros2_available") != ros2_state:
            facts["ros2_available"] = ros2_state
            changed = True
        if self._is_install_confirmation(content):
            facts["ros2_install_confirmed"] = True
            changed = True

        next_status = state.status
        next_stage = state.stage
        if state.is_ready and (sensor_changes or serial_path or ros2_state is not None):
            next_status = SetupStatus.REFINING
            next_stage = SetupStage.MATERIALIZE_ASSEMBLY

        setup_id, intake_slug, assembly_id, deployment_id, adapter_id = self._canonical_ids(
            current_setup_id=state.setup_id,
            robots=robots,
        )

        next_state = replace(
            state,
            setup_id=setup_id,
            intake_slug=intake_slug,
            assembly_id=assembly_id,
            deployment_id=deployment_id,
            adapter_id=adapter_id,
            robot_attachments=robots,
            sensor_attachments=sensors,
            detected_facts=facts,
            status=next_status,
            stage=next_stage,
        )
        return next_state, changed

    def _extract_robot_ids(self, content: str) -> list[str]:
        normalized = content.lower()
        matched: list[str] = []
        for robot_id, aliases in self._ROBOT_ALIASES.items():
            if any(alias in normalized for alias in aliases):
                matched.append(robot_id)
        for manifest in self.catalog.robots.list():
            if manifest.id in normalized and manifest.id not in matched:
                matched.append(manifest.id)
        return matched

    def _extract_sensor_changes(self, content: str) -> list[dict[str, Any]]:
        lower = content.lower()
        if not any(token in lower for token in ("camera", "sensor")):
            return []
        mounts: list[str] = []
        if any(token in lower for token in ("wrist",)):
            mounts.append("wrist")
        if any(token in lower for token in ("overhead",)):
            mounts.append("overhead")
        if any(token in lower for token in ("external",)):
            mounts.append("external")
        if not mounts:
            mounts = ["wrist"]
        remove = any(token in lower for token in ("remove", "drop"))
        mode = "replace" if any(token in lower for token in ("replace", "switch")) else "add"
        return [
            {"sensor_id": "rgb_camera", "mount": mount, "remove": remove, "mode": mode}
            for mount in mounts
        ]

    def _apply_sensor_change(
        self,
        sensors: list[dict[str, Any]],
        change: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        updated = list(sensors)
        sensor_id = change["sensor_id"]
        mount = change["mount"]
        remove = bool(change.get("remove"))
        mode = change.get("mode", "add")
        existing_index = next(
            (index for index, item in enumerate(updated) if item["sensor_id"] == sensor_id and item["mount"] == mount),
            None,
        )
        if remove:
            if existing_index is None:
                return updated, False
            del updated[existing_index]
            return updated, True

        if mode == "add":
            if existing_index is not None:
                return updated, False
            attachment_id = self._sensor_attachment_id(mount, len(updated))
            updated.append({"attachment_id": attachment_id, "sensor_id": sensor_id, "mount": mount})
            return updated, True

        existing_mount_index = next(
            (index for index, item in enumerate(updated) if item["sensor_id"] == sensor_id),
            None,
        )
        if existing_mount_index is not None:
            current = updated[existing_mount_index]
            if current["mount"] == mount:
                return updated, False
            updated[existing_mount_index] = {
                **current,
                "mount": mount,
                "attachment_id": self._sensor_attachment_id(mount, existing_mount_index),
            }
            return updated, True

        attachment_id = self._sensor_attachment_id(mount, len(updated))
        updated.append({"attachment_id": attachment_id, "sensor_id": sensor_id, "mount": mount})
        return updated, True

    @staticmethod
    def _sensor_attachment_id(mount: str, index: int) -> str:
        base = {
            "wrist": "wrist_camera",
            "overhead": "overhead_camera",
            "external": "external_camera",
        }.get(mount, "camera")
        return base if index == 0 or base != "camera" else f"{base}_{index + 1}"

    def _extract_connected_state(self, content: str) -> bool | None:
        lower = content.lower()
        if any(token in lower for token in ("connected",)):
            return True
        if "connect" in lower and any(token in lower for token in ("not", "no")):
            return False
        return None

    def _extract_serial_path(self, content: str) -> str | None:
        match = self._SERIAL_RE.search(content)
        return match.group(1) if match else None

    def _extract_ros2_state(self, content: str) -> bool | None:
        lower = content.lower()
        if "ros2" not in lower:
            return None
        if any(token in lower for token in ("missing", "not installed", "unavailable")):
            return False
        if any(token in lower for token in ("available", "installed")):
            return True
        return None

    def _is_install_confirmation(self, content: str) -> bool:
        stripped = content.strip().lower()
        return any(stripped == token or stripped.startswith(f"{token} ") for token in self._YES_WORDS)

    async def _advance(
        self,
        state: SetupOnboardingState,
        content: str,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        state = await self._write_intake(state, on_progress=on_progress)

        if not state.robot_attachments:
            next_state = replace(
                state,
                stage=SetupStage.IDENTIFY_SETUP_SCOPE,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["robot_or_setup_components"],
            )
            return {
                "state": next_state,
                "content": (
                    "I will build this setup as an assembly-first onboarding flow."
                    "\nFirst tell me which robots and sensors belong in this setup."
                    "\nFor example: `SO101`, `SO101 + wrist camera`, or `dual arms + overhead camera`."
                ),
            }

        if state.detected_facts.get("connected") is not True:
            next_state = replace(
                state,
                stage=SetupStage.CONFIRM_CONNECTED,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["connected"],
            )
            return {
                "state": next_state,
                "content": (
                    f"I recorded the current setup scope in intake: {self._component_summary(state)}."
                    "\nNext I only need one minimal fact: are these devices already connected to this machine?"
                    "\nReply with `connected` or `not connected`."
                ),
            }

        if state.stage in (SetupStage.CONFIRM_CONNECTED, SetupStage.IDENTIFY_SETUP_SCOPE, SetupStage.PROBE_LOCAL_ENVIRONMENT):
            state = await self._probe_environment(state, on_progress=on_progress)
            if state.detected_facts.get("ros2_available") is False:
                guide_summary = await self._read_ros2_guide(on_progress=on_progress)
                next_state = replace(
                    state,
                    stage=SetupStage.RESOLVE_PREREQUISITES,
                    status=SetupStatus.BOOTSTRAPPING,
                    missing_facts=["ros2_install"],
                )
                content = (
                    "Local probing is complete. This setup needs ROS2, but ROS2 is not available on this machine yet."
                    f"\nI also read the workspace ROS2 install guide: {guide_summary}."
                    "\nIf you want me to continue along that guide, reply with `start ROS2 install`. Once ROS2 is installed, I will continue generating the assembly, deployment, and adapter."
                )
                return {"state": next_state, "content": content}

            state = replace(
                state,
                stage=SetupStage.MATERIALIZE_ASSEMBLY,
                missing_facts=[],
            )

        if state.stage == SetupStage.RESOLVE_PREREQUISITES:
            if state.detected_facts.get("ros2_install_confirmed") and state.detected_facts.get("ros2_available") is not True:
                return {
                    "state": state,
                    "content": (
                        "This setup is now locked into the ROS2 prerequisite branch."
                        "\nFinish the ROS2 install using the workspace guide, then reply with `ROS2 installed` and I will continue generating the assembly, deployment, and adapter."
                    ),
                }
            if state.detected_facts.get("ros2_available") is not True:
                return {
                    "state": state,
                    "content": (
                        "This setup is still waiting in the ROS2 prerequisite stage."
                        "\nOnce ROS2 is installed, or once you ask me to start the ROS2 install path, I will continue generating the setup assets."
                    ),
                }
            state = replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[])

        if state.stage == SetupStage.MATERIALIZE_ASSEMBLY:
            state = await self._write_assembly(state, on_progress=on_progress)
            state = replace(state, stage=SetupStage.MATERIALIZE_DEPLOYMENT_ADAPTER)

        if state.stage == SetupStage.MATERIALIZE_DEPLOYMENT_ADAPTER:
            state = await self._write_deployment(state, on_progress=on_progress)
            state = await self._write_adapter(state, on_progress=on_progress)
            state = replace(state, stage=SetupStage.VALIDATE_SETUP)

        if state.stage == SetupStage.VALIDATE_SETUP:
            validation = inspect_workspace_assets(
                self.workspace,
                options=WorkspaceInspectOptions(lint_profile=WorkspaceLintProfile.BASIC),
            )
            if validation.has_errors:
                issues = "\n".join(f"- {issue.path}: {issue.message}" for issue in validation.issues[:5])
                return {
                    "state": state,
                    "content": f"The setup assets were written, but validation is still failing:\n{issues}",
                }
            state = replace(state, stage=SetupStage.HANDOFF_READY, status=SetupStatus.READY)

        return {
            "state": state,
            "content": (
                f"This setup is now ready: {self._component_summary(state)}."
                "\nI wrote the assembly, deployment, and adapter into the workspace. You can keep refining setup details in chat, or continue with connect / calibrate / move / debug / reset."
                f"\nGenerated assets: {self._asset_summary(state)}"
            ),
        }

    async def _probe_environment(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        facts = dict(state.detected_facts)
        if any(item["robot_id"] == "so101" for item in state.robot_attachments) and not facts.get("serial_device"):
            probe = await self._run_tool(
                "exec",
                {
                    "command": (
                        "bash -lc 'ls -1 /dev/serial/by-id 2>/dev/null; "
                        "ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null'"
                    )
                },
                on_progress=on_progress,
            )
            facts["serial_device"] = self._select_serial_device(probe)
        if "ros2_available" not in facts:
            probe = await self._run_tool(
                "exec",
                {
                    "command": (
                        "bash -lc 'if command -v ros2 >/dev/null 2>&1; then "
                        "printf \"ROS2_OK\\n\"; "
                        "ros2 --version 2>/dev/null || true; "
                        "printf \"ROS_DISTRO=%s\\n\" \"${ROS_DISTRO:-}\"; "
                        "else printf \"ROS2_MISSING\\n\"; fi'"
                    )
                },
                on_progress=on_progress,
            )
            facts["ros2_available"] = "ROS2_OK" in probe
            distro_match = re.search(r"ROS_DISTRO=([^\n]+)", probe)
            if distro_match and distro_match.group(1).strip():
                facts["ros2_distro"] = distro_match.group(1).strip()
        notes = list(state.notes)
        if facts.get("serial_device"):
            notes = self._extend_unique(notes, f"probe:serial={facts['serial_device']}")
        if facts.get("ros2_distro"):
            notes = self._extend_unique(notes, f"probe:ros2={facts['ros2_distro']}")
        return replace(state, stage=SetupStage.PROBE_LOCAL_ENVIRONMENT, detected_facts=facts, notes=notes)

    async def _read_ros2_guide(self, *, on_progress: Callable[..., Awaitable[None]] | None = None) -> str:
        guide_path = self.workspace / "embodied" / "guides" / "ROS2_INSTALL.md"
        content = await self._run_tool("read_file", {"path": str(guide_path)}, on_progress=on_progress)
        if content.startswith("Error"):
            return str(guide_path)
        first_heading = next((line[2:].strip() for line in content.splitlines() if line.startswith("## ")), None)
        return f"{guide_path.name}{f' / {first_heading}' if first_heading else ''}"

    async def _write_intake(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        intake_path = self.workspace / "embodied" / "intake" / f"{state.intake_slug}.md"
        content = self._render_intake(state)
        await self._run_tool("write_file", {"path": str(intake_path), "content": content}, on_progress=on_progress)
        assets = dict(state.generated_assets)
        assets["intake"] = str(intake_path)
        return replace(state, generated_assets=assets)

    async def _write_assembly(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "assemblies" / f"{state.assembly_id}.py"
        await self._run_tool("write_file", {"path": str(path), "content": self._render_assembly(state)}, on_progress=on_progress)
        assets = dict(state.generated_assets)
        assets["assembly"] = str(path)
        return replace(state, generated_assets=assets)

    async def _write_deployment(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "deployments" / f"{state.deployment_id}.py"
        await self._run_tool("write_file", {"path": str(path), "content": self._render_deployment(state)}, on_progress=on_progress)
        assets = dict(state.generated_assets)
        assets["deployment"] = str(path)
        return replace(state, generated_assets=assets)

    async def _write_adapter(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        path = self.workspace / "embodied" / "adapters" / f"{state.adapter_id}.py"
        await self._run_tool("write_file", {"path": str(path), "content": self._render_adapter(state)}, on_progress=on_progress)
        assets = dict(state.generated_assets)
        assets["adapter"] = str(path)
        return replace(state, generated_assets=assets)

    async def _run_tool(
        self,
        name: str,
        params: dict[str, Any],
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        if on_progress is not None:
            await on_progress(self._format_tool_hint(name, params), tool_hint=True)
        logger.info("Onboarding tool call: {}({})", name, json.dumps(params, ensure_ascii=False)[:200])
        result = await self.tools.execute(name, params)
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

    def _tool_result_summary(self, name: str, params: dict[str, Any], result: str) -> str | None:
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

    @staticmethod
    def _select_serial_device(output: str) -> str | None:
        for line in output.splitlines():
            candidate = line.strip()
            if candidate.startswith("/dev/serial/by-id/"):
                return candidate
        for line in output.splitlines():
            candidate = line.strip()
            if candidate.startswith("/dev/ttyACM") or candidate.startswith("/dev/ttyUSB"):
                return candidate
        return None

    def _render_intake(self, state: SetupOnboardingState) -> str:
        robot_lines = "\n".join(
            f"- `{item['attachment_id']}`: `{item['robot_id']}` ({item['role']})" for item in state.robot_attachments
        ) or "- pending"
        sensor_lines = "\n".join(
            f"- `{item['attachment_id']}`: `{item['sensor_id']}` mounted as `{item['mount']}`" for item in state.sensor_attachments
        ) or "- none yet"
        facts = state.detected_facts
        fact_lines = [
            f"- connected: `{facts.get('connected', 'unknown')}`",
            f"- serial_device: `{facts.get('serial_device', 'unknown')}`",
            f"- ros2_available: `{facts.get('ros2_available', 'unknown')}`",
            f"- ros2_distro: `{facts.get('ros2_distro', 'unknown')}`",
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

    def _render_assembly(self, state: SetupOnboardingState) -> str:
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
                        f"            mount_frame={self._mount_frame(item['mount'])!r},",
                        "            mount_transform=Transform3D(),",
                        "        ),",
                    ]
                )
                for item in state.sensor_attachments
            ]
        )
        if not sensor_blocks:
            sensor_blocks = ""
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
                "from roboclaw.embodied.execution.integration.carriers.real import build_real_ros2_target",
                "from roboclaw.embodied.execution.integration.transports.ros2 import build_standard_ros2_contract",
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
                f"        generated_at={self._generated_at()!r},",
                "    ),",
                ")",
                "",
                "REAL_TARGET = build_real_ros2_target(",
                "    target_id='real',",
                f"    description={f'Real target for {state.setup_id}'!r},",
                f"    ros2=build_standard_ros2_contract({state.assembly_id!r}, 'real'),",
                ")",
                "",
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
                "    execution_targets=(REAL_TARGET,),",
                "    default_execution_target_id='real',",
                "    frame_transforms=(",
                "        FrameTransform(parent_frame='world', child_frame='base_link', transform=Transform3D()),",
                "        FrameTransform(parent_frame='base_link', child_frame='tool0', transform=Transform3D()),",
                "    ),",
                "    tools=(),",
                "    control_groups=(),",
                "    safety_zones=(),",
                "    safety_boundaries=(),",
                "    failure_domains=(),",
                "    resource_ownerships=(),",
                "    notes=('Generated by assembly-centered onboarding.',),",
                ").build()",
                "",
            ]
        )

    def _render_deployment(self, state: SetupOnboardingState) -> str:
        facts = state.detected_facts
        robot_entries = "\n".join(
            [
                "\n".join(
                    [
                        f"        {item['attachment_id']!r}: {{",
                        f"            'serial_device': {facts.get('serial_device')!r},",
                        f"            'namespace': {item['attachment_id']!r},",
                        "        },",
                    ]
                )
                for item in state.robot_attachments
            ]
        )
        sensor_entries = "\n".join(
            [
                "\n".join(
                    [
                        f"        {item['attachment_id']!r}: {{",
                        "            'driver': 'ros2',",
                        f"            'topic': {self._sensor_topic(item)!r},",
                        "        },",
                    ]
                )
                for item in state.sensor_attachments
            ]
        )
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
                f"        generated_at={self._generated_at()!r},",
                "    ),",
                ")",
                "",
                "DEPLOYMENT = DeploymentProfile(",
                f"    id={state.deployment_id!r},",
                f"    assembly_id={state.assembly_id!r},",
                "    target_id='real',",
                f"    connection={{'transport': 'ros2', 'ros_distro': {state.detected_facts.get('ros2_distro')!r}}},",
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

    def _render_adapter(self, state: SetupOnboardingState) -> str:
        implementation = f"workspace.adapters.{state.adapter_id}:Adapter"
        return "\n".join(
            [
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
                f"        generated_at={self._generated_at()!r},",
                "    ),",
                ")",
                "",
                "COMPATIBILITY = AdapterCompatibilitySpec(",
                "    constraints=(",
                "        VersionConstraint(",
                "            component=CompatibilityComponent.TRANSPORT,",
                "            target='ros2',",
                "            requirement='>=1.0,<2.0',",
                "        ),",
                "        VersionConstraint(",
                "            component=CompatibilityComponent.BRIDGE,",
                f"            target={ARM_HAND_BRIDGE.id!r},",
                "            requirement='>=1.0,<2.0',",
                "        ),",
                "    ),",
                ")",
                "",
                "ADAPTER = AdapterBinding(",
                f"    id={state.adapter_id!r},",
                f"    assembly_id={state.assembly_id!r},",
                "    transport=TransportKind.ROS2,",
                f"    implementation={implementation!r},",
                "    supported_targets=('real',),",
                f"    bridge_id={ARM_HAND_BRIDGE.id!r},",
                "    compatibility=COMPATIBILITY,",
                "    notes=('Generated by assembly-centered onboarding.',),",
                ")",
                "",
            ]
        )

    @staticmethod
    def _mount_frame(mount: str) -> str:
        if mount == "wrist":
            return "tool0"
        return "world"

    @staticmethod
    def _sensor_topic(sensor: dict[str, Any]) -> str:
        if sensor["mount"] == "wrist":
            return "/wrist_camera/image_raw"
        if sensor["mount"] == "overhead":
            return "/overhead_camera/image_raw"
        return f"/{sensor['attachment_id']}/image_raw"

    @staticmethod
    def _extend_unique(items: list[str], value: str) -> list[str]:
        if value not in items:
            items.append(value)
        return items

    @staticmethod
    def _component_summary(state: SetupOnboardingState) -> str:
        robots = ", ".join(item["robot_id"] for item in state.robot_attachments) or "no robot yet"
        sensors = ", ".join(f"{item['sensor_id']}@{item['mount']}" for item in state.sensor_attachments) or "no sensor yet"
        return f"robots=[{robots}] sensors=[{sensors}]"

    @staticmethod
    def _asset_summary(state: SetupOnboardingState) -> str:
        return ", ".join(f"{key}={Path(value).name}" for key, value in sorted(state.generated_assets.items()))

    @staticmethod
    def _generated_at() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _canonical_ids(
        self,
        *,
        current_setup_id: str,
        robots: list[dict[str, Any]],
    ) -> tuple[str, str, str, str, str]:
        if not robots:
            setup_id = current_setup_id
        elif current_setup_id.startswith("embodied_setup"):
            primary_robot_id = robots[0]["robot_id"]
            setup_id = f"{primary_robot_id}_setup"
        else:
            setup_id = current_setup_id
        return (
            setup_id,
            setup_id,
            setup_id,
            f"{setup_id}_real_local",
            f"{setup_id}_ros2_local",
        )
