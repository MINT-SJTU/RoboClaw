"""Assembly-centered setup onboarding controller."""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.catalog import build_default_catalog
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import get_ros2_profile
from roboclaw.embodied.execution.integration.control_surfaces import ARM_HAND_CONTROL_SURFACE_PROFILE
from roboclaw.embodied.onboarding.model import SETUP_STATE_KEY, SetupOnboardingState, SetupStage, SetupStatus
from roboclaw.embodied.onboarding.ros2_install import (
    advance_ros2_install_step,
    extract_ros2_profile,
    extract_ros2_state,
    is_ros2_install_request,
    is_ros2_step_advance,
    parse_key_value_output,
    render_ros2_install_step,
    render_ros2_shell_repair,
    ros2_install_summary,
    select_ros2_recipe,
)
from roboclaw.embodied.workspace import WorkspaceInspectOptions, WorkspaceLintProfile, inspect_workspace_assets
from roboclaw.config.paths import resolve_serial_by_id_path
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]
SO101_SERIAL_PROBE_MODULE = "roboclaw.embodied.execution.integration.control_surfaces.ros2.scservo_probe"


class OnboardingController:
    """Handle first-run embodied setup and later setup refinements."""

    _ROBOT_ALIASES = {
        "so101": ("so101", "so 101"),
    }
    _SETUP_START_KEYWORDS = (
        "connect", "real robot", "setup", "onboard",
        "真实机器人", "真实的机器人", "机械臂", "机器人",
    )
    _SETUP_EDIT_KEYWORDS = (
        "camera", "sensor", "serial", "/dev/", "ros2", "deployment", "adapter", "installed", "replace",
    )
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

    def has_active_onboarding(self, session: Session) -> bool:
        """Return whether the session is currently mid-onboarding."""
        state = self._load_state(session)
        return state is not None and not state.is_ready

    def should_handle_setup_edit(self, session: Session, content: str) -> bool:
        """Return whether a ready session should be routed back into setup editing."""
        state = self._load_state(session)
        return state is not None and state.is_ready and self._looks_like_setup_edit(content)

    def looks_like_setup_start(self, content: str) -> bool:
        """Expose setup-start detection for outer routing."""
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
        if serial_path:
            serial_by_id = self._normalize_serial_device_by_id(serial_path)
            if serial_by_id is not None and facts.get("serial_device_by_id") != serial_by_id:
                self._set_serial_device_by_id(facts, serial_by_id)
                changed = True
            elif serial_by_id is None:
                if facts.get("serial_device_unstable") is not True or facts.get("serial_device_unresponsive") is True:
                    self._set_unstable_serial_device(facts)
                    changed = True

        ros2_profile = extract_ros2_profile(content)
        if ros2_profile and facts.get("ros2_install_profile") != ros2_profile:
            facts["ros2_install_profile"] = ros2_profile
            changed = True

        ros2_state = extract_ros2_state(content)
        if ros2_state is False and facts.get("ros2_available") is not False:
            facts["ros2_available"] = False
            changed = True
        if ros2_state is True:
            facts["ros2_reported_installed"] = True
            changed = True
        if is_ros2_install_request(content):
            facts["ros2_install_requested"] = True
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
        if any(token in lower for token in ("connected", "已连接", "连接好了", "连好了", "接好了", "接上了", "连上了", "都连好了")):
            return True
        if (
            ("connect" in lower and any(token in lower for token in ("not", "no")))
            or any(token in lower for token in ("没连", "没有连接", "未连接", "还没连", "没接好"))
        ):
            return False
        return None

    def _extract_serial_path(self, content: str) -> str | None:
        match = self._SERIAL_RE.search(content)
        return match.group(1) if match else None

    async def _advance(
        self,
        state: SetupOnboardingState,
        content: str,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        state = await self._write_intake(state, on_progress=on_progress)
        primary_profile = self._primary_profile(state)

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

        if primary_profile is None:
            primary_robot_id = state.robot_attachments[0]["robot_id"]
            next_state = replace(
                state,
                stage=SetupStage.IDENTIFY_SETUP_SCOPE,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["control_surface_profile"],
            )
            return {
                "state": next_state,
                "content": (
                    f"RoboClaw does not have a framework ROS2 control surface profile for `{primary_robot_id}` yet."
                    "\nThis onboarding flow will not generate a standard ROS2 deployment/adapter until that profile exists."
                    "\nSwitch to a supported robot profile or add an embodiment-owned control surface profile first."
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
            if state.detected_facts.get("serial_device_unresponsive") is True:
                next_state = replace(
                    state,
                    stage=SetupStage.PROBE_LOCAL_ENVIRONMENT,
                    status=SetupStatus.BOOTSTRAPPING,
                    missing_facts=["serial_device_by_id"],
                )
                detail = state.detected_facts.get("serial_probe_error", "SO101 serial probe did not receive a status packet.")
                return {
                    "state": next_state,
                    "content": (
                        "I found a stable `/dev/serial/by-id/...` device, but it did not answer an SO101 servo probe."
                        f"\nProbe result: `{detail}`."
                        "\nConnect the actual SO101 controller or expose the correct stable by-id device, then reply again."
                    ),
                }
            if primary_profile is not None and primary_profile.auto_probe_serial and not state.detected_facts.get("serial_device_by_id"):
                next_state = replace(
                    state,
                    stage=SetupStage.PROBE_LOCAL_ENVIRONMENT,
                    status=SetupStatus.BOOTSTRAPPING,
                    missing_facts=["serial_device_by_id"],
                )
                return {
                    "state": next_state,
                    "content": (
                        "I found a serial device, but I will not persist an unstable tty node."
                        "\nPlease expose a stable `/dev/serial/by-id/...` mapping for the SO101 controller, then reply again."
                    ),
                }
            if primary_profile is not None and primary_profile.requires_calibration:
                calibration_path = str(state.detected_facts.get("calibration_path") or "").strip()
                if not calibration_path:
                    next_state = replace(
                        state,
                        stage=SetupStage.PROBE_LOCAL_ENVIRONMENT,
                        status=SetupStatus.BOOTSTRAPPING,
                        missing_facts=["calibration_file"],
                    )
                    expected_path = primary_profile.canonical_calibration_path()
                    return {
                        "state": next_state,
                        "content": (
                            f"This `{primary_profile.robot_id}` profile requires framework-managed calibration before execution."
                            f"\nExpected canonical path: `{expected_path}`."
                            "\nNo calibration file is available in this container instance yet."
                            "\nReply with `calibrate` and I will guide the calibration step next."
                        ),
                    }
            if (
                state.detected_facts.get("ros2_available") is not True
                and state.detected_facts.get("ros2_installed_distros")
                and state.detected_facts.get("ros2_shell_initialized") is True
            ):
                state = replace(
                    state,
                    stage=SetupStage.MATERIALIZE_ASSEMBLY,
                    missing_facts=[],
                )
            elif state.detected_facts.get("ros2_available") is not True:
                state = await self._probe_install_host(state, on_progress=on_progress)
                guide_summary = await self._read_ros2_guide(on_progress=on_progress)
                recipe = select_ros2_recipe(state.detected_facts)
                next_state = replace(
                    state,
                    stage=SetupStage.RESOLVE_PREREQUISITES,
                    status=SetupStatus.BOOTSTRAPPING,
                    missing_facts=["ros2_install"],
                )
                if state.detected_facts.get("ros2_installed_distros") and state.detected_facts.get("ros2_shell_initialized") is False:
                    content = (
                        "Local probing is complete. This setup needs ROS2, and RoboClaw found a partial install on this machine."
                        f"\nI also read the workspace ROS2 install guide: {guide_summary}."
                        f"\nSelected install path: {ros2_install_summary(recipe, state.detected_facts)}."
                        f"\n{render_ros2_shell_repair(state.detected_facts, recipe)}"
                    )
                else:
                    content = (
                        "Local probing is complete. This setup needs ROS2, but ROS2 is not available on this machine yet."
                        f"\nI also read the workspace ROS2 install guide: {guide_summary}."
                        f"\nSelected install path: {ros2_install_summary(recipe, state.detected_facts)}."
                        "\nReply with `start ROS2 install` to let me prepare the RoboClaw-guided install flow."
                        "\nIf you want GUI tools such as RViz, say `need desktop tools` before starting the install."
                    )
                return {"state": next_state, "content": content}

            state = replace(
                state,
                stage=SetupStage.MATERIALIZE_ASSEMBLY,
                missing_facts=[],
            )

        if state.stage in (
            SetupStage.RESOLVE_PREREQUISITES,
            SetupStage.INSTALL_PREREQUISITES,
            SetupStage.VALIDATE_PREREQUISITES,
        ):
            if state.detected_facts.get("ros2_reported_installed"):
                facts = dict(state.detected_facts)
                facts.pop("ros2_reported_installed", None)
                state = replace(state, detected_facts=facts)
                state = await self._probe_environment(state, on_progress=on_progress, force_ros2_probe=True)
                if state.detected_facts.get("ros2_available") is True:
                    state = replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[])
                else:
                    if state.detected_facts.get("ros2_installed_distros") and state.detected_facts.get("ros2_shell_initialized") is False:
                        return {
                            "state": state,
                            "content": render_ros2_shell_repair(state.detected_facts),
                        }
                    return {
                        "state": state,
                        "content": (
                            "I re-checked this machine after your update, but `ros2` is still not available in the shell yet."
                            "\nContinue with the guided install steps, open a fresh shell if needed, then reply with `ROS2 installed`."
                        ),
                    }

            if (
                state.stage == SetupStage.RESOLVE_PREREQUISITES
                and state.detected_facts.get("ros2_install_requested")
                and state.detected_facts.get("ros2_available") is not True
            ):
                state, install_message = await self._prepare_ros2_install(state, on_progress=on_progress)
                if install_message is not None:
                    return {"state": state, "content": install_message}

            if state.stage == SetupStage.VALIDATE_PREREQUISITES and state.detected_facts.get("ros2_available") is not True:
                return {
                    "state": state,
                    "content": (
                        "The guided ROS2 install steps are complete."
                        "\nFinish the commands in your shell, then reply with `ROS2 installed` and I will verify the environment before generating the setup assets."
                    ),
                }

            if state.stage == SetupStage.INSTALL_PREREQUISITES and state.detected_facts.get("ros2_available") is not True:
                if is_ros2_step_advance(content):
                    next_facts, should_validate = advance_ros2_install_step(state.detected_facts)
                    state = replace(
                        state,
                        stage=SetupStage.VALIDATE_PREREQUISITES if should_validate else SetupStage.INSTALL_PREREQUISITES,
                        detected_facts=next_facts,
                    )
                if state.stage == SetupStage.VALIDATE_PREREQUISITES:
                    return {
                        "state": state,
                        "content": (
                            "The guided ROS2 install steps are complete."
                            "\nFinish the commands in your shell, then reply with `ROS2 installed` and I will verify the environment before generating the setup assets."
                        ),
                    }
                return {"state": state, "content": render_ros2_install_step(state.detected_facts)}

            if state.detected_facts.get("ros2_available") is not True:
                return {
                    "state": state,
                    "content": (
                        "This setup is still waiting in the ROS2 prerequisite stage."
                        "\nReply with `start ROS2 install` and I will prepare or run the guided install flow."
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
        force_ros2_probe: bool = False,
    ) -> SetupOnboardingState:
        facts = dict(state.detected_facts)
        primary_robot_id = state.robot_attachments[0]["robot_id"] if state.robot_attachments else None
        primary_profile = get_ros2_profile(primary_robot_id)
        if primary_profile is not None and primary_profile.auto_probe_serial and not facts.get("serial_device_by_id"):
            probe = await self._run_tool(
                "exec",
                {
                "command": (
                        "bash -lc 'ROOT=\"${ROBOCLAW_HOST_DEV_ROOT:-/dev}\"; "
                        "for link in /dev/serial/by-id/* \"$ROOT\"/serial/by-id/*; do "
                        "[ -e \"$link\" ] || continue; "
                        "name=\"$link\"; "
                        "case \"$link\" in \"$ROOT\"/*) name=\"/dev/${link#\"$ROOT\"/}\" ;; esac; "
                        "resolved=\"$(readlink -f \"$link\" 2>/dev/null || true)\"; "
                        "case \"$resolved\" in \"$ROOT\"/*) resolved=\"/dev/${resolved#\"$ROOT\"/}\" ;; esac; "
                        "printf \"%s -> %s\\n\" \"$name\" \"$resolved\"; "
                        "done | awk '!seen[$0]++'; "
                        "ls -1 /dev/ttyACM* /dev/ttyUSB* \"$ROOT\"/ttyACM* \"$ROOT\"/ttyUSB* 2>/dev/null "
                        "| sed \"s#^$ROOT#/dev#\" | awk '!seen[$0]++''"
                    )
                },
                on_progress=on_progress,
            )
            serial_by_id = self._select_serial_device_by_id(probe)
            if serial_by_id is not None:
                serial_check = await self._probe_so101_serial_device(serial_by_id, on_progress=on_progress)
                if serial_check["ok"]:
                    self._set_serial_device_by_id(facts, serial_by_id)
                else:
                    self._set_unresponsive_serial_device(facts, str(serial_check["detail"]))
            else:
                self._set_unstable_serial_device(facts)
        if primary_profile is not None and primary_profile.requires_calibration:
            calibration_path = primary_profile.ensure_canonical_calibration()
            if calibration_path is not None and calibration_path.exists():
                facts["calibration_path"] = str(calibration_path)
                facts.pop("calibration_missing", None)
            else:
                facts.pop("calibration_path", None)
                facts["calibration_missing"] = True
        if force_ros2_probe or "ros2_available" not in facts:
            probe = await self._run_tool(
                "exec",
                {
                    "command": (
                        "bash -lc 'installed=$(for d in /opt/ros/*; do [ -x \"$d/bin/ros2\" ] && basename \"$d\"; done 2>/dev/null | paste -sd, -); "
                        "shell_init=0; "
                        "if [ -n \"$installed\" ]; then "
                        "for distro in $(printf \"%s\" \"$installed\" | tr \",\" \" \"); do "
                        "if grep -F \"/opt/ros/$distro/\" ~/.bashrc ~/.zshrc 2>/dev/null >/dev/null; then shell_init=1; break; fi; "
                        "done; "
                        "fi; "
                        "if command -v ros2 >/dev/null 2>&1; then "
                        "printf \"ROS2_OK\\n\"; "
                        "ros2 --version 2>/dev/null || true; "
                        "printf \"ROS_DISTRO=%s\\n\" \"${ROS_DISTRO:-}\"; "
                        "printf \"ROS2_SHELL_INIT=%s\\n\" \"$shell_init\"; "
                        "else "
                        "if [ -n \"$installed\" ]; then printf \"ROS2_PRESENT\\nINSTALLED_DISTROS=%s\\n\" \"$installed\"; "
                        "else printf \"ROS2_MISSING\\n\"; fi; "
                        "printf \"ROS2_SHELL_INIT=%s\\n\" \"$shell_init\"; "
                        "fi'"
                    )
                },
                on_progress=on_progress,
            )
            facts["ros2_available"] = "ROS2_OK" in probe
            facts["ros2_shell_initialized"] = "ROS2_SHELL_INIT=1" in probe
            if facts["ros2_available"]:
                facts.pop("ros2_install_requested", None)
                facts.pop("ros2_install_step_index", None)
                facts.pop("ros2_install_step_total", None)
            distro_match = re.search(r"ROS_DISTRO=([^\n]+)", probe)
            if distro_match and distro_match.group(1).strip():
                facts["ros2_distro"] = distro_match.group(1).strip()
            installed_match = re.search(r"INSTALLED_DISTROS=([^\n]+)", probe)
            if installed_match and installed_match.group(1).strip():
                facts["ros2_installed_distros"] = installed_match.group(1).strip().split(",")
                if not facts.get("ros2_distro"):
                    facts["ros2_distro"] = facts["ros2_installed_distros"][0]
            else:
                facts.pop("ros2_installed_distros", None)
        notes = list(state.notes)
        if facts.get("serial_device_by_id"):
            notes = self._extend_unique(notes, f"probe:serial={facts['serial_device_by_id']}")
        if facts.get("serial_probe_error"):
            notes = self._extend_unique(notes, f"probe:serial_check={facts['serial_probe_error']}")
        if facts.get("calibration_path"):
            notes = self._extend_unique(notes, f"probe:calibration={facts['calibration_path']}")
        if facts.get("ros2_distro"):
            notes = self._extend_unique(notes, f"probe:ros2={facts['ros2_distro']}")
        return replace(state, stage=SetupStage.PROBE_LOCAL_ENVIRONMENT, detected_facts=facts, notes=notes)

    async def _probe_so101_serial_device(
        self,
        serial_by_id: str,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, str | bool]:
        probe = await self._run_tool(
            "exec",
            {
                "command": (
                    "bash -lc 'PY_BIN=\"$(command -v python3 || command -v python || true)\"; "
                    "if [ -z \"$PY_BIN\" ]; then printf \"ROBOCLAW_SO101_SERIAL_PYTHON_MISSING\\n\"; exit 0; fi; "
                    f"\"$PY_BIN\" -m {SO101_SERIAL_PROBE_MODULE} {shlex.quote(serial_by_id)}'"
                )
            },
            on_progress=on_progress,
        )
        if "ROBOCLAW_SO101_SERIAL_OK" in probe:
            return {"ok": True, "detail": ""}
        lines = [line.strip() for line in probe.splitlines() if line.strip()]
        detail = lines[-1] if lines else "SO101 serial probe failed"
        return {"ok": False, "detail": detail}

    async def _read_ros2_guide(self, *, on_progress: Callable[..., Awaitable[None]] | None = None) -> str:
        guide_path = self.workspace / "embodied" / "guides" / "ROS2_INSTALL.md"
        content = await self._run_tool("read_file", {"path": str(guide_path)}, on_progress=on_progress)
        if content.startswith("Error"):
            return str(guide_path)
        first_heading = next((line[2:].strip() for line in content.splitlines() if line.startswith("## ")), None)
        return f"{guide_path.name}{f' / {first_heading}' if first_heading else ''}"

    async def _probe_install_host(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> SetupOnboardingState:
        facts = dict(state.detected_facts)
        if facts.get("host_os_id") and facts.get("host_shell") and "host_passwordless_sudo" in facts:
            return state
        probe = await self._run_tool(
            "exec",
            {
                "command": (
                    "bash -lc '. /etc/os-release 2>/dev/null || true; "
                    "printf \"ID=%s\\n\" \"${ID:-}\"; "
                    "printf \"VERSION_ID=%s\\n\" \"${VERSION_ID:-}\"; "
                    "printf \"VERSION_CODENAME=%s\\n\" \"${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}\"; "
                    "printf \"PRETTY_NAME=%s\\n\" \"${PRETTY_NAME:-}\"; "
                    "printf \"SHELL_NAME=%s\\n\" \"${SHELL##*/}\"; "
                    "printf \"CONDA_PREFIX=%s\\n\" \"${CONDA_PREFIX:-}\"; "
                    "if grep -qi microsoft /proc/version 2>/dev/null; then printf \"WSL=1\\n\"; else printf \"WSL=0\\n\"; fi; "
                    "if command -v sudo >/dev/null 2>&1; then printf \"SUDO=1\\n\"; else printf \"SUDO=0\\n\"; fi; "
                    "if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then printf \"SUDO_PASSWORDLESS=1\\n\"; else printf \"SUDO_PASSWORDLESS=0\\n\"; fi'"
                )
            },
            on_progress=on_progress,
        )
        values = parse_key_value_output(probe)
        facts["host_os_id"] = values.get("ID", "").strip().lower()
        facts["host_os_version"] = values.get("VERSION_ID", "").strip()
        facts["host_os_codename"] = values.get("VERSION_CODENAME", "").strip().lower()
        facts["host_pretty_name"] = values.get("PRETTY_NAME", "").strip()
        facts["host_shell"] = values.get("SHELL_NAME", "").strip().lower() or "bash"
        facts["conda_prefix"] = values.get("CONDA_PREFIX", "").strip()
        facts["host_is_wsl"] = values.get("WSL", "0").strip() == "1"
        facts["host_has_sudo"] = values.get("SUDO", "0").strip() == "1"
        facts["host_passwordless_sudo"] = values.get("SUDO_PASSWORDLESS", "0").strip() == "1"
        notes = list(state.notes)
        if facts.get("host_pretty_name"):
            notes = self._extend_unique(notes, f"probe:host={facts['host_pretty_name']}")
        return replace(state, detected_facts=facts, notes=notes)

    async def _prepare_ros2_install(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[SetupOnboardingState, str | None]:
        state = await self._probe_install_host(state, on_progress=on_progress)
        recipe = select_ros2_recipe(state.detected_facts)
        if recipe is None:
            next_state = replace(
                state,
                stage=SetupStage.RESOLVE_PREREQUISITES,
                missing_facts=["supported_ros2_host"],
            )
            return next_state, (
                "RoboClaw does not have a safe first-run ROS2 install recipe for this host yet."
                f"\nDetected host: `{state.detected_facts.get('host_pretty_name', 'unknown')}`."
                "\nThe current guided path supports Ubuntu 22.04/24.04 and WSL2 Ubuntu."
            )
        facts = dict(state.detected_facts)
        facts["ros2_install_recipe"] = recipe.distro
        facts["ros2_install_package"] = recipe.package_name
        facts["ros2_install_profile"] = recipe.profile
        facts["ros2_install_step_index"] = 0
        facts["ros2_install_step_total"] = len(recipe.steps)
        facts.pop("ros2_install_requested", None)
        next_state = replace(
            state,
            stage=SetupStage.INSTALL_PREREQUISITES,
            missing_facts=["guided_ros2_install"],
            detected_facts=facts,
        )
        return next_state, render_ros2_install_step(next_state.detected_facts, recipe)

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
    def _select_serial_device_by_id(output: str) -> str | None:
        for line in output.splitlines():
            candidate = line.strip()
            if candidate.startswith("/dev/serial/by-id/"):
                if "->" in candidate:
                    candidate = candidate.split("->", 1)[0].strip()
                return candidate
        return None

    @staticmethod
    def _normalize_serial_device_by_id(device_path: str) -> str | None:
        candidate = device_path.strip()
        if not candidate:
            return None
        serial_by_id = resolve_serial_by_id_path(candidate)
        if serial_by_id is None:
            return None
        return str(serial_by_id)

    @staticmethod
    def _clear_serial_probe_facts(facts: dict[str, Any]) -> None:
        for key in ("serial_device_by_id", "serial_device_unstable", "serial_device_unresponsive", "serial_probe_error"):
            facts.pop(key, None)

    @classmethod
    def _set_serial_device_by_id(cls, facts: dict[str, Any], serial_by_id: str) -> None:
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_by_id"] = serial_by_id

    @classmethod
    def _set_unstable_serial_device(cls, facts: dict[str, Any]) -> None:
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_unstable"] = True

    @classmethod
    def _set_unresponsive_serial_device(cls, facts: dict[str, Any], detail: str) -> None:
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_unresponsive"] = True
        facts["serial_probe_error"] = detail

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
        namespace = self._ros2_namespace(state)
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
        launch_command = self._launch_command(state)
        launch_line = f"        'launch_command': {launch_command!r}," if launch_command else None
        connection_lines = [
            "        'transport': 'ros2',",
            f"        'ros_distro': {self._resolved_ros2_distro(state)!r},",
            f"        'profile_id': {self._profile_id(state)!r},",
            f"        'namespace': {namespace!r},",
            f"        'serial_device_by_id': {facts.get('serial_device_by_id')!r},",
        ]
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
                f"        generated_at={self._generated_at()!r},",
                "    ),",
                ")",
                "",
                "DEPLOYMENT = DeploymentProfile(",
                f"    id={state.deployment_id!r},",
                f"    assembly_id={state.assembly_id!r},",
                "    target_id='real',",
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

    def _render_adapter(self, state: SetupOnboardingState) -> str:
        implementation = "roboclaw.embodied.execution.integration.adapters.ros2.standard:Ros2ActionServiceAdapter"
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
                "            component=CompatibilityComponent.CONTROL_SURFACE_PROFILE,",
                f"            target={ARM_HAND_CONTROL_SURFACE_PROFILE.id!r},",
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
                f"    control_surface_profile_id={ARM_HAND_CONTROL_SURFACE_PROFILE.id!r},",
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
    def _profile_id(state: SetupOnboardingState) -> str | None:
        profile = OnboardingController._primary_profile(state)
        return profile.id if profile is not None else None

    @staticmethod
    def _primary_profile(state: SetupOnboardingState) -> Any:
        if not state.robot_attachments:
            return None
        primary_robot = state.robot_attachments[0]["robot_id"]
        return get_ros2_profile(primary_robot)

    @staticmethod
    def _resolved_ros2_distro(state: SetupOnboardingState) -> str | None:
        facts = state.detected_facts
        distro = str(facts.get("ros2_distro") or "").strip()
        if distro:
            return distro
        installed = facts.get("ros2_installed_distros")
        if isinstance(installed, list) and installed:
            value = str(installed[0]).strip()
            if value:
                return value
        recipe = select_ros2_recipe(facts)
        if recipe is not None:
            return recipe.distro
        for distro in ("jazzy", "humble", "iron", "rolling", "foxy"):
            if Path(f"/opt/ros/{distro}/setup.bash").exists() or Path(f"/opt/ros/{distro}/setup.zsh").exists():
                return distro
        return None

    @staticmethod
    def _launch_command(state: SetupOnboardingState) -> str | None:
        if not state.robot_attachments:
            return None
        primary_robot = state.robot_attachments[0]["robot_id"]
        profile = OnboardingController._primary_profile(state)
        facts = state.detected_facts
        device_by_id = str(facts.get("serial_device_by_id") or "").strip()
        if profile is None or not device_by_id:
            return None
        namespace = OnboardingController._ros2_namespace(state)
        return profile.control_launch_command(
            namespace=namespace,
            robot_id=primary_robot,
            device_by_id=device_by_id,
        )

    @staticmethod
    def _ros2_namespace(state: SetupOnboardingState) -> str:
        prefix = str(os.environ.get("ROBOCLAW_ROS2_NAMESPACE_PREFIX") or "/roboclaw").strip() or "/roboclaw"
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        prefix = re.sub(r"[^A-Za-z0-9_/]", "_", prefix).rstrip("/") or "/roboclaw"
        return f"{prefix}/{state.assembly_id}/real"

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
