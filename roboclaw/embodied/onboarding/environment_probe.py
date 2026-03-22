"""Environment probing helpers for onboarding."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.embodied.builtins import get_builtin_probe_provider
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import get_ros2_profile
from roboclaw.embodied.localization import localize_text
from roboclaw.embodied.onboarding.model import SetupOnboardingState, SetupStage
from roboclaw.embodied.onboarding.ros2_install import (
    parse_key_value_output,
    render_ros2_install_step,
    select_ros2_recipe,
)
from roboclaw.embodied.probes import ToolRunner
from roboclaw.embodied.workspace import WorkspaceInspectOptions, WorkspaceIssueLevel, WorkspaceLintProfile, inspect_workspace_assets

try:
    from roboclaw.embodied.onboarding.asset_generator import AssetGenerator
except Exception:  # pragma: no cover - fallback for partial extraction states.
    AssetGenerator = None

try:
    from roboclaw.embodied.onboarding.helpers import (
        extend_unique as _extend_unique_helper,
        normalize_serial_device_by_id as _normalize_serial_device_by_id_helper,
        primary_profile as _primary_profile_helper,
        select_serial_device_by_id as _select_serial_device_by_id_helper,
        set_serial_device_by_id as _set_serial_device_by_id_helper,
        set_unresponsive_serial_device as _set_unresponsive_serial_device_helper,
        set_unstable_serial_device as _set_unstable_serial_device_helper,
    )
except Exception:  # pragma: no cover - fallback for partial extraction states.
    _extend_unique_helper = None
    _normalize_serial_device_by_id_helper = None
    _primary_profile_helper = None
    _select_serial_device_by_id_helper = None
    _set_serial_device_by_id_helper = None
    _set_unresponsive_serial_device_helper = None
    _set_unstable_serial_device_helper = None

ProgressCallback = Callable[[str], Awaitable[None]]
ProgressStateWriter = Callable[[SetupOnboardingState, Callable[..., Awaitable[None]] | None], Awaitable[SetupOnboardingState]]


class EnvironmentProbe:
    """Handle onboarding environment detection and setup validation."""

    def __init__(
        self,
        workspace: Path,
        tool_registry,
        *,
        tool_runner: ToolRunner | None = None,
        write_assembly: ProgressStateWriter | None = None,
        write_deployment: ProgressStateWriter | None = None,
        write_adapter: ProgressStateWriter | None = None,
    ):
        self.workspace = workspace
        self.tools = tool_registry
        self._tool_runner = tool_runner
        self._asset_generator = AssetGenerator(workspace, tool_registry) if AssetGenerator is not None else None
        self._write_assembly = write_assembly or (self._asset_generator.write_assembly if self._asset_generator is not None else None)
        self._write_deployment = write_deployment or (self._asset_generator.write_deployment if self._asset_generator is not None else None)
        self._write_adapter = write_adapter or (self._asset_generator.write_adapter if self._asset_generator is not None else None)

    async def probe_environment(
        self,
        state: SetupOnboardingState,
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
                serial_check = await self.probe_serial_device(primary_profile.probe_provider_id, serial_by_id, on_progress=on_progress)
                if serial_check["ok"]:
                    self._set_serial_device_by_id(facts, serial_by_id)
                else:
                    self._set_unresponsive_serial_device(facts, serial_check["detail"])
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

    async def probe_serial_device(
        self,
        probe_provider_id: str | None,
        serial_by_id: str,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        provider = get_builtin_probe_provider(probe_provider_id)
        if provider is None:
            return {"ok": False, "detail": "No probe provider is registered for this embodiment."}
        result = await provider.probe_serial_device(
            serial_by_id,
            run_tool=self._run_tool,
            on_progress=on_progress,
        )
        return {"ok": result.ok, "detail": result.detail}

    async def read_ros2_guide(self, on_progress: Callable[..., Awaitable[None]] | None = None) -> str:
        guide_path = self.workspace / "embodied" / "guides" / "ROS2_INSTALL.md"
        content = await self._run_tool("read_file", {"path": str(guide_path)}, on_progress=on_progress)
        if content.startswith("Error"):
            return str(guide_path)
        first_heading = next((line[2:].strip() for line in content.splitlines() if line.startswith("## ")), None)
        return f"{guide_path.name}{f' / {first_heading}' if first_heading else ''}"

    async def probe_install_host(
        self,
        state: SetupOnboardingState,
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

    async def refresh_calibration_facts(self, state: SetupOnboardingState) -> SetupOnboardingState:
        facts = dict(state.detected_facts)
        primary_profile = self._primary_profile(state)
        if primary_profile is None or not getattr(primary_profile, "requires_calibration", False):
            return state
        calibration_path = primary_profile.ensure_canonical_calibration()
        if calibration_path is not None and calibration_path.exists():
            facts["calibration_path"] = str(calibration_path)
            facts.pop("calibration_missing", None)
        else:
            facts.pop("calibration_path", None)
            facts["calibration_missing"] = True
        return replace(state, detected_facts=facts)

    async def materialize_for_calibration(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[SetupOnboardingState, str | None]:
        """Write setup assets early so execution can resolve the setup for calibration."""
        if self._write_assembly is None or self._write_deployment is None or self._write_adapter is None:
            return state, "The setup cannot be materialized for calibration because asset writers are not configured."
        state = await self._write_assembly(state, on_progress=on_progress)
        state = await self._write_deployment(state, on_progress=on_progress)
        state = await self._write_adapter(state, on_progress=on_progress)
        return self.validate_materialized_setup(state)

    def validate_materialized_setup(
        self,
        state: SetupOnboardingState,
    ) -> tuple[SetupOnboardingState, str | None]:
        """Validate already-written setup assets for handoff into execution."""
        validation = inspect_workspace_assets(
            self.workspace,
            options=WorkspaceInspectOptions(lint_profile=WorkspaceLintProfile.BASIC),
        )
        if validation.has_errors:
            # Only block on errors (not warnings) related to this setup's assets
            setup_prefix = state.setup_id or ""
            errors_only = [i for i in validation.issues if i.level == WorkspaceIssueLevel.ERROR]
            relevant = [i for i in errors_only if setup_prefix and setup_prefix in i.path] if setup_prefix else errors_only
            if relevant:
                issues = "\n".join(f"- {issue.path}: {issue.message}" for issue in relevant[:5])
                return (
                    state,
                    "The setup assets were written for calibration handoff, but validation is still failing:\n"
                    f"{issues}",
                )
        return state, None

    async def prepare_ros2_install(
        self,
        state: SetupOnboardingState,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        state = await self.probe_install_host(state, on_progress=on_progress)
        recipe = select_ros2_recipe(state.detected_facts)
        if recipe is None:
            next_state = replace(
                state,
                stage=SetupStage.RESOLVE_PREREQUISITES,
                missing_facts=["supported_ros2_host"],
            )
            return {
                "state": next_state,
                "content": localize_text(
                    language,
                    en=(
                        "RoboClaw does not have a safe first-run ROS2 install recipe for this host yet."
                        f"\nDetected host: `{state.detected_facts.get('host_pretty_name', 'unknown')}`."
                        "\nThe current guided path supports Ubuntu 22.04/24.04 and WSL2 Ubuntu."
                    ),
                    zh=(
                        "RoboClaw 目前还没有为这台机器提供安全的首次 ROS2 安装配方。"
                        f"\n检测到的宿主机：`{state.detected_facts.get('host_pretty_name', 'unknown')}`。"
                        "\n当前引导流程支持 Ubuntu 22.04/24.04，以及 WSL2 里的 Ubuntu。"
                    ),
                ),
            }
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
        return {
            "state": next_state,
            "content": render_ros2_install_step(next_state.detected_facts, recipe, language=language),
        }

    async def _run_tool(
        self,
        name: str,
        params: dict[str, Any],
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        if self._tool_runner is not None:
            return await self._tool_runner(name, params, on_progress)
        if self._asset_generator is not None:
            return await self._asset_generator.run_tool(name, params, on_progress=on_progress)
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
        if _select_serial_device_by_id_helper is not None:
            return _select_serial_device_by_id_helper(output)
        for line in output.splitlines():
            candidate = line.strip()
            if candidate.startswith("/dev/serial/by-id/"):
                if "->" in candidate:
                    candidate = candidate.split("->", 1)[0].strip()
                return candidate
        return None

    @staticmethod
    def _normalize_serial_device_by_id(device_path: str) -> str | None:
        if _normalize_serial_device_by_id_helper is not None:
            return _normalize_serial_device_by_id_helper(device_path)
        from roboclaw.config.paths import resolve_serial_by_id_path

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
        if _set_serial_device_by_id_helper is not None:
            _set_serial_device_by_id_helper(facts, serial_by_id)
            return
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_by_id"] = serial_by_id

    @classmethod
    def _set_unstable_serial_device(cls, facts: dict[str, Any]) -> None:
        if _set_unstable_serial_device_helper is not None:
            _set_unstable_serial_device_helper(facts)
            return
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_unstable"] = True

    @classmethod
    def _set_unresponsive_serial_device(cls, facts: dict[str, Any], detail: str) -> None:
        if _set_unresponsive_serial_device_helper is not None:
            _set_unresponsive_serial_device_helper(facts, detail)
            return
        cls._clear_serial_probe_facts(facts)
        facts["serial_device_unresponsive"] = True
        facts["serial_probe_error"] = detail

    @staticmethod
    def _extend_unique(items: list[str], value: str) -> list[str]:
        if _extend_unique_helper is not None:
            return _extend_unique_helper(items, value)
        if value not in items:
            items.append(value)
        return items

    @staticmethod
    def _primary_profile(state: SetupOnboardingState) -> Any:
        if _primary_profile_helper is not None:
            return _primary_profile_helper(state)
        if not state.robot_attachments:
            return None
        primary_robot = state.robot_attachments[0]["robot_id"]
        return get_ros2_profile(primary_robot)
