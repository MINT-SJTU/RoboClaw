"""Stage-oriented onboarding flow handler."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from roboclaw.embodied.builtins import (
    list_builtin_embodiments as _list_builtin_embodiments,
    list_supported_robot_labels as _list_supported_robot_labels,
)
from roboclaw.embodied.localization import choose_language, infer_language
from roboclaw.embodied.onboarding.asset_generator import AssetGenerator
from roboclaw.embodied.onboarding.environment_probe import EnvironmentProbe
from roboclaw.embodied.onboarding.helpers import (
    calibration_missing_message,
    calibration_ready_message,
    canonical_ids,
    connection_confirmation_message,
    final_ready_message,
    primary_profile,
    request_robot_message,
    ros2_install_complete_message,
    ros2_missing_message,
    ros2_partial_install_message,
    ros2_recheck_failed_message,
    ros2_waiting_message,
    serial_unresponsive_message,
    simulation_options_message,
    simulation_ready_message,
    unknown_robot_message,
    unstable_serial_message,
    validation_failed_message,
    viewer_mode_question_message,
)
from roboclaw.embodied.onboarding.intent_engine import IntentEngine
from roboclaw.embodied.onboarding.model import OnboardingIntent, SetupOnboardingState, SetupStage, SetupStatus
from roboclaw.embodied.onboarding.ros2_install import (
    advance_ros2_install_step,
    render_ros2_install_step,
    render_ros2_shell_repair,
    ros2_install_summary,
    select_ros2_recipe,
)
from roboclaw.embodied.workspace import WorkspaceInspectOptions, WorkspaceIssueLevel, WorkspaceLintProfile, inspect_workspace_assets
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]
CalibrationStarter = Callable[..., Awaitable[Any]]
StateLoader = Callable[[Session], SetupOnboardingState | None]
BuiltinEmbodiments = Callable[[], tuple[Any, ...] | list[Any]]
SupportedRobotLabels = Callable[[], tuple[str, ...] | list[str]]


class StageHandler:
    """Coordinate the onboarding state machine using extracted submodules."""

    materialize_stage = SetupStage.MATERIALIZE_ASSEMBLY

    def __init__(
        self,
        workspace: Path,
        *,
        intent_engine: IntentEngine,
        asset_generator: AssetGenerator,
        environment_probe: EnvironmentProbe,
        calibration_starter: CalibrationStarter | None = None,
        state_loader: StateLoader | None = None,
        list_builtin_embodiments: BuiltinEmbodiments | None = None,
        list_supported_robot_labels: SupportedRobotLabels | None = None,
    ):
        self.workspace = workspace
        self.intent_engine = intent_engine
        self.asset_generator = asset_generator
        self.environment_probe = environment_probe
        self.calibration_starter = calibration_starter
        self.state_loader = state_loader
        self._list_builtin_embodiments = list_builtin_embodiments or (lambda: tuple(_list_builtin_embodiments()))
        self._list_supported_robot_labels = list_supported_robot_labels or (lambda: tuple(_list_supported_robot_labels()))

    async def advance(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        *,
        intent: OnboardingIntent | None = None,
        preferred_language: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        intent = intent or self.intent_engine.heuristic_intent(content)
        language = choose_language(preferred_language, infer_language(content))
        state = await self.asset_generator.write_intake(state, on_progress=on_progress)

        response = await self.handle_simulation_setup(state, language=language, on_progress=on_progress)
        if response is not None:
            return response

        response = self.handle_robot_identification(state, language=language)
        if response is not None:
            return response

        response = self.handle_connection_confirmation(state, language=language)
        if response is not None:
            return response

        state, response = await self.handle_calibration_flow(
            session,
            state,
            content,
            intent=intent,
            language=language,
            on_progress=on_progress,
        )
        if response is not None:
            return response

        state, response = await self.handle_environment_probing(
            session,
            state,
            content,
            intent=intent,
            language=language,
            on_progress=on_progress,
        )
        if response is not None:
            return response

        state, response = await self.handle_ros2_prerequisites(
            state,
            intent=intent,
            language=language,
            on_progress=on_progress,
        )
        if response is not None:
            return response

        state = await self.handle_asset_materialization(state, on_progress=on_progress)
        state, response = self.handle_validation(state, language=language)
        if response is not None:
            return response

        return self._ready_response(state, language=language)

    async def handle_simulation_setup(
        self,
        state: SetupOnboardingState,
        *,
        language: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any] | None:
        if state.detected_facts.get("simulation_requested") is not True:
            return None
        sim_builtins = tuple(item for item in self._list_builtin_embodiments() if item.sim_model_path)
        selected = next(
            (item for item in sim_builtins if any(robot["robot_id"] == item.robot.id for robot in state.robot_attachments)),
            None,
        )
        if selected is None and len(sim_builtins) == 1:
            robots = [{"attachment_id": "primary", "robot_id": sim_builtins[0].robot.id, "role": "primary"}]
            setup_id, intake_slug, assembly_id, _, _ = canonical_ids(state.setup_id, robots)
            selected = sim_builtins[0]
            state = replace(state, setup_id=setup_id, intake_slug=intake_slug, assembly_id=assembly_id, robot_attachments=robots)
        if selected is None:
            options = ", ".join(item.robot.name for item in sim_builtins) or "none"
            return {
                "state": replace(state, stage=SetupStage.IDENTIFY_SETUP_SCOPE, missing_facts=["simulation_robot"]),
                "content": simulation_options_message(language, options),
            }
        if selected is not None and not state.detected_facts.get("sim_viewer_mode"):
            return {
                "state": replace(state, stage=SetupStage.IDENTIFY_SETUP_SCOPE, missing_facts=["sim_viewer_mode"]),
                "content": viewer_mode_question_message(language),
            }

        state = replace(
            state,
            deployment_id=f"{state.assembly_id}_sim_local",
            adapter_id=f"{state.assembly_id}_sim_ros2",
            execution_targets=[{"id": "sim", "carrier": "sim", "transport": "ros2", "simulator": "mujoco"}],
            detected_facts={
                **state.detected_facts,
                "simulation_requested": True,
                "sim_model_path": selected.sim_model_path,
                "sim_joint_mapping": selected.sim_joint_mapping or {},
                "sim_viewer_mode": state.detected_facts.get("sim_viewer_mode", "web"),
            },
            stage=SetupStage.MATERIALIZE_ASSEMBLY,
            missing_facts=[],
        )
        for writer in (
            self.asset_generator.write_assembly,
            self.asset_generator.write_deployment,
            self.asset_generator.write_adapter,
        ):
            state = await writer(state, on_progress=on_progress)
        issues = self._validation_issues(setup_id=state.setup_id)
        if issues is not None:
            return {
                "state": replace(state, stage=SetupStage.VALIDATE_SETUP),
                "content": validation_failed_message(language, issues, simulation=True),
            }
        return {
            "state": replace(state, stage=SetupStage.HANDOFF_READY, status=SetupStatus.READY),
            "content": simulation_ready_message(language),
        }

    def handle_robot_identification(
        self,
        state: SetupOnboardingState,
        *,
        language: str | None,
    ) -> dict[str, Any] | None:
        if state.robot_attachments:
            profile = primary_profile(state)
            if profile is not None:
                return None
            primary_robot_id = state.robot_attachments[0]["robot_id"]
            supported_models = ", ".join(self._list_supported_robot_labels()) or "none"
            return {
                "state": replace(
                    state,
                    stage=SetupStage.IDENTIFY_SETUP_SCOPE,
                    status=SetupStatus.BOOTSTRAPPING,
                    missing_facts=["control_surface_profile"],
                ),
                "content": unknown_robot_message(language, primary_robot_id, supported_models),
            }

        example_robot = next(iter(self._list_supported_robot_labels()), "a supported robot")
        return {
            "state": replace(
                state,
                stage=SetupStage.IDENTIFY_SETUP_SCOPE,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["robot_or_setup_components"],
            ),
            "content": request_robot_message(language, example_robot),
        }

    def handle_connection_confirmation(
        self,
        state: SetupOnboardingState,
        *,
        language: str | None,
    ) -> dict[str, Any] | None:
        if state.detected_facts.get("connected") is True:
            return None
        return {
            "state": replace(
                state,
                stage=SetupStage.CONFIRM_CONNECTED,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["connected"],
            ),
            "content": connection_confirmation_message(language, state),
        }

    async def handle_calibration_flow(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        *,
        intent: OnboardingIntent,
        language: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[SetupOnboardingState, dict[str, Any] | None]:
        if state.stage != SetupStage.AWAIT_CALIBRATION:
            return state, None

        state = await self.environment_probe.refresh_calibration_facts(state)
        if state.detected_facts.get("calibration_path"):
            state, validation_error = self.environment_probe.validate_materialized_setup(state)
            if validation_error is not None:
                return state, {"state": state, "content": validation_error}
            ready_state = replace(state, stage=SetupStage.HANDOFF_READY, status=SetupStatus.READY, missing_facts=[])
            return ready_state, {"state": ready_state, "content": calibration_ready_message(language, ready_state)}

        state, validation_content = await self.environment_probe.materialize_for_calibration(state, on_progress=on_progress)
        next_state = replace(
            state,
            stage=SetupStage.AWAIT_CALIBRATION,
            status=SetupStatus.BOOTSTRAPPING,
            missing_facts=["calibration_file"],
        )
        if validation_content is not None:
            return next_state, {"state": next_state, "content": validation_content}

        profile = primary_profile(state)
        expected_path = profile.canonical_calibration_path() if profile is not None else None
        if intent.calibration_requested and self.calibration_starter is not None:
            started = await self.calibration_starter(
                session=session,
                action="calibrate",
                setup_id=state.setup_id,
                on_progress=on_progress,
            )
            latest_state = self.state_loader(session) if self.state_loader is not None else None
            latest_state = latest_state or next_state
            return latest_state, {"state": latest_state, "content": started.message}

        return next_state, {"state": next_state, "content": calibration_missing_message(language, profile.robot_id, expected_path)}

    async def handle_environment_probing(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        *,
        intent: OnboardingIntent,
        language: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[SetupOnboardingState, dict[str, Any] | None]:
        if state.stage not in (
            SetupStage.CONFIRM_CONNECTED,
            SetupStage.IDENTIFY_SETUP_SCOPE,
            SetupStage.PROBE_LOCAL_ENVIRONMENT,
        ):
            return state, None

        state = await self.environment_probe.probe_environment(state, on_progress=on_progress)
        profile = primary_profile(state)
        if state.detected_facts.get("serial_device_unresponsive") is True:
            detail = state.detected_facts.get("serial_probe_error", "The registered embodiment probe did not receive a valid status packet.")
            next_state = replace(
                state,
                stage=SetupStage.PROBE_LOCAL_ENVIRONMENT,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["serial_device_by_id"],
            )
            return next_state, {
                "state": next_state,
                "content": serial_unresponsive_message(language, detail),
            }
        if profile is not None and profile.auto_probe_serial and not state.detected_facts.get("serial_device_by_id"):
            next_state = replace(
                state,
                stage=SetupStage.PROBE_LOCAL_ENVIRONMENT,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["serial_device_by_id"],
            )
            return next_state, {
                "state": next_state,
                "content": unstable_serial_message(language),
            }
        if profile is not None and profile.requires_calibration and not str(state.detected_facts.get("calibration_path") or "").strip():
            next_state = replace(
                state,
                stage=SetupStage.AWAIT_CALIBRATION,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["calibration_file"],
            )
            response = await self.advance(
                session,
                next_state,
                content,
                intent=intent,
                preferred_language=language,
                on_progress=on_progress,
            )
            return response["state"], response
        if (
            state.detected_facts.get("ros2_available") is not True
            and state.detected_facts.get("ros2_installed_distros")
            and state.detected_facts.get("ros2_shell_initialized") is True
        ):
            return replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[]), None
        if state.detected_facts.get("ros2_available") is not True:
            state = await self.environment_probe.probe_install_host(state, on_progress=on_progress)
            guide_summary = await self.environment_probe.read_ros2_guide(on_progress=on_progress)
            recipe = select_ros2_recipe(state.detected_facts)
            next_state = replace(
                state,
                stage=SetupStage.RESOLVE_PREREQUISITES,
                status=SetupStatus.BOOTSTRAPPING,
                missing_facts=["ros2_install"],
            )
            if state.detected_facts.get("ros2_installed_distros") and state.detected_facts.get("ros2_shell_initialized") is False:
                content_text = ros2_partial_install_message(
                    language,
                    guide_summary,
                    ros2_install_summary(recipe, state.detected_facts),
                    render_ros2_shell_repair(state.detected_facts, recipe, language=language),
                )
            else:
                content_text = ros2_missing_message(language, guide_summary, ros2_install_summary(recipe, state.detected_facts))
            return next_state, {"state": next_state, "content": content_text}
        return replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[]), None

    async def handle_ros2_prerequisites(
        self,
        state: SetupOnboardingState,
        *,
        intent: OnboardingIntent,
        language: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[SetupOnboardingState, dict[str, Any] | None]:
        if state.stage not in (
            SetupStage.RESOLVE_PREREQUISITES,
            SetupStage.INSTALL_PREREQUISITES,
            SetupStage.VALIDATE_PREREQUISITES,
        ):
            return state, None

        if state.detected_facts.get("ros2_reported_installed"):
            facts = dict(state.detected_facts)
            facts.pop("ros2_reported_installed", None)
            state = replace(state, detected_facts=facts)
            state = await self.environment_probe.probe_environment(state, on_progress=on_progress, force_ros2_probe=True)
            if state.detected_facts.get("ros2_available") is True:
                state = replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[])
            elif state.detected_facts.get("ros2_installed_distros") and state.detected_facts.get("ros2_shell_initialized") is False:
                return state, {"state": state, "content": render_ros2_shell_repair(state.detected_facts, language=language)}
            else:
                return state, {"state": state, "content": ros2_recheck_failed_message(language)}

        if (
            state.stage == SetupStage.RESOLVE_PREREQUISITES
            and state.detected_facts.get("ros2_install_requested")
            and state.detected_facts.get("ros2_available") is not True
        ):
            response = await self.environment_probe.prepare_ros2_install(state, on_progress=on_progress, language=language)
            return response["state"], response

        if state.stage == SetupStage.VALIDATE_PREREQUISITES and state.detected_facts.get("ros2_available") is not True:
            return state, {"state": state, "content": ros2_install_complete_message(language)}

        if state.stage == SetupStage.INSTALL_PREREQUISITES and state.detected_facts.get("ros2_available") is not True:
            if intent.ros2_step_advance or state.detected_facts.get("ros2_step_advance_requested"):
                next_facts, should_validate = advance_ros2_install_step(state.detected_facts)
                next_facts.pop("ros2_step_advance_requested", None)
                state = replace(
                    state,
                    stage=SetupStage.VALIDATE_PREREQUISITES if should_validate else SetupStage.INSTALL_PREREQUISITES,
                    detected_facts=next_facts,
                )
            if state.stage == SetupStage.VALIDATE_PREREQUISITES:
                return state, {"state": state, "content": ros2_install_complete_message(language)}
            return state, {"state": state, "content": render_ros2_install_step(state.detected_facts, language=language)}

        if state.detected_facts.get("ros2_available") is not True:
            return state, {"state": state, "content": ros2_waiting_message(language)}
        return replace(state, stage=SetupStage.MATERIALIZE_ASSEMBLY, missing_facts=[]), None

    async def handle_asset_materialization(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> SetupOnboardingState:
        if state.stage == SetupStage.MATERIALIZE_ASSEMBLY:
            state = await self.asset_generator.write_assembly(state, on_progress=on_progress)
            state = replace(state, stage=SetupStage.MATERIALIZE_DEPLOYMENT_ADAPTER)
        if state.stage == SetupStage.MATERIALIZE_DEPLOYMENT_ADAPTER:
            state = await self.asset_generator.write_deployment(state, on_progress=on_progress)
            state = await self.asset_generator.write_adapter(state, on_progress=on_progress)
            state = replace(state, stage=SetupStage.VALIDATE_SETUP)
        return state

    def handle_validation(
        self,
        state: SetupOnboardingState,
        *,
        language: str | None,
    ) -> tuple[SetupOnboardingState, dict[str, Any] | None]:
        if state.stage != SetupStage.VALIDATE_SETUP:
            return state, None
        issues = self._validation_issues(setup_id=state.setup_id)
        if issues is not None:
            return state, {"state": state, "content": validation_failed_message(language, issues)}
        return replace(state, stage=SetupStage.HANDOFF_READY, status=SetupStatus.READY), None

    def _ready_response(self, state: SetupOnboardingState, *, language: str | None) -> dict[str, Any]:
        return {"state": state, "content": final_ready_message(language, state)}

    def _validation_issues(self, setup_id: str | None = None) -> str | None:
        validation = inspect_workspace_assets(
            self.workspace,
            options=WorkspaceInspectOptions(lint_profile=WorkspaceLintProfile.BASIC),
        )
        if not validation.has_errors:
            return None
        # Only report errors (not warnings) related to this setup's assets
        issues = [i for i in validation.issues if i.level == WorkspaceIssueLevel.ERROR]
        if setup_id:
            issues = [i for i in issues if setup_id in i.path]
        if not issues:
            return None
        return "\n".join(f"- {issue.path}: {issue.message}" for issue in issues[:5])
