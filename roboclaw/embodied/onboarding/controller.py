"""Assembly-centered setup onboarding controller."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.builtins import list_builtin_embodiments, list_builtin_robot_aliases, list_supported_robot_labels
from roboclaw.embodied.intent import IntentClassifier, UserIntent
from roboclaw.embodied.localization import choose_language, infer_language
from roboclaw.embodied.onboarding.asset_generator import AssetGenerator
from roboclaw.embodied.onboarding.environment_probe import EnvironmentProbe
from roboclaw.embodied.onboarding.helpers import (
    canonical_ids,
    select_serial_device_by_id,
    set_serial_device_by_id,
    set_unresponsive_serial_device,
    set_unstable_serial_device,
)
from roboclaw.embodied.onboarding.intent_engine import IntentEngine
from roboclaw.embodied.onboarding.model import (
    PREFERRED_LANGUAGE_KEY,
    SETUP_STATE_KEY,
    OnboardingIntent,
    SetupOnboardingState,
    SetupStatus,
)
from roboclaw.embodied.onboarding.stage_handler import StageHandler
from roboclaw.config.paths import resolve_serial_by_id_path
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]
IntentParser = Callable[[Session, SetupOnboardingState, str], Awaitable[OnboardingIntent | None]]
CalibrationStarter = Callable[..., Awaitable[Any]]


class OnboardingController:
    """Handle first-run embodied setup and later setup refinements."""

    def __init__(
        self,
        workspace: Path,
        tools: ToolRegistry,
        *,
        intent_parser: IntentParser | None = None,
        calibration_starter: CalibrationStarter | None = None,
    ):
        self.workspace = workspace
        self.tools = tools
        self.intent_parser = intent_parser
        self.calibration_starter = calibration_starter
        self._intent_classifier = IntentClassifier(
            llm_caller=None,
            known_robots=tuple(robot_id for robot_id in list_builtin_robot_aliases()),
            robot_aliases=dict(list_builtin_robot_aliases()),
        )
        self.intent_engine = IntentEngine(
            self._intent_classifier,
            dict(list_builtin_robot_aliases()),
            intent_parser=intent_parser,
        )
        self.asset_generator = AssetGenerator(workspace, tools)
        self.environment_probe = EnvironmentProbe(
            workspace,
            tools,
            tool_runner=self.asset_generator.run_tool,
            write_assembly=self.asset_generator.write_assembly,
            write_deployment=self.asset_generator.write_deployment,
            write_adapter=self.asset_generator.write_adapter,
        )
        self.stage_handler = StageHandler(
            workspace,
            intent_engine=self.intent_engine,
            asset_generator=self.asset_generator,
            environment_probe=self.environment_probe,
            calibration_starter=calibration_starter,
            state_loader=self._load_state,
            list_builtin_embodiments=lambda: list_builtin_embodiments(),
            list_supported_robot_labels=lambda: tuple(list_supported_robot_labels()),
        )
        self.catalog = self.intent_engine.catalog
        self._intent_cache = self.intent_engine._intent_cache

    def set_llm_caller(self, llm_caller) -> None:
        self._intent_classifier._llm = llm_caller
        self._intent_cache.clear()

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

        classified_intent = await self._classify(msg.content, context=self._intent_context(state))
        intent = await self._resolve_intent(session, state, msg.content, user_intent=classified_intent)
        preferred_language = choose_language(
            intent.preferred_language if intent is not None else None,
            session.metadata.get(PREFERRED_LANGUAGE_KEY),
            infer_language(msg.content),
        )
        session.metadata[PREFERRED_LANGUAGE_KEY] = preferred_language

        state, changed = self._apply_user_input(state, msg.content, intent=intent)
        response = await self._advance(
            session,
            state,
            msg.content,
            intent=intent,
            preferred_language=preferred_language,
            on_progress=on_progress,
        )

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

    @staticmethod
    def _intent_context(state: SetupOnboardingState) -> str:
        robot_ids = ",".join(item["robot_id"] for item in state.robot_attachments) or "none"
        facts = ",".join(sorted(state.detected_facts)) or "none"
        return f"stage={state.stage.value}; status={state.status.value}; robots={robot_ids}; facts={facts}"

    async def _classify(self, content: str, *, context: str = "") -> UserIntent:
        return await self.intent_engine.classify(content, context=context)

    def _cached_intent(self, content: str) -> UserIntent | None:
        return self.intent_engine.cached_intent(content)

    def _looks_like_setup_start(self, content: str) -> bool:
        return self.intent_engine.looks_like_setup_start(content)

    def _looks_like_sim_request(self, content: str) -> bool:
        return self.intent_engine.looks_like_sim_request(content)

    def _looks_like_setup_edit(self, content: str) -> bool:
        return self.intent_engine.looks_like_setup_edit(content)

    async def _resolve_intent(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        *,
        user_intent: UserIntent | None = None,
    ) -> OnboardingIntent:
        return await self.intent_engine.resolve_intent(session, state, content, user_intent=user_intent)

    def _apply_user_input(
        self,
        state: SetupOnboardingState,
        content: str,
        *,
        intent: OnboardingIntent | None = None,
    ) -> tuple[SetupOnboardingState, bool]:
        intent = intent or self.intent_engine.heuristic_intent(content)
        changed = False
        robots = list(state.robot_attachments)
        sensors = list(state.sensor_attachments)
        facts = dict(state.detected_facts)

        for robot_id in intent.robot_ids:
            if not any(item["robot_id"] == robot_id for item in robots):
                attachment_id = "primary" if not robots else f"robot_{len(robots) + 1}"
                robots.append({"attachment_id": attachment_id, "robot_id": robot_id, "role": "primary" if not robots else "secondary"})
                changed = True

        for sensor_change in intent.sensor_changes:
            sensors, sensor_changed = self.intent_engine._apply_sensor_change(sensors, sensor_change)
            changed = changed or sensor_changed

        if intent.connected is not None and facts.get("connected") != intent.connected:
            facts["connected"] = intent.connected
            changed = True

        if intent.serial_path:
            serial_by_id = self._normalize_serial_device_by_id(intent.serial_path)
            if serial_by_id is not None and facts.get("serial_device_by_id") != serial_by_id:
                set_serial_device_by_id(facts, serial_by_id)
                changed = True
            elif serial_by_id is None:
                if facts.get("serial_device_unstable") is not True or facts.get("serial_device_unresponsive") is True:
                    set_unstable_serial_device(facts)
                    changed = True

        if intent.ros2_install_profile and facts.get("ros2_install_profile") != intent.ros2_install_profile:
            facts["ros2_install_profile"] = intent.ros2_install_profile
            changed = True

        if intent.ros2_state is False and facts.get("ros2_available") is not False:
            facts["ros2_available"] = False
            changed = True
        if intent.ros2_state is True:
            facts["ros2_reported_installed"] = True
            changed = True
        if intent.ros2_install_requested:
            facts["ros2_install_requested"] = True
            changed = True
        if intent.ros2_step_advance:
            facts["ros2_step_advance_requested"] = True
            changed = True
        if intent.simulation_requested and facts.get("simulation_requested") is not True:
            facts["simulation_requested"] = True
            changed = True

        next_status = state.status
        next_stage = state.stage
        if state.is_ready and (intent.sensor_changes or intent.serial_path or intent.ros2_state is not None):
            next_status = SetupStatus.REFINING
            next_stage = self.stage_handler.materialize_stage

        setup_id, intake_slug, assembly_id, deployment_id, adapter_id = canonical_ids(state.setup_id, robots)
        return (
            replace(
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
            ),
            changed,
        )

    def _extract_robot_ids(self, content: str) -> list[str]:
        return self.intent_engine.extract_robot_ids(content)

    def _extract_sensor_changes(self, content: str) -> list[dict[str, Any]]:
        return self.intent_engine.extract_sensor_changes(content)

    def _extract_connected_state(self, content: str) -> bool | None:
        return self.intent_engine.extract_connected_state(content)

    def _extract_calibration_request(self, content: str) -> bool:
        return self.intent_engine.extract_calibration_request(content)

    def _extract_serial_path(self, content: str) -> str | None:
        return self.intent_engine.extract_serial_path(content)

    async def _advance(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        *,
        intent: OnboardingIntent | None = None,
        preferred_language: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        return await self.stage_handler.advance(
            session,
            state,
            content,
            intent=intent,
            preferred_language=preferred_language,
            on_progress=on_progress,
        )

    async def _write_intake(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> SetupOnboardingState:
        return await self.asset_generator.write_intake(state, on_progress=on_progress)

    async def _write_assembly(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> SetupOnboardingState:
        return await self.asset_generator.write_assembly(state, on_progress=on_progress)

    async def _write_deployment(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> SetupOnboardingState:
        return await self.asset_generator.write_deployment(state, on_progress=on_progress)

    async def _write_adapter(
        self,
        state: SetupOnboardingState,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> SetupOnboardingState:
        return await self.asset_generator.write_adapter(state, on_progress=on_progress)

    async def _run_tool(
        self,
        name: str,
        params: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        return await self.asset_generator.run_tool(name, params, on_progress=on_progress)

    @staticmethod
    def _select_serial_device_by_id(output: str) -> str | None:
        return select_serial_device_by_id(output)

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
    def _set_serial_device_by_id(facts: dict[str, Any], serial_by_id: str) -> None:
        set_serial_device_by_id(facts, serial_by_id)

    @staticmethod
    def _set_unstable_serial_device(facts: dict[str, Any]) -> None:
        set_unstable_serial_device(facts)

    @staticmethod
    def _set_unresponsive_serial_device(facts: dict[str, Any], detail: str) -> None:
        set_unresponsive_serial_device(facts, detail)
