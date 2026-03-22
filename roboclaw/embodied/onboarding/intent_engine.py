"""Intent classification and input extraction for onboarding."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from loguru import logger

from roboclaw.embodied.catalog import build_default_catalog
from roboclaw.embodied.intent import IntentClassifier, UserIntent
from roboclaw.embodied.localization import infer_language
from roboclaw.embodied.onboarding.model import OnboardingIntent, SetupOnboardingState
from roboclaw.embodied.onboarding.ros2_install import (
    extract_ros2_profile,
    extract_ros2_state,
    is_ros2_install_request,
    is_ros2_step_advance,
)
from roboclaw.session.manager import Session

IntentParser = Callable[[Session, SetupOnboardingState, str], Awaitable[OnboardingIntent | None]]


class IntentEngine:
    """Handle onboarding intent classification and message extraction."""

    _SETUP_START_KEYWORDS = (
        "connect", "real robot", "setup", "onboard",
        "真实机器人", "真实的机器人", "机械臂", "机器人",
    )
    _SIM_KEYWORDS = ("simulation", "no robot", "try simulation", "virtual robot", "仿真", "没有机器人", "试试仿真", "虚拟机器人")
    _SETUP_EDIT_KEYWORDS = (
        "camera", "sensor", "serial", "/dev/", "ros2", "deployment", "adapter", "installed", "replace",
    )
    _SERIAL_RE = re.compile(r"(/dev/[^\s,;]+)")

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        robot_aliases: dict[str, tuple[str, ...]] | dict[str, list[str]],
        *,
        intent_parser: IntentParser | None = None,
    ):
        self._intent_classifier = intent_classifier
        self._robot_aliases = {
            robot_id: tuple(str(alias).lower() for alias in aliases)
            for robot_id, aliases in robot_aliases.items()
        }
        self.catalog = build_default_catalog()
        self.intent_parser = intent_parser
        self._intent_cache: dict[str, UserIntent] = {}

    async def classify(self, content: str, context: str = "") -> UserIntent:
        if content not in self._intent_cache:
            self._intent_cache[content] = await self._intent_classifier.classify(content, context=context)
        return self._intent_cache[content]

    def cached_intent(self, content: str) -> UserIntent | None:
        if self._intent_classifier._llm is None:
            return None
        return self._intent_cache.get(content)

    def looks_like_setup_start(self, content: str) -> bool:
        intent = self.cached_intent(content)
        if intent is not None:
            return intent.wants_setup or intent.is_embodied
        return self._intent_classifier._keyword_fallback(content).wants_setup

    def looks_like_sim_request(self, content: str) -> bool:
        intent = self.cached_intent(content)
        if intent is not None:
            return intent.wants_simulation
        return self._intent_classifier._keyword_fallback(content).wants_simulation

    def looks_like_setup_edit(self, content: str) -> bool:
        intent = self.cached_intent(content)
        if intent is not None:
            return intent.wants_edit
        return self._intent_classifier._keyword_fallback(content).wants_edit

    async def resolve_intent(
        self,
        session: Session,
        state: SetupOnboardingState,
        content: str,
        user_intent: UserIntent | None = None,
    ) -> OnboardingIntent:
        heuristic = self.heuristic_intent(content, user_intent=user_intent)
        if self.intent_parser is None or self.intent_has_signal(heuristic):
            return heuristic
        try:
            parsed = await self.intent_parser(session, state, content)
        except Exception:
            logger.exception("Failed to parse onboarding intent for session {}", session.key)
            parsed = None
        return self.merge_intents(heuristic, parsed)

    @staticmethod
    def intent_has_signal(intent: OnboardingIntent) -> bool:
        return any(
            (
                intent.robot_ids,
                intent.simulation_requested,
                intent.sensor_changes,
                intent.connected is not None,
                intent.serial_path,
                intent.ros2_install_profile,
                intent.ros2_state is not None,
                intent.ros2_install_requested,
                intent.ros2_step_advance,
                intent.calibration_requested,
            )
        )

    def heuristic_intent(
        self,
        content: str,
        user_intent: UserIntent | None = None,
    ) -> OnboardingIntent:
        use_classified_intent = user_intent is not None and self._intent_classifier._llm is not None
        inferred_language = infer_language(content)
        return OnboardingIntent(
            robot_ids=((user_intent.robot_id,) if use_classified_intent and user_intent.robot_id else tuple(self.extract_robot_ids(content))),
            simulation_requested=user_intent.wants_simulation if use_classified_intent else self.looks_like_sim_request(content),
            sensor_changes=tuple(self.extract_sensor_changes(content)),
            connected=user_intent.connection_confirmed if use_classified_intent else self.extract_connected_state(content),
            serial_path=user_intent.serial_path if use_classified_intent and user_intent.serial_path else self.extract_serial_path(content),
            ros2_install_profile=extract_ros2_profile(content),
            ros2_state=extract_ros2_state(content),
            ros2_install_requested=is_ros2_install_request(content),
            ros2_step_advance=is_ros2_step_advance(content),
            calibration_requested=user_intent.wants_calibration if use_classified_intent else self.extract_calibration_request(content),
            preferred_language="zh" if inferred_language == "zh" else None,
        )

    @staticmethod
    def merge_intents(primary: OnboardingIntent, secondary: OnboardingIntent | None) -> OnboardingIntent:
        if secondary is None:
            return primary
        return OnboardingIntent(
            robot_ids=secondary.robot_ids or primary.robot_ids,
            simulation_requested=secondary.simulation_requested or primary.simulation_requested,
            sensor_changes=secondary.sensor_changes or primary.sensor_changes,
            connected=secondary.connected if secondary.connected is not None else primary.connected,
            serial_path=secondary.serial_path or primary.serial_path,
            ros2_install_profile=secondary.ros2_install_profile or primary.ros2_install_profile,
            ros2_state=secondary.ros2_state if secondary.ros2_state is not None else primary.ros2_state,
            ros2_install_requested=secondary.ros2_install_requested or primary.ros2_install_requested,
            ros2_step_advance=secondary.ros2_step_advance or primary.ros2_step_advance,
            calibration_requested=secondary.calibration_requested or primary.calibration_requested,
            preferred_language=secondary.preferred_language or primary.preferred_language,
        )

    def extract_robot_ids(self, content: str) -> list[str]:
        intent = self.cached_intent(content)
        if intent is not None and intent.robot_id:
            return [intent.robot_id]
        normalized = content.lower()
        matched: list[str] = []
        for robot_id, aliases in self._robot_aliases.items():
            if any(alias in normalized for alias in aliases):
                matched.append(robot_id)
        for manifest in self.catalog.robots.list():
            if manifest.id in normalized and manifest.id not in matched:
                matched.append(manifest.id)
        return matched

    def extract_sensor_changes(self, content: str) -> list[dict[str, Any]]:
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

    def extract_connected_state(self, content: str) -> bool | None:
        intent = self.cached_intent(content)
        if intent is not None:
            return intent.connection_confirmed
        return self._intent_classifier._keyword_fallback(content).connection_confirmed

    def extract_calibration_request(self, content: str) -> bool:
        intent = self.cached_intent(content)
        if intent is not None:
            return intent.wants_calibration
        return self._intent_classifier._keyword_fallback(content).wants_calibration

    def extract_serial_path(self, content: str) -> str | None:
        match = self._SERIAL_RE.search(content)
        return match.group(1) if match else None
