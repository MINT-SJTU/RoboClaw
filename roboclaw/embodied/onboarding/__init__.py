"""Embodied onboarding controller entrypoints."""

from roboclaw.embodied.onboarding.controller import OnboardingController
from roboclaw.embodied.onboarding.model import SETUP_STATE_KEY, SetupOnboardingState, SetupStage, SetupStatus

__all__ = [
    "OnboardingController",
    "SETUP_STATE_KEY",
    "SetupOnboardingState",
    "SetupStage",
    "SetupStatus",
]
