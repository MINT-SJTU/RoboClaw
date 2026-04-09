"""Board package — per-embodiment unified state + pub/sub hub (看板)."""

from roboclaw.embodied.board.board import Board, Subscriber
from roboclaw.embodied.board.channels import (
    CH_CALIBRATION,
    CH_CONFIG,
    CH_FAULT_DETECTED,
    CH_FAULT_RESOLVED,
    CH_SESSION,
    WS_TYPES,
)
from roboclaw.embodied.board.constants import Command, EpisodePhase, SessionState
from roboclaw.embodied.board.consumer import Consumer, InputConsumer, OutputConsumer

__all__ = [
    "Board",
    "CH_CALIBRATION",
    "CH_CONFIG",
    "CH_FAULT_DETECTED",
    "CH_FAULT_RESOLVED",
    "CH_SESSION",
    "Command",
    "Consumer",
    "EpisodePhase",
    "InputConsumer",
    "OutputConsumer",
    "SessionState",
    "Subscriber",
    "WS_TYPES",
]
