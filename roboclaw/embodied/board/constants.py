"""Constants for session state and commands."""

from enum import StrEnum


class SessionState(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    CALIBRATING = "calibrating"
    TELEOPERATING = "teleoperating"
    RECORDING = "recording"
    REPLAYING = "replaying"
    INFERRING = "inferring"
    STOPPING = "stopping"
    ERROR = "error"


class Command(StrEnum):
    SAVE_EPISODE = "save_episode"
    DISCARD_EPISODE = "discard_episode"
    SKIP_RESET = "skip_reset"
    STOP = "stop"
    CONFIRM = "confirm"
    RECALIBRATE = "recalibrate"
