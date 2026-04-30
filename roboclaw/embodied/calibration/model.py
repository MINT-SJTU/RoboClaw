from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CalibrationStatus = Literal["pending", "running", "success", "skipped", "failed"]


@dataclass(frozen=True)
class MotorCalibrationProfile:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorCalibrationProfile:
        return cls(
            id=int(data["id"]),
            drive_mode=int(data.get("drive_mode", 0)),
            homing_offset=int(data.get("homing_offset", 0)),
            range_min=int(data["range_min"]),
            range_max=int(data["range_max"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "id": self.id,
            "drive_mode": self.drive_mode,
            "homing_offset": self.homing_offset,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }


@dataclass(frozen=True)
class CalibrationProfile:
    motors: dict[str, MotorCalibrationProfile]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        return cls({
            motor: MotorCalibrationProfile.from_dict(config)
            for motor, config in data.items()
        })

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            motor: profile.to_dict()
            for motor, profile in self.motors.items()
        }


@dataclass(frozen=True)
class CalibrationBatchResult:
    alias: str
    status: CalibrationStatus
    reason: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "status": self.status,
            "reason": self.reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
