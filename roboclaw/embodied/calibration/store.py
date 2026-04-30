from __future__ import annotations

import json
from pathlib import Path

from roboclaw.embodied.calibration.model import CalibrationProfile
from roboclaw.embodied.embodiment.manifest.binding import ArmBinding


class CalibrationStore:
    def profile_path(self, arm: ArmBinding) -> Path:
        calibration_dir = Path(arm.calibration_dir).expanduser()
        if not arm.calibration_dir or not arm.arm_id:
            raise RuntimeError(f"Arm '{arm.alias}' has no calibration path.")
        return calibration_dir / f"{arm.arm_id}.json"

    def has_profile(self, arm: ArmBinding) -> bool:
        return self.profile_path(arm).exists()

    def load_profile(self, arm: ArmBinding) -> CalibrationProfile:
        path = self.profile_path(arm)
        if not path.exists():
            raise RuntimeError(f"Calibration profile missing for {arm.alias}.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationProfile.from_dict(data)

    def save_profile(self, arm: ArmBinding, profile: CalibrationProfile) -> Path:
        path = self.profile_path(arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile.to_dict(), indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
