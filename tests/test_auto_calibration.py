from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from roboclaw.embodied.board import Board
from roboclaw.embodied.calibration.batch import AutoCalibrationBatch
from roboclaw.embodied.calibration.model import CalibrationProfile, MotorCalibrationProfile
from roboclaw.embodied.calibration.so101.auto import (
    AutoCalibrationStopped,
    ProbeResult,
    SO101AutoCalibrationStrategy,
    _SO101AutoCalibrator,
)
from roboclaw.embodied.calibration.store import CalibrationStore
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.binding import load_binding
from roboclaw.embodied.embodiment.manifest.helpers import save_manifest


class DummyManifest:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_arm_calibrated(self, alias: str) -> None:
        self.marked.append(alias)


class FastStrategy:
    def recalibrate(self, arm: object, store: CalibrationStore, *, stop_event=None) -> CalibrationProfile:
        return store.load_profile(arm)


class SlowStrategy:
    def recalibrate(self, arm: object, store: CalibrationStore, *, stop_event=None) -> CalibrationProfile:
        while stop_event is not None and not stop_event.is_set():
            time.sleep(0.02)
        raise AutoCalibrationStopped("stopped")


class ParallelStrategy:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def recalibrate(self, arm: object, store: CalibrationStore, *, stop_event=None) -> CalibrationProfile:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return store.load_profile(arm)
        finally:
            with self._lock:
                self.active -= 1


def _build_manifest(tmp_path: Path) -> tuple[Manifest, Path]:
    cal_dir = tmp_path / "calibration" / "SIM001"
    cal_dir.mkdir(parents=True)
    port = tmp_path / "tty.sim001"
    port.write_text("", encoding="utf-8")
    path = tmp_path / "manifest.json"
    save_manifest(
        {
            "version": 2,
            "arms": [
                {
                    "alias": "test_arm",
                    "type": "so101_follower",
                    "port": str(port),
                    "calibration_dir": str(cal_dir),
                    "calibrated": True,
                }
            ],
            "hands": [],
            "cameras": [],
            "datasets": {"root": "/data"},
            "policies": {"root": "/policies"},
        },
        path,
    )
    return Manifest(path=path), cal_dir


def _baseline_profile() -> dict[str, dict[str, int]]:
    return {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1276, "range_min": 974, "range_max": 3369},
        "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1609, "range_min": 820, "range_max": 3231},
        "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 1951, "range_min": 910, "range_max": 3122},
        "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1413, "range_min": 920, "range_max": 3228},
        "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": -1110, "range_min": 0, "range_max": 4095},
        "gripper": {"id": 6, "drive_mode": 0, "homing_offset": -528, "range_min": 1532, "range_max": 2772},
    }


def _write_profile(cal_dir: Path, data: dict[str, dict[str, int]]) -> None:
    (cal_dir / f"{cal_dir.name}.json").write_text(
        json.dumps(data, indent=4) + "\n",
        encoding="utf-8",
    )


def _make_arm(tmp_path: Path, *, alias: str, arm_type: str, connected: bool, with_profile: bool) -> object:
    cal_dir = tmp_path / "calibration" / alias
    cal_dir.mkdir(parents=True, exist_ok=True)
    if with_profile:
        _write_profile(cal_dir, _baseline_profile())
    port = tmp_path / f"{alias}.tty"
    if connected:
        port.write_text("", encoding="utf-8")
    return load_binding({
        "alias": alias,
        "type": arm_type,
        "port": str(port),
        "calibration_dir": str(cal_dir),
        "calibrated": with_profile,
    }, "arm", {})


def test_so101_strategy_builds_profile_from_probe_results_without_baseline(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    arm = manifest.find_arm("test_arm")
    strategy = SO101AutoCalibrationStrategy()
    store = CalibrationStore()

    with patch("roboclaw.embodied.calibration.so101.auto._SO101AutoCalibrator") as calibrator_cls:
        calibrator_cls.return_value.calibrate.return_value = {
            "shoulder_pan": ProbeResult(1, "shoulder_pan", 900, 3400, 920, 3380, homing_offset=120, drive_mode=0),
            "shoulder_lift": ProbeResult(2, "shoulder_lift", 810, 3200, 830, 3180, homing_offset=-50, drive_mode=0),
            "elbow_flex": ProbeResult(3, "elbow_flex", 880, 3090, 900, 3070, homing_offset=34, drive_mode=0),
            "wrist_flex": ProbeResult(4, "wrist_flex", 860, 3000, 880, 2980, homing_offset=-17, drive_mode=0),
            "gripper": ProbeResult(6, "gripper", 1540, 2780, 1560, 2760, homing_offset=66, drive_mode=0),
            "wrist_roll": ProbeResult(5, "wrist_roll", 0, 4095, 0, 4095, homing_offset=-800, drive_mode=0),
        }
        profile = strategy.recalibrate(arm, store)

    assert profile.motors["shoulder_pan"].homing_offset == 120
    assert profile.motors["shoulder_pan"].range_min == 920
    assert profile.motors["shoulder_pan"].range_max == 3380
    assert profile.motors["gripper"].homing_offset == 66
    assert profile.motors["gripper"].range_min == 1560
    assert profile.motors["gripper"].range_max == 2760
    assert profile.motors["wrist_roll"].homing_offset == -800
    assert profile.motors["wrist_roll"].range_min == 0
    assert profile.motors["wrist_roll"].range_max == 4095


def test_calibrator_happy_path_prepares_all_motors_then_probes_and_applies() -> None:
    call_order: list[str] = []
    results = {
        "shoulder_pan": ProbeResult(1, "shoulder_pan", 900, 3400, 920, 3380),
        "shoulder_lift": ProbeResult(2, "shoulder_lift", 810, 3200, 830, 3180),
        "elbow_flex": ProbeResult(3, "elbow_flex", 880, 3090, 900, 3070),
        "wrist_flex": ProbeResult(4, "wrist_flex", 860, 3000, 880, 2980),
        "gripper": ProbeResult(6, "gripper", 1540, 2780, 1560, 2760),
        "wrist_roll": ProbeResult(5, "wrist_roll", 0, 4095, 0, 4095),
    }

    with patch("roboclaw.embodied.calibration.so101.auto.FeetechMotorsBus"):
        calibrator = _SO101AutoCalibrator("/tmp/tty.so101")
        with patch.object(
            calibrator, "_prepare_motor",
            side_effect=lambda name: call_order.append(f"prepare:{name}"),
        ), patch.object(
            calibrator, "_run_sequence",
            side_effect=lambda: call_order.append("run_sequence"),
        ), patch.object(
            calibrator, "_build_results",
            side_effect=lambda: (call_order.append("build"), results)[1],
        ), patch.object(
            calibrator, "_apply_results",
            side_effect=lambda payload: call_order.append(f"apply:{len(payload)}"),
        ), patch.object(
            calibrator, "_move_to_final_pose",
            side_effect=lambda: call_order.append("final_pose"),
        ):
            returned = calibrator.calibrate()

    assert returned == results
    assert call_order == [
        *(f"prepare:{name}" for name in calibrator.ALL_MOTORS),
        "run_sequence",
        "build",
        "final_pose",
        "apply:6",
    ]


def test_calibrator_restores_orig_eeprom_on_failure() -> None:
    with patch("roboclaw.embodied.calibration.so101.auto.FeetechMotorsBus"):
        calibrator = _SO101AutoCalibrator("/tmp/tty.so101")

        with patch.object(calibrator, "_prepare_motor"), patch.object(
            calibrator, "_run_sequence", side_effect=AutoCalibrationStopped("stopped"),
        ), patch.object(calibrator, "_restore_via_fresh_bus") as restore:
            with pytest.raises(AutoCalibrationStopped):
                calibrator.calibrate()

    restore.assert_called_once_with()


@pytest.mark.asyncio
async def test_auto_calibration_batch_skips_unsupported_and_disconnected(tmp_path: Path) -> None:
    store = CalibrationStore()
    board = Board()
    manifest = DummyManifest()
    batch = AutoCalibrationBatch(board=board, manifest=manifest, store=store, strategy=FastStrategy())

    supported = _make_arm(tmp_path, alias="supported", arm_type="so101_follower", connected=True, with_profile=True)
    unsupported = _make_arm(tmp_path, alias="unsupported", arm_type="koch_follower", connected=True, with_profile=True)
    disconnected = _make_arm(tmp_path, alias="disconnected", arm_type="so101_follower", connected=False, with_profile=True)
    no_profile = _make_arm(tmp_path, alias="no_profile_ok", arm_type="so101_follower", connected=True, with_profile=True)

    total = await batch.start([supported, unsupported, disconnected, no_profile])
    assert total == 4
    assert batch._task is not None
    await asyncio.wait_for(batch._task, timeout=2.0)

    state = board.state
    results = {item["alias"]: item for item in state["calibration_results"]}
    assert state["state"] == "idle"
    assert results["supported"]["status"] == "success"
    assert results["unsupported"]["status"] == "skipped"
    assert results["unsupported"]["reason"] == "unsupported_arm_type"
    assert results["disconnected"]["status"] == "skipped"
    assert results["disconnected"]["reason"] == "disconnected"
    assert results["no_profile_ok"]["status"] == "success"
    assert set(manifest.marked) == {"supported", "no_profile_ok"}


@pytest.mark.asyncio
async def test_auto_calibration_batch_runs_eligible_arms_in_parallel(tmp_path: Path) -> None:
    store = CalibrationStore()
    board = Board()
    manifest = DummyManifest()
    strategy = ParallelStrategy()
    batch = AutoCalibrationBatch(board=board, manifest=manifest, store=store, strategy=strategy)

    first = _make_arm(tmp_path, alias="first", arm_type="so101_follower", connected=True, with_profile=True)
    second = _make_arm(tmp_path, alias="second", arm_type="so101_follower", connected=True, with_profile=True)

    await batch.start([first, second])
    assert batch._task is not None
    await asyncio.wait_for(batch._task, timeout=2.0)

    state = board.state
    results = {item["alias"]: item for item in state["calibration_results"]}
    assert strategy.max_active == 2
    assert state["state"] == "idle"
    assert state["calibration_total"] == 2
    assert results["first"]["status"] == "success"
    assert results["second"]["status"] == "success"
    assert set(manifest.marked) == {"first", "second"}


@pytest.mark.asyncio
async def test_auto_calibration_batch_stop_marks_running_items_stopped(tmp_path: Path) -> None:
    store = CalibrationStore()
    board = Board()
    manifest = DummyManifest()
    batch = AutoCalibrationBatch(board=board, manifest=manifest, store=store, strategy=SlowStrategy())

    first = _make_arm(tmp_path, alias="first", arm_type="so101_follower", connected=True, with_profile=True)
    second = _make_arm(tmp_path, alias="second", arm_type="so101_follower", connected=True, with_profile=True)

    await batch.start([first, second])
    await asyncio.sleep(0.05)
    await batch.stop()

    state = board.state
    results = {item["alias"]: item for item in state["calibration_results"]}
    assert state["state"] == "idle"
    assert results["first"]["status"] == "failed"
    assert results["first"]["reason"] == "stopped"
    assert results["second"]["status"] == "failed"
    assert results["second"]["reason"] == "stopped"
