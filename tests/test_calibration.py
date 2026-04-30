"""Focused tests for calibration session success semantics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.board import Board
from roboclaw.embodied.board.constants import SessionState
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.helpers import save_manifest
from roboclaw.embodied.service.session.calibrate import CalibrationSession


class DummyProcess:
    """Minimal process stub for session wait-path tests."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdin = None

    async def wait(self) -> int:
        return self.returncode


class DummyParent:
    """Small parent stub with just the session dependencies we need."""

    def __init__(self, manifest: Manifest) -> None:
        self.board = Board()
        self.manifest = manifest
        self.embodiment_busy = True
        self.release_calls = 0

    def acquire_embodiment(self, owner: str) -> None:
        self.embodiment_busy = True

    def release_embodiment(self, owner: str = "") -> None:
        self.embodiment_busy = False
        self.release_calls += 1


def _build_manifest(tmp_path: Path) -> tuple[Manifest, Path]:
    cal_dir = tmp_path / "calibration" / "SIM001"
    cal_dir.mkdir(parents=True)
    path = tmp_path / "manifest.json"
    save_manifest(
        {
            "version": 2,
            "arms": [
                {
                    "alias": "test_arm",
                    "type": "so101_follower",
                    "port": "/dev/serial/by-id/usb-SIM_Serial_SIM001-if00",
                    "calibration_dir": str(cal_dir),
                    "calibrated": False,
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


def _saved_calibration_path(cal_dir: Path) -> Path:
    return cal_dir / f"{cal_dir.name}.json"


@pytest.mark.asyncio
async def test_wait_process_marks_calibrated_after_saved_file(tmp_path: Path) -> None:
    manifest, cal_dir = _build_manifest(tmp_path)
    _saved_calibration_path(cal_dir).write_text("{}", encoding="utf-8")

    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    session._arm = manifest.find_arm("test_arm")
    session._cal_manifest = manifest
    session._process = DummyProcess(0)

    await session.board.update(
        state=SessionState.CALIBRATING,
        calibration_arm="test_arm",
        calibration_step="done",
    )
    await session._wait_process()

    assert manifest.find_arm("test_arm").calibrated is True
    assert session.board.state["state"] == SessionState.IDLE
    assert parent.release_calls == 1


@pytest.mark.asyncio
async def test_wait_process_missing_saved_file_sets_error(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)

    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    session._arm = manifest.find_arm("test_arm")
    session._cal_manifest = manifest
    session._process = DummyProcess(0)

    await session.board.update(
        state=SessionState.CALIBRATING,
        calibration_arm="test_arm",
        calibration_step="done",
    )
    await session._wait_process()

    state = session.board.state
    assert state["state"] == SessionState.ERROR
    assert "did not save SIM001.json" in state["error"]
    assert manifest.find_arm("test_arm").calibrated is False
    assert parent.release_calls == 1


@pytest.mark.asyncio
async def test_wait_process_manifest_failure_sets_error(tmp_path: Path) -> None:
    manifest, cal_dir = _build_manifest(tmp_path)
    _saved_calibration_path(cal_dir).write_text("{}", encoding="utf-8")

    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    session._arm = manifest.find_arm("test_arm")
    session._cal_manifest = manifest
    session._process = DummyProcess(0)

    with patch.object(manifest, "mark_arm_calibrated", side_effect=RuntimeError("persist failed")):
        await session.board.update(
            state=SessionState.CALIBRATING,
            calibration_arm="test_arm",
            calibration_step="done",
        )
        await session._wait_process()

    state = session.board.state
    assert state["state"] == SessionState.ERROR
    assert state["error"] == "persist failed"
    assert manifest.find_arm("test_arm").calibrated is False
    assert parent.release_calls == 1


@pytest.mark.asyncio
async def test_calibrate_one_tty_requires_saved_file(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    arm = manifest.find_arm("test_arm")
    runner = AsyncMock()
    runner.run_interactive.return_value = (0, "")
    tty_handoff = AsyncMock()

    result = await session._calibrate_one_tty(arm, manifest, runner, tty_handoff)

    assert result == "test_arm: FAILED (Calibration for test_arm did not save SIM001.json.)"
    assert manifest.find_arm("test_arm").calibrated is False


@pytest.mark.asyncio
async def test_calibrate_one_tty_marks_calibrated_after_saved_file(tmp_path: Path) -> None:
    manifest, cal_dir = _build_manifest(tmp_path)
    _saved_calibration_path(cal_dir).write_text("{}", encoding="utf-8")

    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    arm = manifest.find_arm("test_arm")
    runner = AsyncMock()
    runner.run_interactive.return_value = (0, "")
    tty_handoff = AsyncMock()

    result = await session._calibrate_one_tty(arm, manifest, runner, tty_handoff)

    assert result == "test_arm: OK"
    assert manifest.find_arm("test_arm").calibrated is True


def test_result_prefers_error_over_done(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    parent = DummyParent(manifest)
    session = CalibrationSession(parent)
    session.board.set_field("state", SessionState.ERROR)
    session.board.set_field("calibration_arm", "test_arm")
    session.board.set_field("calibration_step", "done")
    session.board.set_field("error", "persist failed")

    assert session.result() == "Calibration of test_arm failed: persist failed"
