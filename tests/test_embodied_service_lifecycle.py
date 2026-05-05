from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

if "roboclaw.embodied.service.hub" not in sys.modules:
    hub_mod = types.ModuleType("roboclaw.embodied.service.hub")

    class HubService:
        def __init__(self, parent) -> None:
            self.parent = parent

    hub_mod.HubService = HubService
    sys.modules["roboclaw.embodied.service.hub"] = hub_mod

if "roboclaw.embodied.embodiment.doctor" not in sys.modules:
    doctor_mod = types.ModuleType("roboclaw.embodied.embodiment.doctor")

    class DoctorService:
        def __init__(self, parent) -> None:
            self.parent = parent

        async def check(self, manifest, kwargs, tty_handoff) -> str:
            return json.dumps({
                "environment": {},
                "manifest": manifest.snapshot,
                "hardware_status": self.parent.get_hardware_status(manifest),
            })

    doctor_mod.DoctorService = DoctorService
    sys.modules["roboclaw.embodied.embodiment.doctor"] = doctor_mod

from roboclaw.embodied.embodiment.hardware.monitor import ArmStatus, CameraStatus
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.lock import EmbodimentFileLock
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService

_MANIFEST_DATA = {
    "version": 2,
    "arms": [],
    "hands": [],
    "cameras": [],
    "datasets": {"root": ""},
    "policies": {"root": ""},
}


class ControlledSession:
    def __init__(self, board, result_text: str) -> None:
        self.board = board
        self._result_text = result_text
        self._busy = False
        self._wait_task: asyncio.Task | None = None
        self._exit_callback = None
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.argv: list[str] | None = None

    async def start(self, argv: list[str]) -> None:
        self.argv = argv
        self._busy = True
        self.started.set()

        async def _runner() -> None:
            await self.finish.wait()
            self._busy = False

        self._wait_task = asyncio.create_task(_runner())

    async def stop(self) -> None:
        self.finish.set()
        if self._wait_task is not None:
            await self._wait_task

    async def wait(self) -> None:
        if self._wait_task is not None:
            await self._wait_task

    @property
    def busy(self) -> bool:
        return self._busy

    def result(self) -> str:
        return self._result_text


class FailingSession:
    def __init__(self) -> None:
        self._exit_callback = None

    async def start(self, argv: list[str]) -> None:
        raise RuntimeError("boom")

    @property
    def busy(self) -> bool:
        return False


def _make_service(tmp_path: Path) -> EmbodiedService:
    manifest_data = {
        **_MANIFEST_DATA,
        "datasets": {"root": str(tmp_path / "datasets")},
        "policies": {"root": str(tmp_path / "policies")},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = Manifest(path=manifest_path)
    service = EmbodiedService(manifest=manifest)
    service._file_lock = EmbodimentFileLock(path=tmp_path / ".embodiment.lock")
    return service


def _write_runtime_dataset(root: Path, name: str) -> None:
    dataset_path = root / "local" / name / "meta"
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 2, "fps": 30}),
        encoding="utf-8",
    )


def _write_policy_checkpoint(root: Path, name: str) -> Path:
    checkpoint = root / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"")
    return checkpoint


def _infer_argv(tmp_path: Path, checkpoint: Path, *, num_episodes: int = 1) -> list[str]:
    return [
        sys.executable,
        "-m",
        "roboclaw.embodied.command.wrapper",
        "record",
        "--robot.type=so101_follower",
        '--robot.cameras={"wrist": {"type": "opencv"}}',
        f"--policy.path={checkpoint}",
        "--dataset.repo_id=local/eval",
        f"--dataset.root={tmp_path / 'datasets' / 'local' / 'eval'}",
        f"--dataset.num_episodes={num_episodes}",
        "--dataset.episode_time_s=60",
    ]


def _single_follower_status() -> list[ArmStatus]:
    return [ArmStatus("follower", "so101_follower", "follower", True, True)]


def _teleop_arm_statuses() -> list[ArmStatus]:
    return [
        ArmStatus("follower", "so101_follower", "follower", True, True),
        ArmStatus("leader", "so101_leader", "leader", True, True),
    ]


def _bind_replay_setup(service: EmbodiedService) -> None:
    service.bind_arm("follower", "so101_follower", SerialInterface(dev="/tmp/follower"))


def _bind_teleop_setup(service: EmbodiedService) -> None:
    _bind_replay_setup(service)
    service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))


def _bind_infer_setup(service: EmbodiedService) -> None:
    _bind_replay_setup(service)
    service.bind_camera("wrist", VideoInterface(dev="/tmp/wrist"))


@pytest.mark.asyncio
async def test_run_replay_waits_for_process_completion_without_tty(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_replay_setup(service)
    _write_runtime_dataset(tmp_path / "datasets", "demo")
    service.replay = ControlledSession(service.board, "Replay finished.")
    run_replay = getattr(service, "run_replay")

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_single_follower_status()), patch(
        "roboclaw.embodied.service.CommandBuilder.replay",
        return_value=["replay-cmd"],
    ):
        task = asyncio.create_task(run_replay(dataset_name="demo", episode=2, fps=15))
        await asyncio.wait_for(service.replay.started.wait(), timeout=1)

        assert service.busy
        assert service.embodiment_busy
        assert not task.done()

        service.replay.finish.set()
        result = await asyncio.wait_for(task, timeout=1)

    assert result == "Replay finished."
    assert service.replay.argv == ["replay-cmd"]
    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None


@pytest.mark.asyncio
async def test_run_inference_waits_for_process_completion_without_tty(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_infer_setup(service)
    service.infer = ControlledSession(service.board, "Inference finished.")
    run_inference = getattr(service, "run_inference")
    checkpoint = _write_policy_checkpoint(tmp_path / "policies", "act")
    infer_argv = _infer_argv(tmp_path, checkpoint, num_episodes=3)

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_single_follower_status()), patch(
        "roboclaw.embodied.service.check_camera_status",
        return_value=CameraStatus("wrist", True, 640, 480),
    ), patch("roboclaw.embodied.service.CommandBuilder.infer", return_value=infer_argv):
        task = asyncio.create_task(run_inference(checkpoint_path="/models/act", num_episodes=3))
        await asyncio.wait_for(service.infer.started.wait(), timeout=1)

        assert service.busy
        assert service.embodiment_busy
        assert not task.done()

        service.infer.finish.set()
        result = await asyncio.wait_for(task, timeout=1)

    assert result == "Inference finished."
    assert service.infer.argv == infer_argv
    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None


@pytest.mark.asyncio
async def test_start_teleop_releases_lock_on_session_start_failure(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_teleop_setup(service)
    service.teleop = FailingSession()

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_teleop_arm_statuses()), patch(
        "roboclaw.embodied.service.CommandBuilder.teleop",
        return_value=["teleop-cmd"],
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await service.start_teleop(fps=20)

    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None


@pytest.mark.asyncio
async def test_start_recording_releases_lock_on_session_start_failure(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_infer_setup(service)
    service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))
    service.record = FailingSession()

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_teleop_arm_statuses()), patch(
        "roboclaw.embodied.service.check_camera_status",
        return_value=CameraStatus("wrist", True, 640, 480),
    ), patch("roboclaw.embodied.service.CommandBuilder.record", return_value=["record-cmd"]), patch.object(
        service, "_verify_record_preflight"
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await service.start_recording(task="pick", dataset_name="demo")

    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None


@pytest.mark.asyncio
async def test_start_replay_releases_lock_on_session_start_failure(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_replay_setup(service)
    _write_runtime_dataset(tmp_path / "datasets", "demo")
    service.replay = FailingSession()

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_single_follower_status()), patch(
        "roboclaw.embodied.service.CommandBuilder.replay",
        return_value=["replay-cmd"],
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await service.start_replay(dataset_name="demo")

    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None


@pytest.mark.asyncio
async def test_start_inference_releases_lock_on_session_start_failure(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _bind_infer_setup(service)
    service.infer = FailingSession()
    checkpoint = _write_policy_checkpoint(tmp_path / "policies", "act")
    infer_argv = _infer_argv(tmp_path, checkpoint)

    with patch("roboclaw.embodied.service.check_arm_status", side_effect=_single_follower_status()), patch(
        "roboclaw.embodied.service.check_camera_status",
        return_value=CameraStatus("wrist", True, 640, 480),
    ), patch("roboclaw.embodied.service.CommandBuilder.infer", return_value=infer_argv):
        with pytest.raises(RuntimeError, match="boom"):
            await service.start_inference(checkpoint_path="/models/act")

    assert not service.busy
    assert not service.embodiment_busy
    assert service._active_session is None
