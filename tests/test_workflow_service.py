from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.workflow import WorkflowPlanner


def _make_service(tmp_path: Path) -> EmbodiedService:
    manifest_data = {
        "version": 2,
        "arms": [],
        "hands": [],
        "cameras": [],
        "datasets": {"root": str(tmp_path / "datasets")},
        "policies": {"root": str(tmp_path / "policies")},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    manifest = Manifest(path=manifest_path)
    service = EmbodiedService(manifest=manifest)
    service.bind_arm("follower", "so101_follower", SerialInterface(dev="/tmp/follower"))
    service.bind_arm("leader", "so101_leader", SerialInterface(dev="/tmp/leader"))
    service.bind_camera("wrist", VideoInterface(dev="/tmp/wrist"))
    return service


def _materialize_runtime_dataset(tmp_path: Path, name: str) -> None:
    dataset_dir = tmp_path / "datasets" / "local" / name / "meta"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "info.json").write_text("{}", encoding="utf-8")


def _materialize_checkpoint(tmp_path: Path, name: str = "checkpoint") -> Path:
    checkpoint_dir = tmp_path / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "model.safetensors").write_text("", encoding="utf-8")
    return checkpoint_dir


def test_start_workflow_phase_train_uses_materialized_record_dataset_name(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _materialize_runtime_dataset(tmp_path, "pick_cube_v1")
    service.train.train = AsyncMock(return_value="Training started. Job ID: job-1")

    result = asyncio.run(service.start_workflow_phase({
        "record": {
            "enabled": True,
            "task": "pick cube",
            "dataset_name": "pick_cube_v1",
        },
        "train": {
            "enabled": True,
            "policy_type": "act",
            "steps": 1000,
        },
    }, "train"))

    assert result == {"message": "Training started. Job ID: job-1", "job_id": "job-1"}
    service.train.train.assert_awaited_once()
    kwargs = service.train.train.await_args.kwargs["kwargs"]
    assert kwargs["dataset_name"] == "pick_cube_v1"


def test_start_workflow_phase_train_rejects_blocked_stage(tmp_path: Path) -> None:
    service = _make_service(tmp_path)

    try:
        asyncio.run(service.start_workflow_phase({
            "record": {
                "enabled": True,
                "task": "pick cube",
                "dataset_name": "pick_cube_v1",
            },
            "train": {
                "enabled": True,
                "policy_type": "act",
                "steps": 1000,
            },
        }, "train"))
    except RuntimeError as exc:
        assert str(exc) == "Workflow phase 'train' is waiting on: record."
    else:
        raise AssertionError("expected blocked train stage to raise")


def test_start_workflow_phase_record_uses_planned_dataset_name(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service.start_recording = AsyncMock(side_effect=lambda **kwargs: kwargs["dataset_name"])
    spec = {
        "record": {
            "enabled": True,
            "task": "pick cube",
        },
    }
    expected = WorkflowPlanner(service.manifest, service.datasets).plan(spec).stage("record").dataset_name

    result = asyncio.run(service.start_workflow_phase(spec, "record"))

    assert result["status"] == "recording"
    assert result["dataset_name"] == expected
    service.start_recording.assert_awaited_once()
    kwargs = service.start_recording.await_args.kwargs
    assert kwargs["dataset_name"] == expected
    assert kwargs["dataset_name"] == result["dataset_name"]


def test_start_workflow_phase_infer_rejects_blocked_train_checkpoint(tmp_path: Path) -> None:
    service = _make_service(tmp_path)

    try:
        asyncio.run(service.start_workflow_phase({
            "record": {
                "enabled": True,
                "task": "pick cube",
                "dataset_name": "pick_cube_v1",
            },
            "train": {
                "enabled": True,
                "policy_type": "act",
                "steps": 1000,
            },
            "infer": {
                "enabled": True,
                "dataset_name": "eval_pick_cube_v1",
            },
        }, "infer"))
    except RuntimeError as exc:
        assert str(exc) == "Workflow phase 'infer' is waiting on: train."
    else:
        raise AssertionError("expected blocked infer stage to raise")


def test_start_workflow_phase_infer_uses_explicit_checkpoint_without_source_dataset(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service.start_inference = AsyncMock(return_value=None)
    checkpoint_dir = _materialize_checkpoint(tmp_path, "model")

    result = asyncio.run(service.start_workflow_phase({
        "name": "pick-cube-pipeline",
        "record": {
            "enabled": True,
            "task": "pick cube",
            "dataset_name": "pick_cube_v1",
        },
        "infer": {
            "enabled": True,
            "checkpoint_path": str(checkpoint_dir),
        },
    }, "infer"))

    assert result["status"] == "inferring"
    assert result["dataset_name"] == "eval_pick-cube-pipeline"
    assert result["checkpoint_path"] == str(checkpoint_dir)
    service.start_inference.assert_awaited_once()
    kwargs = service.start_inference.await_args.kwargs
    assert kwargs["checkpoint_path"] == str(checkpoint_dir)
    assert kwargs["source_dataset"] == ""
    assert kwargs["dataset_name"] == "eval_pick-cube-pipeline"


def test_start_workflow_phase_record_builds_record_command_from_planned_defaults(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service._require_capability = lambda *_args, **_kwargs: None
    starts: list[tuple[str, list[str]]] = []

    async def fake_start_session(_session, *, owner: str, argv: list[str]) -> None:
        starts.append((owner, argv))

    service._start_managed_session = fake_start_session
    spec = {
        "name": "pick-cube-pipeline",
        "record": {
            "enabled": True,
            "task": "pick cube",
        },
    }
    expected = WorkflowPlanner(service.manifest, service.datasets).plan(spec).stage("record").dataset_name

    result = asyncio.run(service.start_workflow_phase(spec, "record"))

    assert result == {"status": "recording", "dataset_name": expected}
    assert starts[0][0] == "recording"
    argv = starts[0][1]
    assert argv[:4] == [
        "/Users/pearl/anaconda3/bin/python",
        "-m",
        "roboclaw.embodied.command.wrapper",
        "record",
    ]
    assert f"--dataset.repo_id=local/{expected}" in argv
    assert f"--dataset.root={tmp_path / 'datasets' / 'local' / expected}" in argv
    assert "--dataset.single_task=pick cube" in argv


def test_start_workflow_phase_train_builds_train_command_for_runtime_dataset(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    _materialize_runtime_dataset(tmp_path, "pick_cube_v1")
    captured: list[tuple[list[str], str]] = []

    async def fake_run_detached(self, argv: list[str], log_dir: Path) -> str:
        captured.append((argv, str(log_dir)))
        return "job-42"

    with patch("roboclaw.embodied.executor.SubprocessExecutor.run_detached", new=fake_run_detached):
        result = asyncio.run(service.start_workflow_phase({
            "train": {
                "enabled": True,
                "dataset_name": "pick_cube_v1",
                "policy_type": "act",
                "steps": 1234,
                "device": "cpu",
            },
        }, "train"))

    assert result == {"message": "Training started. Job ID: job-42", "job_id": "job-42"}
    argv = captured[0][0]
    assert argv[0] == "lerobot-train"
    assert "--dataset.repo_id=local/pick_cube_v1" in argv
    assert f"--dataset.root={tmp_path / 'datasets' / 'local' / 'pick_cube_v1'}" in argv
    assert f"--output_dir={tmp_path / 'policies' / 'pick_cube_v1'}" in argv
    assert "--steps=1234" in argv
    assert "--policy.device=cpu" in argv


def test_start_workflow_phase_infer_builds_command_from_explicit_checkpoint(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service._require_capability = lambda *_args, **_kwargs: None
    starts: list[tuple[str, list[str]]] = []

    async def fake_start_session(_session, *, owner: str, argv: list[str]) -> None:
        starts.append((owner, argv))

    service._start_managed_session = fake_start_session
    checkpoint_dir = _materialize_checkpoint(tmp_path, "model")

    result = asyncio.run(service.start_workflow_phase({
        "name": "pick-cube-pipeline",
        "infer": {
            "enabled": True,
            "checkpoint_path": str(checkpoint_dir),
            "task": "eval pick",
            "num_episodes": 2,
            "episode_time_s": 15,
        },
    }, "infer"))

    assert result == {
        "status": "inferring",
        "dataset_name": "eval_pick-cube-pipeline",
        "checkpoint_path": str(checkpoint_dir),
    }
    assert starts[0][0] == "inferring"
    argv = starts[0][1]
    assert argv[:4] == [
        "/Users/pearl/anaconda3/bin/python",
        "-m",
        "roboclaw.embodied.command.wrapper",
        "record",
    ]
    assert f"--policy.path={checkpoint_dir}" in argv
    assert "--dataset.single_task=eval pick" in argv
    assert "--dataset.num_episodes=2" in argv
    assert "--dataset.episode_time_s=15" in argv
    assert "--dataset.repo_id=local/eval_pick-cube-pipeline" in argv
