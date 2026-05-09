from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

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
