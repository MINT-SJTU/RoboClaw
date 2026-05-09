from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService


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


def test_start_workflow_phase_train_inherits_record_dataset_name(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
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


def test_start_workflow_phase_infer_inherits_train_source_dataset(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    service.start_inference = AsyncMock(return_value=None)

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
        "infer": {
            "enabled": True,
            "dataset_name": "eval_pick_cube_v1",
        },
    }, "infer"))

    assert result["status"] == "inferring"
    assert result["dataset_name"] == "eval_pick_cube_v1"
    assert result["checkpoint_path"].endswith("pick_cube_v1/checkpoints/last/pretrained_model")
    service.start_inference.assert_awaited_once()
    kwargs = service.start_inference.await_args.kwargs
    assert kwargs["source_dataset"] == "pick_cube_v1"
