from __future__ import annotations

import json
from pathlib import Path

from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.embodied.workflow import WorkflowPlanner, WorkflowSpec


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


def _materialize_checkpoint(tmp_path: Path, name: str = "checkpoint") -> Path:
    checkpoint_dir = tmp_path / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "model.safetensors").write_text("", encoding="utf-8")
    return checkpoint_dir


def test_workflow_planner_compiles_record_train_infer_chain(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    plan = WorkflowPlanner(service.manifest, service.datasets).plan(WorkflowSpec.model_validate({
        "name": "pick-cube-pipeline",
        "hardware": {"use_cameras": True},
        "record": {
            "enabled": True,
            "task": "pick cube",
            "dataset_name": "pick_cube_v1",
            "num_episodes": 5,
        },
        "train": {
            "enabled": True,
            "policy_type": "act",
            "steps": 2000,
        },
        "infer": {
            "enabled": True,
            "dataset_name": "eval_pick_cube_v1",
            "num_episodes": 2,
        },
    }))

    assert plan.ok is True

    record_stage = plan.stage("record")
    assert record_stage.ready is True
    assert record_stage.dataset_name == "pick_cube_v1"
    assert "--dataset.repo_id=local/pick_cube_v1" in record_stage.command

    train_stage = plan.stage("train")
    assert train_stage.ready is False
    assert train_stage.blocked_by == ["record"]
    assert train_stage.dataset_name == "pick_cube_v1"
    assert "--policy.type=act" in train_stage.command
    assert train_stage.checkpoint_path.endswith(
        "pick_cube_v1/checkpoints/last/pretrained_model"
    )

    infer_stage = plan.stage("infer")
    assert infer_stage.ready is False
    assert infer_stage.blocked_by == ["train"]
    assert infer_stage.dataset_name == "eval_pick_cube_v1"
    assert infer_stage.source_dataset == "pick_cube_v1"
    assert infer_stage.checkpoint_path.endswith(
        "pick_cube_v1/checkpoints/last/pretrained_model"
    )
    assert any(arg.startswith("--policy.path=") for arg in infer_stage.command)


def test_workflow_planner_generates_stable_default_dataset_names(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    planner = WorkflowPlanner(service.manifest, service.datasets)
    checkpoint_dir = _materialize_checkpoint(tmp_path)
    spec = {
        "name": "Pick Cube Pipeline",
        "record": {
            "enabled": True,
            "task": "pick cube",
        },
        "infer": {
            "enabled": True,
            "checkpoint_path": str(checkpoint_dir),
        },
    }

    first = planner.plan(spec)
    second = planner.plan(spec)

    assert first.stage("record").dataset_name == "rec_pick_cube_pipeline"
    assert first.stage("infer").dataset_name == "eval_pick_cube_pipeline"
    assert first.stage("record").dataset_name == second.stage("record").dataset_name
    assert first.stage("infer").dataset_name == second.stage("infer").dataset_name


def test_workflow_planner_default_record_name_ignores_unrelated_downstream_changes(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    planner = WorkflowPlanner(service.manifest, service.datasets)

    base = planner.plan({
        "record": {
            "enabled": True,
            "task": "pick cube",
        },
    })
    with_train = planner.plan({
        "record": {
            "enabled": True,
            "task": "pick cube",
        },
        "train": {
            "enabled": True,
            "steps": 2000,
        },
    })

    assert base.stage("record").dataset_name == with_train.stage("record").dataset_name


def test_workflow_planner_reports_missing_explicit_checkpoint(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    plan = WorkflowPlanner(service.manifest, service.datasets).plan({
        "infer": {
            "enabled": True,
            "checkpoint_path": str(tmp_path / "missing_checkpoint"),
        },
    })

    assert plan.ok is False
    infer_stage = plan.stage("infer")
    assert infer_stage.ready is False
    assert any(issue.code == "invalid_checkpoint" for issue in infer_stage.issues)


def test_workflow_planner_reports_missing_train_dataset_without_record_stage(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    plan = WorkflowPlanner(service.manifest, service.datasets).plan({
        "train": {"enabled": True},
    })

    assert plan.ok is False
    train_stage = plan.stage("train")
    assert train_stage.ready is False
    assert any(issue.code == "missing_dataset" for issue in train_stage.issues)
