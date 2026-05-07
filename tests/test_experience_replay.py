from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.agent.experience import ExperienceRecord, ExperienceStore
from roboclaw.data.datasets import DatasetRuntimeRef
from roboclaw.embodied.board import Board
from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService


@pytest.fixture(autouse=True)
def isolated_roboclaw_home(tmp_path):
    with patch(
        "roboclaw.embodied.embodiment.lock.get_roboclaw_home",
        return_value=tmp_path,
    ), patch(
        "roboclaw.embodied.embodiment.manifest.helpers.get_roboclaw_home",
        return_value=tmp_path,
    ):
        yield


@pytest.fixture()
def embodied_service(tmp_path: Path) -> EmbodiedService:
    board = Board()
    manifest = Manifest(path=tmp_path / "workspace" / "embodied" / "manifest.json", board=board)
    monitor = HardwareMonitor(board=board, manifest=manifest)
    return EmbodiedService(hardware_monitor=monitor, board=board, manifest=manifest)


def test_get_replay_datasets_returns_empty_without_history(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)

    assert store.get_replay_datasets(current_dataset="demo", policy="act") == []


def test_get_replay_datasets_returns_recent_success_dataset(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    _append_record(
        store,
        timestamp="2026-05-07T10:00:00+00:00",
        dataset="old_pick",
        policy="act",
        outcome="success",
    )
    _append_record(
        store,
        timestamp="2026-05-07T11:00:00+00:00",
        dataset="new_place",
        policy="act",
        outcome="success",
    )

    assert store.get_replay_datasets(current_dataset="demo", policy="act") == ["new_place", "old_pick"]


def test_get_replay_datasets_excludes_current_dataset(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    _append_record(
        store,
        timestamp="2026-05-07T11:00:00+00:00",
        dataset="demo",
        policy="act",
        outcome="success",
    )
    _append_record(
        store,
        timestamp="2026-05-07T10:00:00+00:00",
        dataset="history_a",
        policy="act",
        outcome="success",
    )

    assert store.get_replay_datasets(current_dataset="demo", policy="act") == ["history_a"]


def test_get_replay_datasets_respects_max_datasets(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    _append_record(store, timestamp="2026-05-07T12:00:00+00:00", dataset="history_c", policy="act", outcome="success")
    _append_record(store, timestamp="2026-05-07T11:00:00+00:00", dataset="history_b", policy="act", outcome="success")
    _append_record(store, timestamp="2026-05-07T10:00:00+00:00", dataset="history_a", policy="act", outcome="success")

    assert store.get_replay_datasets(current_dataset="demo", policy="act", max_datasets=2) == [
        "history_c",
        "history_b",
    ]


def test_start_job_state_keeps_argv_unchanged_when_continual_learning_disabled(
    embodied_service: EmbodiedService,
) -> None:
    dataset_map = _dataset_map("demo", "history_a")
    embodied_service.datasets.resolve_runtime_dataset = lambda name: dataset_map[name]  # type: ignore[method-assign]
    embodied_service.datasets.get_local_dataset = lambda dataset_id: dataset_map.get(dataset_id.removeprefix("local/"))  # type: ignore[method-assign]
    _append_record(
        ExperienceStore(embodied_service.manifest._path.parent.parent),
        timestamp="2026-05-07T12:00:00+00:00",
        dataset="history_a",
        policy="act",
        outcome="success",
    )
    captured: dict[str, object] = {}

    def fake_train(manifest, *, dataset, policy_type, steps, device):
        argv = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--dataset.root={dataset.local_path}",
        ]
        captured["argv"] = argv
        captured["dataset"] = dataset
        return argv

    with patch(
        "roboclaw.embodied.service.session.train.CommandBuilder.train",
        side_effect=fake_train,
    ), patch(
        "roboclaw.embodied.executor.SubprocessExecutor.run_detached",
        new=AsyncMock(return_value="job-1"),
    ):
        state = asyncio.run(embodied_service.train.start_job_state(
            embodied_service.manifest,
            {
                "dataset_name": "demo",
                "policy_type": "act",
                "steps": 1000,
                "device": "cuda",
                "continual_learning": False,
            },
        ))

    assert captured["dataset"].local_path == Path("/tmp/demo")
    assert "history_a" not in " ".join(captured["argv"])
    assert state["replay_datasets"] == ""


def test_start_job_state_includes_replay_dataset_when_continual_learning_enabled(
    embodied_service: EmbodiedService,
) -> None:
    dataset_map = _dataset_map("demo", "history_a")
    embodied_service.datasets.resolve_runtime_dataset = lambda name: dataset_map[name]  # type: ignore[method-assign]
    embodied_service.datasets.get_local_dataset = lambda dataset_id: dataset_map.get(dataset_id.removeprefix("local/"))  # type: ignore[method-assign]
    _append_record(
        ExperienceStore(embodied_service.manifest._path.parent.parent),
        timestamp="2026-05-07T12:00:00+00:00",
        dataset="history_a",
        policy="act",
        outcome="success",
    )
    captured: dict[str, object] = {}
    replay_runtime = DatasetRuntimeRef(
        name="demo",
        repo_id="local/demo",
        local_path=Path("/tmp/replay_demo__history_a"),
    )

    def fake_train(manifest, *, dataset, policy_type, steps, device):
        argv = [
            "python3",
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--dataset.root={dataset.local_path}",
        ]
        captured["argv"] = argv
        captured["dataset"] = dataset
        return argv

    with patch(
        "roboclaw.embodied.service.session.train.CommandBuilder.train",
        side_effect=fake_train,
    ), patch(
        "roboclaw.embodied.service.session.train.TrainSession._prepare_replay_runtime_dataset",
        return_value=replay_runtime,
    ), patch(
        "roboclaw.embodied.executor.SubprocessExecutor.run_detached",
        new=AsyncMock(return_value="job-1"),
    ):
        state = asyncio.run(embodied_service.train.start_job_state(
            embodied_service.manifest,
            {
                "dataset_name": "demo",
                "policy_type": "act",
                "steps": 1000,
                "device": "cuda",
                "continual_learning": True,
            },
        ))

    assert captured["dataset"].local_path == replay_runtime.local_path
    assert "history_a" in " ".join(captured["argv"])
    assert state["replay_datasets"] == "history_a"
    assert "history_a" in str(state["experience_hint"])


def _append_record(
    store: ExperienceStore,
    *,
    timestamp: str,
    dataset: str,
    policy: str,
    outcome: str,
) -> None:
    store.append(ExperienceRecord(
        timestamp=timestamp,
        task_type="train",
        summary=f"{dataset} -> {outcome}",
        outcome=outcome,
        dataset=dataset,
        policy=policy,
        provider="local",
    ))


def _dataset_map(*dataset_names: str) -> dict[str, SimpleNamespace]:
    mapping: dict[str, SimpleNamespace] = {}
    for dataset_name in dataset_names:
        mapping[dataset_name] = SimpleNamespace(
            id=f"local/{dataset_name}",
            runtime=DatasetRuntimeRef(
                name=dataset_name,
                repo_id=f"local/{dataset_name}",
                local_path=Path(f"/tmp/{dataset_name}"),
            ),
        )
    return mapping
