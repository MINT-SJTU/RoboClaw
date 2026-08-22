from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.agent.experience import ExperienceStore
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


def test_train_session_records_experience_and_reuses_it_as_hint(embodied_service: EmbodiedService) -> None:
    dataset = SimpleNamespace(
        name="demo",
        runtime=SimpleNamespace(name="demo", repo_id="local/demo", local_path=Path("/tmp/demo")),
    )
    embodied_service.datasets.resolve_runtime_dataset = lambda name: dataset  # type: ignore[method-assign]

    with patch(
        "roboclaw.embodied.service.session.train.CommandBuilder.train",
        return_value=["python3", "-m", "lerobot.scripts.lerobot_train"],
    ), patch(
        "roboclaw.embodied.executor.SubprocessExecutor.run_detached",
        new=AsyncMock(side_effect=["job-1", "job-2"]),
    ), patch(
        "roboclaw.embodied.executor.SubprocessExecutor.job_status",
        new=AsyncMock(return_value={
            "job_id": "job-1",
            "status": "finished",
            "running": False,
            "pid": 123,
            "log_path": "/tmp/job-1.log",
            "log_tail": "training complete",
        }),
    ):
        first = asyncio.run(embodied_service.train.start_job_state(
            embodied_service.manifest,
            {"dataset_name": "demo", "policy_type": "act", "steps": 1000, "device": "cuda"},
        ))
        assert first["experience_hint"] == ""

        finished = asyncio.run(embodied_service.train.job_status_state("job-1"))
        assert finished["status"] == "finished"

        second = asyncio.run(embodied_service.train.start_job_state(
            embodied_service.manifest,
            {"dataset_name": "demo", "policy_type": "act", "steps": 1000, "device": "cuda"},
        ))

    assert "success" in str(second["experience_hint"])
    assert "demo" in str(second["message"])

    store = ExperienceStore(embodied_service.manifest._path.parent.parent)
    records = store.read_all()
    outcomes = {record.outcome for record in records}
    assert "submitted" in outcomes
    assert "success" in outcomes
