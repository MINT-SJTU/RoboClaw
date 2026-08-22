"""Focused tests for provider-aware TrainSession behavior."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.data.datasets import DatasetCatalog
from roboclaw.embodied.command import ActionError
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service.session.train import TrainSession
from roboclaw.embodied.training import (
    TrainingJobState,
    TrainingJobStatus,
    TrainingJobStore,
    TrainingProvider,
    TrainingSubmitResult,
)


def _make_session(tmp_path: Path) -> TrainSession:
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
    parent = SimpleNamespace(
        datasets=DatasetCatalog(root_resolver=lambda: tmp_path / "datasets"),
        manifest=manifest,
    )
    session = TrainSession(parent)
    session._jobs = TrainingJobStore(root=tmp_path / "logs")
    return session


def _write_runtime_dataset(root: Path, name: str) -> None:
    dataset_path = root / "local" / name / "meta"
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 2, "fps": 30}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_train_session_dispatches_remote_provider_and_collects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(tmp_path)
    _write_runtime_dataset(tmp_path / "datasets", "demo")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", "registry.example/roboclaw/train:stable")
    backend = AsyncMock()
    backend.submit.return_value = TrainingSubmitResult(
        job_id="cloud-job",
        provider=TrainingProvider.ALIYUN,
        message="Aliyun training submitted. Job ID: cloud-job",
        remote_job_id="provider-job-1",
    )
    backend.wait.return_value = TrainingJobStatus(
        job_id="cloud-job",
        provider=TrainingProvider.ALIYUN,
        state=TrainingJobState.SUCCEEDED,
        running=False,
        message="succeeded",
        remote_job_id="provider-job-1",
        output_dir=str(tmp_path / "policies" / "demo"),
    )
    backend.collect.return_value = [str(tmp_path / "policies" / "demo")]

    with patch("roboclaw.embodied.service.session.train.build_training_backend", return_value=backend):
        message = await session.train(
            session._parent.manifest,
            {
                "dataset_name": "demo",
                "provider": "aliyun",
                "wait": True,
            },
            tty_handoff=None,
        )

    assert "Aliyun training submitted" in message
    assert "Final status: succeeded" in message
    assert "Collected artifacts" in message
    backend.submit.assert_awaited_once()
    backend.wait.assert_awaited_once()
    backend.collect.assert_awaited_once()
    assert session._jobs.load("cloud-job") is not None


@pytest.mark.asyncio
async def test_train_session_uses_aliyun_preset_defaults_and_server_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(tmp_path)
    _write_runtime_dataset(tmp_path / "datasets", "demo")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", "registry.example/roboclaw/train:stable")
    backend = AsyncMock()
    backend.submit.return_value = TrainingSubmitResult(
        job_id="cloud-job",
        provider=TrainingProvider.ALIYUN,
        message="Aliyun training submitted. Job ID: cloud-job",
        remote_job_id="provider-job-1",
    )

    with patch("roboclaw.embodied.service.session.train.build_training_backend", return_value=backend):
        await session.train(
            session._parent.manifest,
            {
                "dataset_name": "demo",
                "provider": "aliyun",
                "preset": "aliyun-a10-recommended",
            },
            tty_handoff=None,
        )

    request = backend.submit.await_args.args[0]
    assert request.resources.image == "registry.example/roboclaw/train:stable"
    assert request.resources.gpu_type == "A10"
    assert request.resources.gpu_count == 1
    assert request.resources.cpu_cores == 16
    assert request.resources.memory_gb == 128
    assert request.resources.node_count == 1


@pytest.mark.asyncio
async def test_train_session_requires_aliyun_image_when_no_default_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(tmp_path)
    _write_runtime_dataset(tmp_path / "datasets", "demo")
    monkeypatch.delenv("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", raising=False)

    with patch("roboclaw.embodied.service.session.train.build_training_backend", return_value=AsyncMock()):
        with pytest.raises(ActionError, match="Aliyun training image is not configured"):
            await session.train(
                session._parent.manifest,
                {
                    "dataset_name": "demo",
                    "provider": "aliyun",
                    "preset": "aliyun-a10-recommended",
                },
                tty_handoff=None,
            )


def test_train_session_capabilities_report_configured_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(tmp_path)
    monkeypatch.setenv("ROBOCLAW_ALIYUN_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_ACCESS_KEY_SECRET", "sk")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_REGION_ID", "cn-shanghai")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_WORKSPACE_ID", "1307901")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_OSS_BUCKET", "bucket")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
    monkeypatch.setenv("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", "registry.example/roboclaw/train:stable")
    monkeypatch.setenv("ROBOCLAW_AUTODL_HOST", "autodl.example")
    monkeypatch.setenv("ROBOCLAW_REMOTE_TRAINING_MODE", "managed")
    monkeypatch.setenv("ROBOCLAW_REMOTE_TRAINING_NOTICE", "Billing or quota is enforced by this deployment.")

    capabilities = session.capabilities()

    assert capabilities["locations"]["current_machine"]["configured"] is True
    assert capabilities["locations"]["remote_backend"]["configured"] is True
    assert capabilities["locations"]["remote_backend"]["mode"] == "managed"
    assert capabilities["locations"]["remote_backend"]["notice"] == "Billing or quota is enforced by this deployment."
    assert capabilities["providers"]["aliyun"]["configured"] is True
    assert capabilities["providers"]["aliyun"]["default_image_configured"] is True
    assert capabilities["providers"]["aliyun"]["presets"][0]["id"] == "aliyun-a10-recommended"
    assert capabilities["providers"]["autodl"]["configured"] is True


def test_train_session_capabilities_hide_unconfigured_remote_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(tmp_path)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_ACCESS_KEY_SECRET", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_REGION_ID", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_OSS_BUCKET", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_OSS_ENDPOINT", raising=False)
    monkeypatch.delenv("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", raising=False)
    monkeypatch.delenv("ROBOCLAW_AUTODL_HOST", raising=False)

    capabilities = session.capabilities()

    assert capabilities["locations"]["remote_backend"]["configured"] is False
    assert capabilities["locations"]["remote_backend"]["mode"] == "unavailable"
    assert capabilities["providers"]["local"]["configured"] is True
    assert capabilities["providers"]["aliyun"]["configured"] is False
    assert capabilities["providers"]["autodl"]["configured"] is False
