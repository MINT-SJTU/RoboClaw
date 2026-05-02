"""Tests for the Alibaba Cloud training integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from roboclaw.embodied.training.cloud.aliyun import (
    AliyunCloudTrainer,
    AliyunTrainingConfig,
    JobResources,
    JobStatus,
    _compose_command,
)


@pytest.fixture
def config() -> AliyunTrainingConfig:
    return AliyunTrainingConfig(
        access_key_id="ak-test",
        access_key_secret="sk-test",
        region_id="cn-hangzhou",
        workspace_id="ws-test",
        oss_bucket="rc-bucket",
        oss_endpoint="oss-cn-hangzhou.aliyuncs.com",
        oss_prefix="roboclaw-training",
    )


@pytest.fixture
def dlc_client() -> MagicMock:
    client = MagicMock(name="DlcClient")
    client.create_job.return_value = SimpleNamespace(body=SimpleNamespace(job_id="dlc-job-123"))
    client.get_job.return_value = SimpleNamespace(body=SimpleNamespace(status="Running"))
    client.list_ecs_specs.return_value = SimpleNamespace(
        body=SimpleNamespace(
            ecs_specs=[
                SimpleNamespace(
                    instance_type="ecs.gn7i-c32g1.8xlarge",
                    gpu_type="A10",
                    gpu=1,
                    cpu=32,
                    memory=188,
                    is_available=True,
                ),
                SimpleNamespace(
                    instance_type="ecs.gn6e-c12g1.3xlarge",
                    gpu_type="V100",
                    gpu=1,
                    cpu=12,
                    memory=92,
                    is_available=True,
                ),
                SimpleNamespace(
                    instance_type="ecs.gn7i-c64g2.16xlarge",
                    gpu_type="A10",
                    gpu=2,
                    cpu=64,
                    memory=376,
                    is_available=True,
                ),
            ],
            total_count=3,
        )
    )
    return client


@pytest.fixture
def oss_bucket() -> MagicMock:
    bucket = MagicMock(name="OssBucket")
    bucket.put_object_from_file.return_value = None
    bucket.put_object.return_value = None
    return bucket


@pytest.fixture
def trainer(config, dlc_client, oss_bucket) -> AliyunCloudTrainer:
    return AliyunCloudTrainer(config, dlc_client=dlc_client, oss_bucket=oss_bucket)


class TestJobStatus:
    def test_from_raw_maps_running(self):
        assert JobStatus.from_raw("Running") is JobStatus.RUNNING

    def test_from_raw_maps_succeeded(self):
        assert JobStatus.from_raw("Succeeded") is JobStatus.SUCCEEDED

    @pytest.mark.parametrize("raw", ["Failed", "Stopped", "Stopping"])
    def test_from_raw_maps_failed_family(self, raw):
        assert JobStatus.from_raw(raw) is JobStatus.FAILED

    @pytest.mark.parametrize("raw", ["Creating", "Queuing", "Dequeued", "", None, "unknown"])
    def test_from_raw_defaults_to_pending(self, raw):
        assert JobStatus.from_raw(raw) is JobStatus.PENDING


class TestSubmitJob:
    def test_uploads_code_and_dataset_archives(self, trainer, oss_bucket, tmp_path: Path):
        code_dir = tmp_path / "code"
        dataset_dir = tmp_path / "dataset"
        code_dir.mkdir()
        dataset_dir.mkdir()
        (code_dir / "train.py").write_text("print('hi')\n", encoding="utf-8")
        (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")

        trainer.submit_job(
            job_name="test-job",
            code_dir=code_dir,
            dataset_dir=dataset_dir,
            entrypoint="python train.py",
            resources=JobResources(gpu_count=1, gpu_type="A10"),
        )

        assert oss_bucket.put_object_from_file.call_count == 2
        keys = [call.args[0] for call in oss_bucket.put_object_from_file.call_args_list]
        assert any(key.startswith("roboclaw-training/code/") for key in keys)
        assert any(key.startswith("roboclaw-training/datasets/") for key in keys)

    def test_creates_dlc_job_with_workspace_and_image(self, trainer, dlc_client, tmp_path: Path):
        code_dir = tmp_path / "code"
        dataset_dir = tmp_path / "dataset"
        code_dir.mkdir()
        dataset_dir.mkdir()
        (code_dir / "train.py").write_text("x\n", encoding="utf-8")
        (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")

        job_id = trainer.submit_job(
            job_name="test-job",
            code_dir=code_dir,
            dataset_dir=dataset_dir,
            entrypoint="python train.py",
            resources=JobResources(gpu_count=2, gpu_type="A10", node_count=1),
        )

        assert job_id == "dlc-job-123"
        request = dlc_client.create_job.call_args.args[0]
        assert request.workspace_id == "ws-test"
        assert request.display_name == "test-job"
        assert "python train.py" in request.user_command
        assert request.job_specs[0].image.startswith("registry.cn-hangzhou.aliyuncs.com/")
        assert request.job_specs[0].ecs_spec == "ecs.gn7i-c64g2.16xlarge"
        assert request.job_specs[0].resource_config is None

    def test_prefers_explicit_ecs_spec(self, trainer, dlc_client, tmp_path: Path):
        code_dir = tmp_path / "code"
        dataset_dir = tmp_path / "dataset"
        code_dir.mkdir()
        dataset_dir.mkdir()
        (code_dir / "train.py").write_text("x\n", encoding="utf-8")
        (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")

        trainer.submit_job(
            job_name="test-job",
            code_dir=code_dir,
            dataset_dir=dataset_dir,
            entrypoint="python train.py",
            resources=JobResources(ecs_spec="ecs.custom-spec", gpu_count=1, gpu_type="A100"),
        )

        request = dlc_client.create_job.call_args.args[0]
        assert request.job_specs[0].ecs_spec == "ecs.custom-spec"

    def test_rejects_unavailable_gpu_type_with_clear_message(self, trainer, tmp_path: Path):
        code_dir = tmp_path / "code"
        dataset_dir = tmp_path / "dataset"
        code_dir.mkdir()
        dataset_dir.mkdir()
        (code_dir / "train.py").write_text("x\n", encoding="utf-8")
        (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Available gpu types: A10, V100"):
            trainer.submit_job(
                job_name="test-job",
                code_dir=code_dir,
                dataset_dir=dataset_dir,
                entrypoint="python train.py",
                resources=JobResources(gpu_count=1, gpu_type="A100"),
            )

    def test_writes_output_marker_to_oss(self, trainer, oss_bucket, tmp_path: Path):
        code_dir = tmp_path / "code"
        dataset_dir = tmp_path / "dataset"
        code_dir.mkdir()
        dataset_dir.mkdir()
        (code_dir / "train.py").write_text("x\n", encoding="utf-8")
        (dataset_dir / "meta.json").write_text("{}", encoding="utf-8")

        trainer.submit_job(
            job_name="test-job",
            code_dir=code_dir,
            dataset_dir=dataset_dir,
            entrypoint="python train.py",
            resources=JobResources(gpu_count=1, gpu_type="A10"),
        )

        oss_bucket.put_object.assert_called_once()
        marker_key, payload = oss_bucket.put_object.call_args.args
        assert marker_key == "roboclaw-training/markers/dlc-job-123.txt"
        assert payload.decode("utf-8").startswith("roboclaw-training/outputs/")

    def test_rejects_missing_code_dir(self, trainer, tmp_path: Path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            trainer.submit_job(
                job_name="x",
                code_dir=tmp_path / "does-not-exist",
                dataset_dir=dataset_dir,
                entrypoint="python train.py",
            )


class TestGetJobStatus:
    def test_returns_mapped_status(self, trainer, dlc_client):
        dlc_client.get_job.return_value = SimpleNamespace(body=SimpleNamespace(status="Succeeded"))
        assert trainer.get_job_status("dlc-job-123") is JobStatus.SUCCEEDED


class TestWaitForJob:
    def test_returns_when_terminal(self, trainer, dlc_client):
        dlc_client.get_job.return_value = SimpleNamespace(body=SimpleNamespace(status="Succeeded"))
        result = trainer.wait_for_job("dlc-job-123", poll_interval=0)
        assert result is JobStatus.SUCCEEDED

    def test_timeout_raises(self, trainer, dlc_client):
        dlc_client.get_job.return_value = SimpleNamespace(body=SimpleNamespace(status="Running"))
        with pytest.raises(TimeoutError):
            trainer.wait_for_job("dlc-job-123", poll_interval=0, timeout=0.05)


class TestDownloadArtifacts:
    def test_downloads_all_objects_under_prefix(self, trainer, oss_bucket, monkeypatch, tmp_path: Path):
        import oss2

        stream_mock = MagicMock()
        stream_mock.read.return_value = b"roboclaw-training/outputs/dlc-job-123/"
        oss_bucket.get_object.return_value = stream_mock

        keys = [
            SimpleNamespace(key="roboclaw-training/outputs/dlc-job-123/ckpt/step_100.pt"),
            SimpleNamespace(key="roboclaw-training/outputs/dlc-job-123/logs/train.log"),
        ]
        monkeypatch.setattr(oss2, "ObjectIterator", lambda bucket, prefix="": iter(keys))

        written = trainer.download_artifacts("dlc-job-123", tmp_path)

        assert len(written) == 2
        get_calls = [call.args for call in oss_bucket.get_object_to_file.call_args_list]
        assert any("ckpt/step_100.pt" in key for key, _ in get_calls)
        assert any("logs/train.log" in key for key, _ in get_calls)


class TestCancelJob:
    def test_calls_stop_job(self, trainer, dlc_client):
        trainer.cancel_job("dlc-job-123")
        dlc_client.stop_job.assert_called_once()


class TestComposeCommand:
    def test_includes_all_stages(self):
        cmd = _compose_command(
            code_oss_uri="oss://rc-bucket/code.tar.gz",
            dataset_oss_uri="oss://rc-bucket/dataset.tar.gz",
            output_oss_uri="oss://rc-bucket/outputs/",
            user_entrypoint="python train.py --cfg a.yaml",
        )
        assert 'OSSUTIL_BIN=$(command -v ossutil64 || command -v ossutil || true)' in cmd
        assert '"$OSSUTIL_BIN" cp -f oss://rc-bucket/code.tar.gz' in cmd
        assert '"$OSSUTIL_BIN" cp -f oss://rc-bucket/dataset.tar.gz' in cmd
        assert "tar -xzf /workspace/roboclaw_training/code.tar.gz" in cmd
        assert "python train.py --cfg a.yaml" in cmd
        assert '"$OSSUTIL_BIN" cp -r -f /workspace/roboclaw_training/outputs/' in cmd


class TestConfigFromEnv:
    def test_reads_required_vars(self, monkeypatch):
        env = {
            "ROBOCLAW_ALIYUN_ACCESS_KEY_ID": "ak",
            "ROBOCLAW_ALIYUN_ACCESS_KEY_SECRET": "sk",
            "ROBOCLAW_ALIYUN_REGION_ID": "cn-hangzhou",
            "ROBOCLAW_ALIYUN_WORKSPACE_ID": "ws-1",
            "ROBOCLAW_ALIYUN_OSS_BUCKET": "bkt",
            "ROBOCLAW_ALIYUN_OSS_ENDPOINT": "oss-cn-hangzhou.aliyuncs.com",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        cfg = AliyunTrainingConfig.from_env()

        assert cfg.access_key_id == "ak"
        assert cfg.region_id == "cn-hangzhou"
        assert cfg.workspace_id == "ws-1"
        assert cfg.oss_prefix == "roboclaw-training"

    def test_raises_on_missing_required(self, monkeypatch):
        monkeypatch.delenv("ROBOCLAW_ALIYUN_ACCESS_KEY_ID", raising=False)
        with pytest.raises(ValueError, match="ACCESS_KEY_ID"):
            AliyunTrainingConfig.from_env()
