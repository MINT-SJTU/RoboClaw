"""Tests for the AutoDL SSH-based training backend helpers."""

from __future__ import annotations

import pytest

from roboclaw.embodied.training.cloud.autodl import (
    AutoDLTrainingConfig,
    _compose_status_script,
    _compose_submit_script,
    _parse_status_output,
)


class TestConfigFromEnv:
    def test_reads_required_vars(self, monkeypatch):
        monkeypatch.setenv("ROBOCLAW_AUTODL_HOST", "gpu.autodl.test")
        cfg = AutoDLTrainingConfig.from_env()
        assert cfg.host == "gpu.autodl.test"
        assert cfg.user == "root"
        assert cfg.port == 22

    def test_raises_on_missing_host(self, monkeypatch):
        monkeypatch.delenv("ROBOCLAW_AUTODL_HOST", raising=False)
        with pytest.raises(ValueError, match="ROBOCLAW_AUTODL_HOST"):
            AutoDLTrainingConfig.from_env()


class TestComposeSubmitScript:
    def test_builds_detached_remote_command(self):
        script = _compose_submit_script(
            remote_root="/root/autodl/rc-job",
            remote_code_archive="/root/autodl/rc-job/code.tar.gz",
            remote_dataset_archive="/root/autodl/rc-job/dataset.tar.gz",
            remote_code_dir="/root/autodl/rc-job/code",
            remote_dataset_dir="/root/autodl/rc-job/dataset",
            remote_output_dir="/root/autodl/rc-job/artifacts",
            remote_log_dir="/root/autodl/rc-job/logs",
            remote_log_path="/root/autodl/rc-job/logs/train.log",
            remote_exit_code="/root/autodl/rc-job/logs/exit_code",
            activate="/opt/conda/bin/activate",
            user_entrypoint="lerobot-train --steps=100",
        )
        assert "tar -xzf /root/autodl/rc-job/code.tar.gz" in script
        assert "tar -xzf /root/autodl/rc-job/dataset.tar.gz" in script
        assert "nohup sh -lc" in script
        assert "lerobot-train --steps=100" in script
        assert "/root/autodl/rc-job/logs/train.log" in script


class TestStatusHelpers:
    def test_compose_status_script(self):
        script = _compose_status_script(
            remote_pid="1234",
            exit_code_path="/root/autodl/rc-job/logs/exit_code",
            remote_log_path="/root/autodl/rc-job/logs/train.log",
        )
        assert "kill -0 1234" in script
        assert "__STATE__" in script
        assert "tail -n 40" in script

    def test_parse_status_output(self):
        state, tail = _parse_status_output("__STATE__:running\n__TAIL__\nhello\nworld\n")
        assert state.value == "running"
        assert tail == "hello\nworld"

