"""Tests for exec tool internal URL blocking."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from roboclaw.agent.tools.shell import ExecTool


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_localhost(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_exec_blocks_curl_metadata():
    tool = ExecTool()
    with patch("roboclaw.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command='curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/'
        )
    assert "Error" in result
    assert "internal" in result.lower() or "private" in result.lower()


@pytest.mark.asyncio
async def test_exec_blocks_wget_localhost():
    tool = ExecTool()
    with patch("roboclaw.security.network.socket.getaddrinfo", _fake_resolve_localhost):
        result = await tool.execute(command="wget http://localhost:8080/secret -O /tmp/out")
    assert "Error" in result


@pytest.mark.asyncio
async def test_exec_allows_normal_commands():
    tool = ExecTool(timeout=5)
    result = await tool.execute(command="echo hello")
    assert "hello" in result
    assert "Error" not in result.split("\n")[0]


def test_exec_blocks_local_training_and_ingest_by_default(monkeypatch):
    monkeypatch.delenv("ROBOCLAW_ALLOW_AGENT_LOCAL_TRAINING", raising=False)
    tool = ExecTool()
    blocked = [
        "git clone https://github.com/example/project.git",
        "python train.py --dataset /tmp/data",
        "torchrun train.py",
        "python -m pip install some-training-package",
        "huggingface-cli download org/dataset",
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
        "whoami && hostname && pwd && date",
    ]

    for command in blocked:
        result = tool._guard_command(command, "/tmp")
        assert result is not None
        assert "blocked" in result.lower()
        assert (
            "evo_studio_cloud_train" in result.lower()
            or "cloud training" in result.lower()
            or "data-pool" in result.lower()
        )


def test_exec_allows_local_training_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("ROBOCLAW_ALLOW_AGENT_LOCAL_TRAINING", "1")
    tool = ExecTool()
    guard_result = tool._guard_command("python train.py --help", "/tmp")
    assert guard_result is None


@pytest.mark.asyncio
async def test_exec_allows_curl_to_public_url():
    """Commands with public URLs should not be blocked by the internal URL check."""
    tool = ExecTool()
    with patch("roboclaw.security.network.socket.getaddrinfo", _fake_resolve_public):
        guard_result = tool._guard_command("curl https://example.com/api", "/tmp")
    assert guard_result is None


@pytest.mark.asyncio
async def test_exec_blocks_chained_internal_url():
    """Internal URLs buried in chained commands should still be caught."""
    tool = ExecTool()
    with patch("roboclaw.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(
            command="echo start && curl http://169.254.169.254/latest/meta-data/ && echo done"
        )
    assert "Error" in result
