from __future__ import annotations

import json
import subprocess
import sys

import pytest


def test_launcher_main_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "roboclaw_vla.rl.launcher", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--dataset_path" in result.stdout


def test_adapters_grpo_advantage_shape() -> None:
    torch = pytest.importorskip("torch")
    from roboclaw_vla.rl.adapters import compute_grpo_outcome_advantage

    token_level_rewards = torch.ones((4, 10))
    eos_mask = torch.ones((4, 10))
    index = torch.tensor([0, 0, 1, 1])

    advantages, returns = compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index)

    assert advantages.shape == (4, 10)
    assert returns.shape == (4, 10)


def test_adapters_logprobs_shape() -> None:
    torch = pytest.importorskip("torch")
    from roboclaw_vla.rl.adapters import logprobs_from_logits

    logits = torch.randn((2, 5, 32_000))
    labels = torch.randint(0, 32_000, (2, 5))

    logprobs = logprobs_from_logits(logits, labels)

    assert logprobs.shape == (2, 5)


def test_adapters_masked_mean() -> None:
    torch = pytest.importorskip("torch")
    from roboclaw_vla.rl.adapters import masked_mean

    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.ones_like(values)

    assert torch.allclose(masked_mean(values, mask, axis=1), values.mean(dim=1))


def test_evaluate_main_no_rlinf(tmp_path) -> None:
    artifact_path = tmp_path / "eval"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboclaw_vla.rl.evaluate",
            "--artifact_path",
            str(artifact_path),
            "--checkpoint_path",
            "/tmp/checkpoint",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads((artifact_path / "eval_info.json").read_text(encoding="utf-8"))
    assert payload["implemented"] is True
    assert payload["experimental"] is True
    assert "error" in payload
