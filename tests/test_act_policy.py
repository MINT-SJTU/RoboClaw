from __future__ import annotations

import json
import sys

import pytest

torch = pytest.importorskip("torch")

from roboclaw.research.policies.act import ACTConfig, ACTPolicy
from roboclaw.research.train import main


def test_act_forward_shapes() -> None:
    model = ACTPolicy(ACTConfig())
    state = torch.randn(2, 6)
    actions = torch.randn(2, 10, 6)
    predicted, losses = model(state, actions)
    assert predicted.shape == (2, 10, 6)
    assert losses["l1_loss"].ndim == 0


def test_act_predict_returns_dict() -> None:
    action = ACTPolicy(ACTConfig()).predict({"joint_positions": {"a": 1.0, "b": 2.0}})
    assert isinstance(action, dict)
    assert set(action) >= {"a", "b"}


def test_train_main_runs_one_epoch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    rows = [
        {"episode_index": 1, "frame_index": i, "state": json.dumps({"joint_positions": {"j0": float(i)}}), "action": json.dumps({"targets": {"j0": float(i + 1)}})}
        for i in range(3)
    ]
    (root / "data" / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", str(root), "--output", str(output), "--epochs", "1", "--batch-size", "2"])
    main()
    assert (output / "policy.pt").exists()
