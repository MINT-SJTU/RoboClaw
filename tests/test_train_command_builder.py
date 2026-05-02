from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from roboclaw.data.datasets import DatasetRuntimeRef
from roboclaw.embodied.command.builder import CommandBuilder
from roboclaw.embodied.training.common import rewrite_train_argv


def test_train_uses_lerobot_train_from_current_python_env(tmp_path: Path, monkeypatch) -> None:
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    python_path = env_bin / "python"
    python_path.write_text("", encoding="utf-8")
    lerobot_train = env_bin / "lerobot-train"
    lerobot_train.write_text("", encoding="utf-8")

    manifest = SimpleNamespace(snapshot={"policies": {"root": str(tmp_path / "policies")}})
    dataset = DatasetRuntimeRef(
        name="demo",
        repo_id="local/demo",
        local_path=tmp_path / "datasets" / "local" / "demo",
    )

    monkeypatch.setattr("roboclaw.embodied.command.builder.sys.executable", str(python_path))

    argv = CommandBuilder.train(
        manifest,
        dataset=dataset,
        policy_type="act",
        steps=10,
        device="cpu",
    )

    assert argv[0] == str(lerobot_train)


def test_rewrite_train_argv_normalizes_local_lerobot_train_path() -> None:
    argv = [
        "/Users/pearl/anaconda3/envs/roboclaw312/bin/lerobot-train",
        "--dataset.root=/tmp/local-dataset",
        "--output_dir=/tmp/local-output",
        "--steps=20",
    ]

    rewritten = rewrite_train_argv(
        argv,
        dataset_root="/workspace/dataset",
        output_dir="/workspace/output",
    )

    assert rewritten[0] == "lerobot-train"
    assert "--dataset.root=/workspace/dataset" in rewritten
    assert "--output_dir=/workspace/output" in rewritten
