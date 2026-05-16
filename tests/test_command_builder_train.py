from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from roboclaw.data.datasets import DatasetRuntimeRef
from roboclaw.embodied.command.builder import CommandBuilder


def test_train_uses_current_python_for_lerobot_train(tmp_path: Path) -> None:
    manifest = SimpleNamespace(snapshot={"policies": {"root": str(tmp_path / "policies")}})
    dataset = DatasetRuntimeRef(
        name="libero_full",
        repo_id="local/libero_full",
        local_path=tmp_path / "datasets" / "local" / "libero_full",
    )

    argv = CommandBuilder.train(manifest, dataset=dataset, policy_type="act", steps=123, device="cpu")

    assert argv[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    assert f"--dataset.repo_id={dataset.repo_id}" in argv
    assert f"--dataset.root={dataset.local_path}" in argv
    assert "--policy.type=act" in argv
    assert f"--output_dir={tmp_path / 'policies' / dataset.name}" in argv
