from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from roboclaw.embodied.service.verification import InferenceConfigVerifier


def test_inference_config_verifier_rejects_missing_checkpoint(tmp_path: Path) -> None:
    verifier = InferenceConfigVerifier()

    with pytest.raises(ValueError, match="Checkpoint path does not exist"):
        verifier.verify(
            checkpoint_path=str(tmp_path / "missing"),
            manifest_snapshot=_manifest_snapshot(),
            dataset_local_path=str(tmp_path / "dataset"),
        )


def test_inference_config_verifier_rejects_action_dim_mismatch(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy", action_dim=7)
    verifier = InferenceConfigVerifier()

    with pytest.raises(ValueError, match="action_dim"):
        verifier.verify(
            checkpoint_path=str(checkpoint),
            manifest_snapshot=_manifest_snapshot(),
            dataset_local_path=str(_dataset(tmp_path / "dataset", version="v2.1", repo_id="local/demo")),
        )


def test_inference_config_verifier_warns_on_old_dataset_version(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    checkpoint = _checkpoint(tmp_path / "policy", action_dim=6)
    dataset = _dataset(tmp_path / "dataset", version="v2.0", repo_id="local/demo")
    verifier = InferenceConfigVerifier()

    with caplog.at_level(logging.WARNING):
        verifier.verify(
            checkpoint_path=str(checkpoint),
            manifest_snapshot=_manifest_snapshot(),
            dataset_local_path=str(dataset),
        )

    assert "codebase_version=v2.0 is older than v2.1" in caplog.text


def test_inference_config_verifier_accepts_consistent_config(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy", action_dim=6)
    dataset = _dataset(tmp_path / "dataset", version="v2.1", repo_id="local/demo")
    verifier = InferenceConfigVerifier()

    verifier.verify(
        checkpoint_path=str(checkpoint),
        manifest_snapshot=_manifest_snapshot(),
        dataset_local_path=str(dataset),
    )


def _checkpoint(path: Path, *, action_dim: int) -> Path:
    pretrained = path / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text(json.dumps({
        "action_dim": action_dim,
    }), encoding="utf-8")
    (pretrained / "train_config.json").write_text(json.dumps({
        "policy": {"device": "cuda"},
        "dataset": {"repo_id": "local/demo"},
    }), encoding="utf-8")
    return path


def _dataset(path: Path, *, version: str, repo_id: str) -> Path:
    info_dir = path / "meta"
    info_dir.mkdir(parents=True)
    (info_dir / "info.json").write_text(json.dumps({
        "codebase_version": version,
        "source_dataset": repo_id,
    }), encoding="utf-8")
    return path


def _manifest_snapshot() -> dict[str, object]:
    return {
        "device": "cuda",
        "arms": [
            {"alias": "follower", "type": "so101_follower"},
        ],
        "cameras": [],
        "hands": [],
        "datasets": {},
        "policies": {},
    }
