"""Tests for RecordPreflightVerifier and TrainPreflightVerifier."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from roboclaw.embodied.service.verification import (
    RecordPreflightVerifier,
    TrainPreflightVerifier,
    VerificationRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest(*, bimanual: bool = False, no_arms: bool = False) -> SimpleNamespace:
    if no_arms:
        return SimpleNamespace(arms=[], cameras=[])
    if bimanual:
        arms = [
            SimpleNamespace(alias="left_follower", role="follower", side="left"),
            SimpleNamespace(alias="right_follower", role="follower", side="right"),
        ]
    else:
        arms = [SimpleNamespace(alias="follower", role="follower", side="")]
    return SimpleNamespace(arms=arms, cameras=[SimpleNamespace(alias="wrist")])


def _record_argv(*, task: str = "pick_cup", fps: int = 30) -> list[str]:
    return [
        sys.executable, "-m", "roboclaw.embodied.command.wrapper", "record",
        "--robot.type=so101_follower",
        "--dataset.repo_id=local/rec",
        "--dataset.root=/tmp/rec",
        "--dataset.num_episodes=10",
        "--dataset.episode_time_s=300",
        f"--dataset.fps={fps}",
        f"--dataset.task={task}",
    ]


def _codes(result) -> set[str]:
    return {v.code for v in result.violations}


def _warn_codes(result) -> set[str]:
    return {v.code for v in result.warnings}


# ---------------------------------------------------------------------------
# RecordPreflightVerifier
# ---------------------------------------------------------------------------

class TestRecordPreflightVerifier:

    def test_accepts_valid_request(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(),
            manifest=_manifest(),
            num_episodes=10,
            episode_time_s=300,
        ))
        assert result.ok

    def test_accepts_bimanual(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(),
            manifest=_manifest(bimanual=True),
            num_episodes=5,
            episode_time_s=120,
        ))
        assert result.ok

    def test_rejects_empty_argv(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=[],
            manifest=_manifest(),
            num_episodes=1,
            episode_time_s=60,
        ))
        assert "empty_argv" in _codes(result)

    def test_rejects_wrong_action(self) -> None:
        argv = _record_argv()
        # replace "record" with "infer"
        idx = argv.index("record")
        argv[idx] = "infer"
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=argv, manifest=_manifest(), num_episodes=1, episode_time_s=60,
        ))
        assert "unexpected_action" in _codes(result)

    def test_rejects_missing_dataset_args(self) -> None:
        argv = [a for a in _record_argv() if not a.startswith("--dataset.repo_id")]
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=argv, manifest=_manifest(), num_episodes=1, episode_time_s=60,
        ))
        assert "missing_record_arg" in _codes(result)

    def test_rejects_zero_episodes(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(), manifest=_manifest(), num_episodes=0, episode_time_s=60,
        ))
        assert "invalid_num_episodes" in _codes(result)

    def test_rejects_episode_time_zero(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(), manifest=_manifest(), num_episodes=1, episode_time_s=0,
        ))
        assert "invalid_episode_time" in _codes(result)

    def test_rejects_fps_out_of_range(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(fps=999), manifest=_manifest(), num_episodes=1, episode_time_s=60,
        ))
        assert "invalid_fps" in _codes(result)

    def test_rejects_empty_task(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(task=""), manifest=_manifest(), num_episodes=1, episode_time_s=60,
        ))
        assert "missing_task" in _codes(result)

    def test_rejects_no_follower_arms(self) -> None:
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(), manifest=_manifest(no_arms=True), num_episodes=1, episode_time_s=60,
        ))
        assert "missing_follower" in _codes(result)

    def test_rejects_bimanual_same_side(self) -> None:
        bad_manifest = SimpleNamespace(arms=[
            SimpleNamespace(alias="left", role="follower", side="left"),
            SimpleNamespace(alias="also_left", role="follower", side="left"),
        ], cameras=[])
        result = RecordPreflightVerifier().verify(VerificationRequest(
            argv=_record_argv(), manifest=bad_manifest, num_episodes=1, episode_time_s=60,
        ))
        assert "invalid_bimanual_sides" in _codes(result)


# ---------------------------------------------------------------------------
# TrainPreflightVerifier
# ---------------------------------------------------------------------------

def _train_request(
    *,
    dataset_name: str = "my_dataset",
    policy_type: str = "act",
    steps: int = 100_000,
    device: str = "cuda",
    dataset_path: str | None = None,
) -> VerificationRequest:
    dataset = None
    if dataset_path is not None:
        dataset = SimpleNamespace(local_path=dataset_path)
    return VerificationRequest(
        argv=[],
        manifest=SimpleNamespace(arms=[], cameras=[]),
        dataset=dataset,
        metadata={
            "dataset_name": dataset_name,
            "policy_type": policy_type,
            "steps": steps,
            "device": device,
        },
    )


class TestTrainPreflightVerifier:

    def test_accepts_valid_request(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request())
        assert result.ok

    def test_rejects_empty_dataset_name(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request(dataset_name=""))
        assert "missing_dataset_name" in _codes(result)

    def test_rejects_unsupported_policy_type(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request(policy_type="unknown_policy"))
        assert "unsupported_policy_type" in _codes(result)

    def test_rejects_zero_steps(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request(steps=0))
        assert "invalid_steps" in _codes(result)

    def test_rejects_steps_over_limit(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request(steps=99_999_999))
        assert "too_many_steps" in _codes(result)

    def test_rejects_empty_device(self) -> None:
        result = TrainPreflightVerifier().verify(_train_request(device=""))
        assert "missing_device" in _codes(result)

    def test_warns_dataset_not_local(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent_dataset")
        result = TrainPreflightVerifier().verify(_train_request(dataset_path=missing))
        assert result.ok  # warning, not violation
        assert "dataset_not_local" in _warn_codes(result)

    def test_no_warning_when_dataset_exists(self, tmp_path: Path) -> None:
        existing = str(tmp_path)
        result = TrainPreflightVerifier().verify(_train_request(dataset_path=existing))
        assert result.ok
        assert "dataset_not_local" not in _warn_codes(result)

    def test_accepts_all_supported_policy_types(self) -> None:
        for pt in ("act", "diffusion", "tdmpc", "vqbet"):
            result = TrainPreflightVerifier().verify(_train_request(policy_type=pt))
            assert result.ok, f"policy_type={pt!r} should be accepted"
