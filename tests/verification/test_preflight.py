from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from roboclaw.embodied.service.verification import PreflightVerifier, VerificationRequest


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path


def _manifest(*, cameras: bool = True, bimanual: bool = False) -> SimpleNamespace:
    if bimanual:
        arms = [
            SimpleNamespace(alias="left_follower", role="follower", side="left"),
            SimpleNamespace(alias="right_follower", role="follower", side="right"),
        ]
    else:
        arms = [SimpleNamespace(alias="follower", role="follower", side="")]
    camera_bindings = [SimpleNamespace(alias="wrist")] if cameras else []
    return SimpleNamespace(arms=arms, cameras=camera_bindings)


def _argv(policy_path: str, *, cameras: bool = True) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "roboclaw.embodied.command.wrapper",
        "record",
        "--robot.type=so101_follower",
        f"--policy.path={policy_path}",
        "--dataset.repo_id=local/eval",
        "--dataset.root=/tmp/eval",
        "--dataset.single_task=eval",
        "--dataset.push_to_hub=false",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=60",
    ]
    if cameras:
        argv.append('--robot.cameras={"wrist": {"type": "opencv"}}')
    return argv


def _replay_argv(*, episode: int = 0, fps: int = 30) -> list[str]:
    return [
        sys.executable,
        "-m",
        "roboclaw.embodied.command.wrapper",
        "replay",
        "--robot.type=so101_follower",
        "--dataset.repo_id=local/demo",
        "--dataset.root=/tmp/demo",
        f"--dataset.episode={episode}",
        f"--dataset.fps={fps}",
    ]


def _codes(result) -> set[str]:
    return {violation.code for violation in result.violations}


def test_preflight_accepts_complete_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy")

    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv(str(checkpoint)),
        manifest=_manifest(),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=True,
    ))

    assert result.ok


def test_preflight_rejects_missing_checkpoint(tmp_path: Path) -> None:
    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv(str(tmp_path / "missing")),
        manifest=_manifest(),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=True,
    ))

    assert "missing_checkpoint" in _codes(result)


def test_preflight_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()

    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv(str(checkpoint)),
        manifest=_manifest(),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=True,
    ))

    assert {"incomplete_checkpoint_config", "incomplete_checkpoint_weights"} <= _codes(result)


def test_preflight_rejects_bad_inference_argv(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy")

    result = PreflightVerifier().verify(VerificationRequest(
        argv=[sys.executable, "-m", "roboclaw.embodied.command.wrapper", "teleoperate", f"--policy.path={checkpoint}"],
        manifest=_manifest(cameras=False),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=False,
    ))

    assert {"unexpected_action", "missing_dataset_arg"} <= _codes(result)


def test_preflight_rejects_invalid_resource_limits(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy")

    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv(str(checkpoint), cameras=False),
        manifest=_manifest(cameras=False),
        num_episodes=0,
        episode_time_s=0,
        use_cameras=False,
    ))

    assert {"invalid_num_episodes", "invalid_episode_time"} <= _codes(result)


def test_preflight_rejects_camera_request_without_camera_config(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy")

    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv(str(checkpoint), cameras=False),
        manifest=_manifest(cameras=True),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=True,
    ))

    assert "missing_camera_argv" in _codes(result)


def test_preflight_allows_remote_policy_ids_without_local_file_check() -> None:
    result = PreflightVerifier().verify(VerificationRequest(
        argv=_argv("lerobot/smolvla-demo", cameras=False),
        manifest=_manifest(cameras=False),
        num_episodes=1,
        episode_time_s=60,
        use_cameras=False,
    ))

    assert result.ok
    assert {warning.code for warning in result.warnings} == {"remote_policy_unchecked"}


def test_preflight_accepts_valid_replay_request() -> None:
    result = PreflightVerifier().verify(VerificationRequest(
        argv=_replay_argv(episode=2, fps=15),
        manifest=_manifest(cameras=False),
        mode="replay",
        episode=2,
        fps=15,
        use_cameras=False,
    ))

    assert result.ok


def test_preflight_rejects_bad_replay_argv() -> None:
    result = PreflightVerifier().verify(VerificationRequest(
        argv=[sys.executable, "-m", "roboclaw.embodied.command.wrapper", "record", "--dataset.root=/tmp/demo"],
        manifest=_manifest(cameras=False),
        mode="replay",
        episode=0,
        fps=30,
        use_cameras=False,
    ))

    assert {"unexpected_action", "missing_dataset_arg"} <= _codes(result)


def test_preflight_rejects_invalid_replay_limits() -> None:
    result = PreflightVerifier().verify(VerificationRequest(
        argv=_replay_argv(episode=-1, fps=0),
        manifest=_manifest(cameras=False),
        mode="replay",
        episode=-1,
        fps=0,
        use_cameras=False,
    ))

    assert {"invalid_replay_episode", "invalid_replay_fps"} <= _codes(result)
