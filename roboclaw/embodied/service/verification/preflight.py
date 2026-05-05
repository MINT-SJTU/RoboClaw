"""Preflight checks for LeRobot subprocess sessions (inference, record, train)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from roboclaw.embodied.service.verification.types import (
    VerificationRequest,
    VerificationResult,
    Violation,
)

_CONFIG_FILES = (
    "config.json",
    "train_config.json",
    "policy_config.json",
    "preprocessor_config.json",
)
_WEIGHT_PATTERNS = (
    "model.safetensors",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.bin",
)
_MAX_INFERENCE_EPISODES = 1_000
_MAX_EPISODE_TIME_S = 3_600


class Verifier(Protocol):
    """Validate information available before a managed session starts."""

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Return violations that should stop launch."""


class PreflightVerifier:
    """Validate host-visible inference inputs before spawning LeRobot.

    This verifier deliberately does not inspect runtime policy actions. In the
    current architecture, RoboClaw launches LeRobot as a subprocess and only has
    access to argv, manifest state, and local checkpoint files before launch.
    """

    def verify(self, request: VerificationRequest) -> VerificationResult:
        violations: list[Violation] = []
        warnings: list[Violation] = []

        argv = list(request.argv)
        violations.extend(_validate_wrapper_argv(argv))
        policy_path = _policy_path_from_request(request, argv)
        violations.extend(_validate_policy_path(policy_path))
        violations.extend(_validate_dataset_args(argv))
        violations.extend(_validate_resource_limits(request))
        violations.extend(_validate_manifest(request.manifest, request.use_cameras, argv))

        if policy_path and _looks_like_remote_policy_id(policy_path):
            warnings.append(Violation(
                "remote_policy_unchecked",
                f"Policy '{policy_path}' looks like a remote repo id; local checkpoint files were not checked.",
                "checkpoint_path",
            ))

        return VerificationResult(tuple(violations), tuple(warnings))


def _validate_wrapper_argv(argv: Sequence[str]) -> list[Violation]:
    violations: list[Violation] = []
    if not argv:
        return [Violation("empty_argv", "Inference command argv is empty.", "argv")]
    if "roboclaw.embodied.command.wrapper" not in argv:
        violations.append(Violation(
            "missing_wrapper",
            "Inference command must launch roboclaw.embodied.command.wrapper.",
            "argv",
        ))
    wrapper_index = _index_or_none(argv, "roboclaw.embodied.command.wrapper")
    if wrapper_index is not None:
        action_index = wrapper_index + 1
        if action_index >= len(argv) or argv[action_index] != "record":
            violations.append(Violation(
                "unexpected_action",
                "Inference command must use the LeRobot record action.",
                "argv",
            ))
    return violations


def _policy_path_from_request(request: VerificationRequest, argv: Sequence[str]) -> str:
    if request.checkpoint_path:
        return str(request.checkpoint_path)
    return _arg_value(argv, "--policy.path=")


def _validate_policy_path(raw_path: str) -> list[Violation]:
    if not raw_path:
        return [Violation("missing_policy_path", "Inference command is missing --policy.path.", "checkpoint_path")]
    if _looks_like_remote_policy_id(raw_path):
        return []

    path = Path(raw_path).expanduser()
    if not path.exists():
        return [Violation(
            "missing_checkpoint",
            f"Policy checkpoint path does not exist: {path}",
            "checkpoint_path",
        )]
    if not path.is_dir():
        return [Violation(
            "invalid_checkpoint",
            f"Policy checkpoint path must be a directory: {path}",
            "checkpoint_path",
        )]

    violations: list[Violation] = []
    if not _has_any_file(path, _CONFIG_FILES):
        violations.append(Violation(
            "incomplete_checkpoint_config",
            f"Policy checkpoint is missing a recognized config file ({', '.join(_CONFIG_FILES)}): {path}",
            "checkpoint_path",
        ))
    if not _has_any_pattern(path, _WEIGHT_PATTERNS):
        violations.append(Violation(
            "incomplete_checkpoint_weights",
            f"Policy checkpoint is missing model weights ({', '.join(_WEIGHT_PATTERNS)}): {path}",
            "checkpoint_path",
        ))
    return violations


def _validate_dataset_args(argv: Sequence[str]) -> list[Violation]:
    required = (
        "--dataset.repo_id=",
        "--dataset.root=",
        "--dataset.num_episodes=",
        "--dataset.episode_time_s=",
    )
    return [
        Violation("missing_dataset_arg", f"Inference command is missing {prefix.rstrip('=')}.", "argv")
        for prefix in required
        if not _has_prefix(argv, prefix)
    ]


def _validate_resource_limits(request: VerificationRequest) -> list[Violation]:
    violations: list[Violation] = []
    if request.num_episodes < 1:
        violations.append(Violation(
            "invalid_num_episodes",
            "num_episodes must be at least 1 for inference.",
            "num_episodes",
        ))
    if request.num_episodes > _MAX_INFERENCE_EPISODES:
        violations.append(Violation(
            "too_many_episodes",
            f"num_episodes must be <= {_MAX_INFERENCE_EPISODES} for inference preflight.",
            "num_episodes",
        ))
    if request.episode_time_s < 1:
        violations.append(Violation(
            "invalid_episode_time",
            "episode_time_s must be at least 1 for inference.",
            "episode_time_s",
        ))
    if request.episode_time_s > _MAX_EPISODE_TIME_S:
        violations.append(Violation(
            "episode_too_long",
            f"episode_time_s must be <= {_MAX_EPISODE_TIME_S} for inference preflight.",
            "episode_time_s",
        ))
    return violations


def _validate_manifest(manifest: Any, use_cameras: bool, argv: Sequence[str]) -> list[Violation]:
    arms = list(getattr(manifest, "arms", []) or [])
    followers = [arm for arm in arms if _role_value(getattr(arm, "role", "")) == "follower"]
    violations: list[Violation] = []
    if not followers:
        violations.append(Violation(
            "missing_follower",
            "Inference requires at least one follower arm in the manifest.",
            "manifest.arms",
        ))
    if len(followers) not in {0, 1, 2}:
        violations.append(Violation(
            "unsupported_follower_count",
            f"Inference supports 1 or 2 follower arms, got {len(followers)}.",
            "manifest.arms",
        ))
    if len(followers) == 2 and {getattr(arm, "side", "") for arm in followers} != {"left", "right"}:
        violations.append(Violation(
            "invalid_bimanual_sides",
            "Bimanual inference requires one left and one right follower arm.",
            "manifest.arms",
        ))

    cameras = list(getattr(manifest, "cameras", []) or [])
    if use_cameras and not cameras:
        violations.append(Violation(
            "missing_cameras",
            "Inference requested cameras, but no cameras are configured in the manifest.",
            "manifest.cameras",
        ))
    if use_cameras and cameras and not _argv_has_camera_config(argv):
        violations.append(Violation(
            "missing_camera_argv",
            "Inference requested cameras, but argv does not include robot camera configuration.",
            "argv",
        ))
    return violations


def _looks_like_remote_policy_id(raw_path: str) -> bool:
    path = Path(raw_path).expanduser()
    if path.exists() or path.is_absolute():
        return False
    if raw_path.startswith(("~", ".", "/")):
        return False
    parts = raw_path.split("/")
    return len(parts) == 2 and all(parts) and not any(part in {".", ".."} for part in parts)


def _has_any_file(path: Path, names: Iterable[str]) -> bool:
    return any((path / name).is_file() for name in names)


def _has_any_pattern(path: Path, patterns: Iterable[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)


def _arg_value(argv: Sequence[str], prefix: str) -> str:
    for arg in argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return ""


def _has_prefix(argv: Sequence[str], prefix: str) -> bool:
    return any(arg.startswith(prefix) for arg in argv)


def _argv_has_camera_config(argv: Sequence[str]) -> bool:
    return any(".cameras=" in arg or arg.startswith("--robot.cameras=") for arg in argv)


def _index_or_none(argv: Sequence[str], value: str) -> int | None:
    try:
        return list(argv).index(value)
    except ValueError:
        return None


def _role_value(role: Any) -> str:
    value = getattr(role, "value", role)
    return str(value)


# ---------------------------------------------------------------------------
# Record preflight
# ---------------------------------------------------------------------------

_MAX_RECORD_EPISODES = 10_000
_MAX_RECORD_EPISODE_TIME_S = 3_600
_MIN_FPS = 1
_MAX_FPS = 120


class RecordPreflightVerifier:
    """Validate host-visible record inputs before spawning LeRobot.

    Checks:
    - wrapper action is ``record``
    - required dataset args present (repo_id, root, num_episodes, episode_time_s)
    - num_episodes and episode_time_s within sane bounds
    - fps within [1, 120]
    - at least one follower arm in the manifest
    - bimanual setup has consistent left/right sides
    - task string is non-empty
    """

    def verify(self, request: VerificationRequest) -> VerificationResult:
        argv = list(request.argv)
        violations: list[Violation] = []

        violations.extend(_validate_record_wrapper_argv(argv))
        violations.extend(_validate_record_dataset_args(argv))
        violations.extend(_validate_record_limits(request, argv))
        violations.extend(_validate_record_manifest(request.manifest))

        return VerificationResult(tuple(violations), ())


def _validate_record_wrapper_argv(argv: Sequence[str]) -> list[Violation]:
    if not argv:
        return [Violation("empty_argv", "Record command argv is empty.", "argv")]
    wrapper_index = _index_or_none(argv, "roboclaw.embodied.command.wrapper")
    if wrapper_index is None:
        return [Violation(
            "missing_wrapper",
            "Record command must launch roboclaw.embodied.command.wrapper.",
            "argv",
        )]
    action_index = wrapper_index + 1
    if action_index >= len(argv) or argv[action_index] != "record":
        return [Violation(
            "unexpected_action",
            "Record command wrapper action must be 'record'.",
            "argv",
        )]
    return []


def _validate_record_dataset_args(argv: Sequence[str]) -> list[Violation]:
    required = (
        "--dataset.repo_id=",
        "--dataset.root=",
        "--dataset.num_episodes=",
        "--dataset.episode_time_s=",
    )
    return [
        Violation("missing_record_arg", f"Record command is missing {p.rstrip('=')}.", "argv")
        for p in required
        if not _has_prefix(argv, p)
    ]


def _validate_record_limits(request: VerificationRequest, argv: Sequence[str]) -> list[Violation]:
    violations: list[Violation] = []
    if request.num_episodes < 1:
        violations.append(Violation(
            "invalid_num_episodes", "num_episodes must be at least 1.", "num_episodes",
        ))
    if request.num_episodes > _MAX_RECORD_EPISODES:
        violations.append(Violation(
            "too_many_episodes",
            f"num_episodes must be <= {_MAX_RECORD_EPISODES}.",
            "num_episodes",
        ))
    if request.episode_time_s < 1:
        violations.append(Violation(
            "invalid_episode_time", "episode_time_s must be at least 1.", "episode_time_s",
        ))
    if request.episode_time_s > _MAX_RECORD_EPISODE_TIME_S:
        violations.append(Violation(
            "episode_too_long",
            f"episode_time_s must be <= {_MAX_RECORD_EPISODE_TIME_S}.",
            "episode_time_s",
        ))
    fps_raw = _arg_value(argv, "--dataset.fps=")
    if fps_raw:
        try:
            fps = int(fps_raw)
            if fps < _MIN_FPS or fps > _MAX_FPS:
                violations.append(Violation(
                    "invalid_fps",
                    f"fps must be between {_MIN_FPS} and {_MAX_FPS}, got {fps}.",
                    "argv",
                ))
        except ValueError:
            violations.append(Violation("invalid_fps", f"fps is not an integer: {fps_raw!r}.", "argv"))
    task_raw = _arg_value(argv, "--dataset.task=")
    if not task_raw or not task_raw.strip():
        violations.append(Violation(
            "missing_task", "Record command is missing a non-empty --dataset.task.", "argv",
        ))
    return violations


def _validate_record_manifest(manifest: Any) -> list[Violation]:
    arms = list(getattr(manifest, "arms", []) or [])
    followers = [arm for arm in arms if _role_value(getattr(arm, "role", "")) == "follower"]
    violations: list[Violation] = []
    if not followers:
        violations.append(Violation(
            "missing_follower",
            "Recording requires at least one follower arm in the manifest.",
            "manifest.arms",
        ))
    if len(followers) == 2 and {getattr(arm, "side", "") for arm in followers} != {"left", "right"}:
        violations.append(Violation(
            "invalid_bimanual_sides",
            "Bimanual recording requires one left and one right follower arm.",
            "manifest.arms",
        ))
    return violations


# ---------------------------------------------------------------------------
# Train preflight
# ---------------------------------------------------------------------------

from roboclaw.embodied.command.builder import TRAIN_POLICY_TYPES as _TRAIN_POLICY_TYPES

_SUPPORTED_POLICY_TYPES = _TRAIN_POLICY_TYPES
_MIN_STEPS = 1
_MAX_STEPS = 10_000_000


class TrainPreflightVerifier:
    """Validate training inputs before spawning lerobot-train.

    Checks:
    - dataset_name is a non-empty slug
    - policy_type is a supported value
    - steps within [1, 10_000_000]
    - device is a non-empty string
    - dataset directory exists locally (warns if absent, does not block)
    """

    def verify(self, request: VerificationRequest) -> VerificationResult:
        violations: list[Violation] = []
        warnings: list[Violation] = []

        dataset_name = request.metadata.get("dataset_name", "")
        policy_type = request.metadata.get("policy_type", "act")
        steps = request.metadata.get("steps", 100_000)
        device = request.metadata.get("device", "cuda")

        if not dataset_name or not dataset_name.strip():
            violations.append(Violation(
                "missing_dataset_name", "Training requires a non-empty dataset_name.", "dataset_name",
            ))

        if policy_type not in _SUPPORTED_POLICY_TYPES:
            allowed = ", ".join(sorted(_SUPPORTED_POLICY_TYPES))
            violations.append(Violation(
                "unsupported_policy_type",
                f"policy_type '{policy_type}' is not supported. Expected one of: {allowed}.",
                "policy_type",
            ))

        if not isinstance(steps, int) or steps < _MIN_STEPS:
            violations.append(Violation(
                "invalid_steps", f"steps must be an integer >= {_MIN_STEPS}.", "steps",
            ))
        elif steps > _MAX_STEPS:
            violations.append(Violation(
                "too_many_steps", f"steps must be <= {_MAX_STEPS}.", "steps",
            ))

        if not device or not str(device).strip():
            violations.append(Violation(
                "missing_device", "Training requires a non-empty device (e.g. 'cuda', 'cpu').", "device",
            ))

        if request.dataset is not None:
            dataset_path = getattr(request.dataset, "local_path", None)
            if dataset_path and not Path(dataset_path).exists():
                warnings.append(Violation(
                    "dataset_not_local",
                    f"Dataset path does not exist locally: {dataset_path}. "
                    "Training will fail unless the dataset is available at runtime.",
                    "dataset",
                ))

        return VerificationResult(tuple(violations), tuple(warnings))
