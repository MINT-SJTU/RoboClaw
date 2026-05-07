"""Preflight checks for LeRobot subprocess inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence

from roboclaw.embodied.embodiment.arm.registry import get_runtime_spec
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
_MIN_DATASET_VERSION = (2, 1)
_VERSION_TOKEN_RE = re.compile(r"\d+")
logger = logging.getLogger(__name__)


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


class InferenceConfigVerifier:
    """Validate checkpoint/dataset consistency before inference starts."""

    def verify(
        self,
        checkpoint_path: str,
        manifest_snapshot: dict[str, Any],
        dataset_local_path: str,
    ) -> None:
        errors: list[str] = []
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            errors.append(f"Checkpoint path does not exist: {checkpoint}")
            self._raise_if_errors(errors)
            return

        pretrained_dir = _pretrained_model_dir(checkpoint)
        config_path = pretrained_dir / "config.json"
        if not config_path.is_file():
            errors.append(f"Checkpoint is missing pretrained_model/config.json: {config_path}")

        train_config_path = pretrained_dir / "train_config.json"
        train_config = _load_json_if_exists(train_config_path)
        if train_config is not None:
            _warn_device_mismatch(train_config, manifest_snapshot)
            _warn_dataset_mismatch(train_config, dataset_local_path)

        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_action_dim = _manifest_action_dim(manifest_snapshot)
            actual_action_dim = _config_action_dim(config)
            if actual_action_dim is None:
                errors.append(f"Checkpoint config is missing action_dim: {config_path}")
            elif actual_action_dim != expected_action_dim:
                errors.append(
                    "Checkpoint action_dim does not match manifest follower motors: "
                    f"{actual_action_dim} != {expected_action_dim}"
                )

        _warn_if_dataset_version_is_old(dataset_local_path)
        self._raise_if_errors(errors)

    def _raise_if_errors(self, errors: list[str]) -> None:
        if errors:
            raise ValueError(" · ".join(errors))


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


def _pretrained_model_dir(checkpoint: Path) -> Path:
    if checkpoint.name == "pretrained_model":
        return checkpoint
    return checkpoint / "pretrained_model"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _warn_device_mismatch(train_config: dict[str, Any], manifest_snapshot: dict[str, Any]) -> None:
    checkpoint_device = str(train_config.get("policy", {}).get("device") or "").strip()
    manifest_device = _manifest_device(manifest_snapshot)
    if checkpoint_device and manifest_device and checkpoint_device != manifest_device:
        logger.warning(
            "Inference config warning: checkpoint policy.device=%s but manifest device=%s",
            checkpoint_device,
            manifest_device,
        )


def _warn_dataset_mismatch(train_config: dict[str, Any], dataset_local_path: str) -> None:
    checkpoint_dataset = str(train_config.get("dataset", {}).get("repo_id") or "").strip()
    current_dataset = _dataset_repo_id_from_local_path(Path(dataset_local_path).expanduser())
    if checkpoint_dataset and current_dataset and checkpoint_dataset != current_dataset:
        logger.warning(
            "Inference config warning: checkpoint dataset.repo_id=%s but current dataset=%s",
            checkpoint_dataset,
            current_dataset,
        )


def _manifest_device(manifest_snapshot: dict[str, Any]) -> str:
    direct = str(manifest_snapshot.get("device") or "").strip()
    if direct:
        return direct
    policies = manifest_snapshot.get("policies", {})
    if isinstance(policies, dict):
        return str(policies.get("device") or "").strip()
    return ""


def _dataset_repo_id_from_local_path(dataset_local_path: Path) -> str:
    info_path = dataset_local_path / "meta" / "info.json"
    if not info_path.is_file():
        return ""
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in ("source_dataset", "repo_id", "dataset_id"):
        value = str(info.get(key) or "").strip()
        if value:
            return value
    return ""


def _manifest_action_dim(manifest_snapshot: dict[str, Any]) -> int:
    arms = list(manifest_snapshot.get("arms", []) or [])
    followers = [arm for arm in arms if _arm_role(arm) == "follower"]
    return sum(_arm_motor_count(arm) for arm in followers)


def _arm_role(arm: dict[str, Any]) -> str:
    role = str(arm.get("role") or "").strip()
    if role:
        return role
    arm_type = str(arm.get("type") or "").strip()
    if arm_type.endswith("_follower"):
        return "follower"
    if arm_type.endswith("_leader"):
        return "leader"
    return ""


def _arm_motor_count(arm: dict[str, Any]) -> int:
    arm_type = str(arm.get("type") or "").strip()
    return len(get_runtime_spec(arm_type).default_joint_names)


def _config_action_dim(config: dict[str, Any]) -> int | None:
    for key in ("action_dim", "max_action_dim"):
        value = config.get(key)
        if isinstance(value, int):
            return value
    output_features = config.get("output_features", {})
    if not isinstance(output_features, dict):
        return None
    action = output_features.get("action", {})
    if not isinstance(action, dict):
        return None
    shape = action.get("shape")
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return shape[0]
    return None


def _warn_if_dataset_version_is_old(dataset_local_path: str) -> None:
    dataset_path = Path(dataset_local_path).expanduser()
    if not dataset_path.exists():
        return
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version") or "").strip()
    if _version_at_least(version, _MIN_DATASET_VERSION):
        return
    logger.warning(
        "Inference config warning: dataset codebase_version=%s is older than v2.1: %s",
        version or "<missing>",
        dataset_path,
    )


def _version_at_least(version: str, minimum: tuple[int, int]) -> bool:
    numbers = [int(token) for token in _VERSION_TOKEN_RE.findall(version)]
    if not numbers:
        return False
    major = numbers[0]
    minor = numbers[1] if len(numbers) > 1 else 0
    return (major, minor) >= minimum
