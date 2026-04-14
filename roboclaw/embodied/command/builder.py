"""CommandBuilder — manifest + params -> lerobot CLI argv."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.manifest.binding import Binding

from roboclaw.embodied.command.helpers import (
    ActionError,
    dataset_path,
    group_arms,
    logs_dir,
    resolve_action_arms,
    resolve_cameras,
    validate_dataset_name,
)

_BIMANUAL: dict[str, tuple[str, str]] = {
    "so101": ("bi_so_follower", "bi_so_leader"),
    "koch": ("bi_koch_follower", "bi_koch_leader"),
}

_BIMANUAL_ID = "bimanual"

_DEFAULT_REPLAY_ROOT = Path("~/.cache/huggingface/lerobot").expanduser()


# ── Private helper functions ─────────────────────────────────────────────


def _wrapper_args(action: str) -> list[str]:
    """Base argv for invoking lerobot via the wrapper."""
    return [sys.executable, "-m", "roboclaw.embodied.command.wrapper", action]


def _arm_args(prefix: str, binding: Binding) -> list[str]:
    """Single-arm CLI args: --{prefix}.type/port/calibration_dir/id."""
    return [
        f"--{prefix}.type={binding.type_name}",
        f"--{prefix}.id={binding.arm_id}",
        f"--{prefix}.port={binding.port}",
        f"--{prefix}.calibration_dir={Path(binding.calibration_dir).expanduser()}",
    ]


_PREFIX_TO_ROLE = {"robot": "followers", "teleop": "leaders"}


def _bimanual_args(
    prefix: str,
    left: Binding,
    right: Binding,
    type_name: str,
    cameras: list[Binding] | None = None,
) -> list[str]:
    """Bimanual CLI args for one role (robot or teleop).

    Cameras are split by ``binding.side`` into left/right arm configs.
    The ``left_``/``right_`` prefix is stripped from each alias before the
    dict is passed to lerobot, because ``bi_*_follower`` re-applies the
    prefix when assembling its observation feature names.
    """
    from roboclaw.embodied.embodiment.manifest.helpers import ensure_bimanual_cal_dir

    role = _PREFIX_TO_ROLE[prefix]
    cal_dir = ensure_bimanual_cal_dir(left, right, role)
    args = [
        f"--{prefix}.type={type_name}",
        f"--{prefix}.id={_BIMANUAL_ID}",
        f"--{prefix}.calibration_dir={Path(cal_dir).expanduser()}",
        f"--{prefix}.left_arm_config.port={left.port}",
        f"--{prefix}.right_arm_config.port={right.port}",
    ]
    if cameras:
        left_cams = _arm_camera_dict(cameras, "left")
        right_cams = _arm_camera_dict(cameras, "right")
        if left_cams:
            args.append(f"--{prefix}.left_arm_config.cameras={json.dumps(left_cams)}")
        if right_cams:
            args.append(f"--{prefix}.right_arm_config.cameras={json.dumps(right_cams)}")
    return args


def _arm_camera_dict(cameras: list[Binding], side: str) -> dict[str, dict[str, Any]]:
    """Build the lerobot camera dict for one arm of a bimanual robot.

    Filters ``cameras`` by ``side`` and strips the ``{side}_`` prefix from
    each alias so that ``bi_*_follower``'s automatic prefixing produces the
    original alias rather than doubling it.
    """
    side_cams = [c for c in cameras if c.side == side]
    resolved = resolve_cameras(side_cams)
    return {alias.removeprefix(f"{side}_"): cfg for alias, cfg in resolved.items()}


def _camera_args(cameras_dict: dict[str, dict[str, Any]]) -> list[str]:
    """Robot camera CLI arg, if cameras are present."""
    if not cameras_dict:
        return []
    return [f"--robot.cameras={json.dumps(cameras_dict)}"]


def _dataset_args(
    repo_id: str,
    root: Path,
    task: str,
    fps: int,
    num_episodes: int,
    episode_time_s: int = 300,
    reset_time_s: int = 10,
    resume: bool = False,
) -> list[str]:
    """Dataset CLI args for record/infer."""
    args = [
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={task}",
        "--dataset.push_to_hub=false",
        f"--dataset.fps={fps}",
        f"--dataset.num_episodes={num_episodes}",
        "--dataset.vcodec=auto",
        "--dataset.streaming_encoding=true",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
    ]
    if resume:
        args.append("--resume=true")
    return args


def _validate_pairing(
    followers: list[Binding], leaders: list[Binding],
) -> None:
    """Raise ActionError if follower/leader pairing is invalid."""
    if not followers:
        raise ActionError("No follower arms configured.")
    if not leaders:
        raise ActionError("No leader arms configured.")
    if len(followers) != len(leaders):
        raise ActionError(
            f"Follower/leader count mismatch: "
            f"{len(followers)} followers, {len(leaders)} leaders."
        )
    if len(followers) not in {1, 2}:
        raise ActionError(
            f"Unsupported arm count: {len(followers)}. "
            f"Use 1 (single) or 2 (bimanual)."
        )


def _bimanual_family(type_name: str) -> str:
    """Extract the family key (e.g. 'so101') from a type_name like 'so101_follower'."""
    for family in _BIMANUAL:
        if type_name.startswith(family):
            return family
    raise ActionError(
        f"No bimanual mapping for arm type '{type_name}'. "
        f"Supported families: {list(_BIMANUAL.keys())}."
    )


def _robot_argv(
    followers: list[Binding],
    leaders: list[Binding] | None = None,
    cameras: list[Binding] | None = None,
) -> list[str]:
    """Build arm argv for robot (and optionally teleop) role.

    Handles single-arm and bimanual branching in one place.
    """
    args: list[str] = []
    if len(followers) == 1:
        args.extend(_arm_args("robot", followers[0]))
        if leaders:
            args.extend(_arm_args("teleop", leaders[0]))
        if cameras:
            args.extend(_camera_args(resolve_cameras(cameras)))
    else:
        family = _bimanual_family(followers[0].type_name)
        bi_follower, bi_leader = _BIMANUAL[family]
        args.extend(_bimanual_args("robot", followers[0], followers[1], bi_follower, cameras))
        if leaders:
            args.extend(_bimanual_args("teleop", leaders[0], leaders[1], bi_leader))
    return args


# ── CommandBuilder ───────────────────────────────────────────────────────


class CommandBuilder:
    """Pure static class — manifest + params -> lerobot CLI argv."""

    @staticmethod
    def teleop(manifest: Any, *, fps: int = 30, arms: str = "") -> list[str]:
        """Build teleoperation argv (no cameras — only arms)."""
        resolved = resolve_action_arms(manifest, arms)
        grouped = group_arms(resolved)
        _validate_pairing(grouped["followers"], grouped["leaders"])
        argv = _wrapper_args("teleoperate")
        argv.extend(_robot_argv(grouped["followers"], grouped["leaders"]))
        return argv

    @staticmethod
    def record(
        manifest: Any,
        *,
        task: str = "default_task",
        dataset_name: str = "",
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
        arms: str = "",
        use_cameras: bool = True,
    ) -> tuple[list[str], str]:
        """Build recording argv. Returns (argv, dataset_name)."""
        resolved = resolve_action_arms(manifest, arms)
        grouped = group_arms(resolved)
        _validate_pairing(grouped["followers"], grouped["leaders"])

        followers = grouped["followers"]
        leaders = grouped["leaders"]
        cameras = list(manifest.cameras) if use_cameras else []

        # Resolve dataset name
        if not dataset_name:
            dataset_name = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validate_dataset_name(dataset_name)

        ds_path = dataset_path(manifest, dataset_name)
        resume = ds_path.exists()
        repo_id = f"local/{dataset_name}"

        ds_args = _dataset_args(
            repo_id, ds_path, task, fps, num_episodes,
            episode_time_s, reset_time_s, resume,
        )

        argv = _wrapper_args("record")
        argv.extend(_robot_argv(followers, leaders, cameras))
        argv.extend(ds_args)
        return argv, dataset_name

    @staticmethod
    def replay(
        manifest: Any,
        *,
        dataset_name: str,
        episode: int = 0,
        fps: int = 30,
        arms: str = "",
    ) -> list[str]:
        """Build replay argv (follower-only, no teleop/leader)."""
        validate_dataset_name(dataset_name)
        resolved = resolve_action_arms(manifest, arms)
        grouped = group_arms(resolved)
        followers = grouped["followers"]
        if not followers:
            raise ActionError("No follower arms configured for replay.")

        ds_path = dataset_path(manifest, dataset_name, _DEFAULT_REPLAY_ROOT)
        repo_id = f"local/{dataset_name}"

        argv = _wrapper_args("replay")
        argv.extend(_robot_argv(followers))
        argv.extend([
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={ds_path}",
            f"--dataset.episode={episode}",
            f"--dataset.fps={fps}",
        ])
        return argv

    @staticmethod
    def train(
        manifest: Any,
        *,
        dataset_name: str,
        steps: int = 100_000,
        device: str = "cuda",
    ) -> list[str]:
        """Build training argv (standalone lerobot-train, not through wrapper)."""
        validate_dataset_name(dataset_name)
        ds_path = dataset_path(manifest, dataset_name)
        policies_root = manifest.snapshot.get("policies", {}).get("root", "")
        output_dir = Path(policies_root).expanduser() / dataset_name

        argv = [
            "lerobot-train",
            f"--dataset.repo_id=local/{dataset_name}",
            f"--dataset.root={ds_path}",
            "--dataset.video_backend=pyav",
            "--policy.type=act",
            "--policy.push_to_hub=false",
            f"--policy.repo_id=local/{dataset_name}",
            f"--output_dir={output_dir}",
            f"--steps={steps}",
            f"--policy.device={device}",
        ]

        # Resume if a previous checkpoint exists
        if output_dir.is_dir():
            argv.append("--resume=true")
            config_path = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            if config_path.exists():
                argv.append(f"--config_path={config_path}")

        return argv

    @staticmethod
    def infer(
        manifest: Any,
        *,
        checkpoint_path: str = "",
        source_dataset: str = "",
        dataset_name: str = "",
        task: str = "eval",
        num_episodes: int = 1,
        arms: str = "",
        use_cameras: bool = True,
    ) -> list[str]:
        """Build inference argv (uses 'record' action with --policy.path)."""
        if not checkpoint_path:
            policies_root = manifest.snapshot.get("policies", {}).get("root", "")
            base = source_dataset or dataset_name or ""
            if base:
                checkpoint_path = str(
                    Path(policies_root).expanduser() / base
                    / "checkpoints" / "last" / "pretrained_model"
                )
            elif policies_root:
                checkpoint_path = str(
                    Path(policies_root).expanduser()
                    / "checkpoints" / "last" / "pretrained_model"
                )
            else:
                raise ActionError("checkpoint_path is required for inference.")

        resolved = resolve_action_arms(manifest, arms)
        grouped = group_arms(resolved)
        followers = grouped["followers"]
        if not followers:
            raise ActionError("No follower arms configured for inference.")

        cameras = list(manifest.cameras) if use_cameras else []

        # Resolve output dataset name
        if not dataset_name:
            dataset_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validate_dataset_name(dataset_name)

        ds_path = dataset_path(manifest, dataset_name)
        resume = ds_path.exists()
        repo_id = f"local/{dataset_name}"

        policy_args = [
            f"--policy.path={Path(checkpoint_path).expanduser()}",
        ]
        ds_args = [
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={ds_path}",
            f"--dataset.single_task={task}",
            "--dataset.push_to_hub=false",
            f"--dataset.num_episodes={num_episodes}",
        ]
        if resume:
            ds_args.append("--resume=true")

        argv = _wrapper_args("record")
        argv.extend(_robot_argv(followers, cameras=cameras))
        argv.extend(policy_args)
        argv.extend(ds_args)
        return argv

    @staticmethod
    def calibrate(arm: Binding) -> list[str]:
        """Build calibration argv for a single arm."""
        prefix = "teleop" if arm.is_leader else "robot"
        argv = _wrapper_args("calibrate")
        argv.extend(_arm_args(prefix, arm))
        return argv
