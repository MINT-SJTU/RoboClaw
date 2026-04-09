"""InferSession - trained policy rollout execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.engine.command_builder import builder_for_arms
from roboclaw.embodied.engine.helpers import (
    _BIMANUAL_ID,
    _arm_id,
    _resolve_action_arms,
    _run,
    _validate_dataset_name,
    dataset_path,
    group_arms,
)

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class InferSession:
    def __init__(self, parent: EmbodiedService):
        self._parent = parent

    async def run_policy(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.sensor.camera import resolve_cameras

        grouped = group_arms(_resolve_action_arms(manifest, kwargs))
        followers = grouped["followers"]
        if not followers:
            return "No follower arm configured."
        if len(followers) not in {1, 2}:
            return f"Unsupported follower arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."

        cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(manifest.cameras)
        policies_root = manifest.snapshot.get("policies", {}).get("root", "")
        checkpoint = kwargs.get("checkpoint_path")
        if not checkpoint:
            source_dataset = kwargs.get("source_dataset", kwargs.get("dataset_name", ""))
            base = str(Path(policies_root) / source_dataset) if source_dataset else policies_root
            checkpoint = str(Path(base).expanduser() / "checkpoints" / "last" / "pretrained_model")

        result = self._resolve_dataset_name(kwargs, "eval")
        if isinstance(result, str):
            return result
        dataset_name, user_specified = result
        if user_specified and not dataset_name.startswith("eval_"):
            dataset_name = f"eval_{dataset_name}"
        ds_root = dataset_path(manifest, dataset_name)
        resume = self._should_resume(user_specified, ds_root)
        controller = builder_for_arms(followers)
        policy_kwargs = {
            "cameras": cameras,
            "policy_path": checkpoint,
            "repo_id": f"local/{dataset_name}",
            "dataset_root": str(ds_root),
            "task": kwargs.get("task", "eval"),
            "num_episodes": kwargs.get("num_episodes", 1),
            "resume": resume,
        }
        if len(followers) == 1:
            follower = followers[0]
            argv = controller.run_policy(
                robot_type=follower.type_name,
                robot_port=follower.port,
                robot_cal_dir=follower.calibration_dir,
                robot_id=_arm_id(follower),
                **policy_kwargs,
            )
            return await _run(LocalLeRobotRunner(), argv)

        from roboclaw.embodied.manifest.helpers import ensure_bimanual_cal_dir

        robot_dir = ensure_bimanual_cal_dir(followers[0], followers[1], "followers")
        argv = controller.run_policy_bimanual(
            robot_id=_BIMANUAL_ID,
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            **policy_kwargs,
        )
        return await _run(LocalLeRobotRunner(), argv)

    def _resolve_dataset_name(
        self,
        kwargs: dict[str, Any],
        prefix: str,
    ) -> tuple[str, bool] | str:
        user_specified = "dataset_name" in kwargs
        if user_specified:
            name = kwargs["dataset_name"]
        else:
            name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        error = _validate_dataset_name(name)
        if error:
            return error
        return name, user_specified

    def _should_resume(self, user_specified: bool, dataset_root: Path) -> bool:
        return user_specified and dataset_root.exists()
