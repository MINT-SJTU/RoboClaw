"""TrainSession - detached policy training and job inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.engine.helpers import _logs_dir, _validate_dataset_name, dataset_path

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TrainSession:
    def __init__(self, parent: EmbodiedService):
        self._parent = parent

    async def train(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        dataset_name = kwargs.get("dataset_name", "default")
        error = _validate_dataset_name(dataset_name)
        if error:
            return error
        ds_root = dataset_path(manifest, dataset_name)
        policies_root = manifest.snapshot.get("policies", {}).get("root", "")
        output_dir = Path(policies_root).expanduser() / dataset_name
        steps = kwargs.get("steps", 100_000)
        device = kwargs.get("device", "cuda")
        resume = output_dir.is_dir()
        argv = [
            "lerobot-train",
            f"--dataset.repo_id=local/{dataset_name}",
            f"--dataset.root={ds_root}",
            "--dataset.video_backend=pyav",
            "--policy.type=act",
            "--policy.push_to_hub=false",
            f"--policy.repo_id=local/{dataset_name}",
            f"--output_dir={output_dir}",
            f"--steps={steps}",
            f"--policy.device={device}",
        ]
        if resume:
            argv.append("--resume=true")
            config_path = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
            if config_path.exists():
                argv.append(f"--config_path={config_path}")
        job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_logs_dir())
        return f"Training started. Job ID: {job_id}"

    async def job_status(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        job_id = kwargs.get("job_id", "")
        status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_logs_dir())
        return "\n".join(f"{key}: {value}" for key, value in status.items())

    def list_datasets(self, manifest: Manifest | None = None) -> str:
        if manifest is None:
            manifest = self._parent.manifest
            manifest.ensure()
        root = Path(manifest.snapshot.get("datasets", {}).get("root", "")) / "local"
        if not root.exists():
            return "No datasets found."
        datasets = []
        for dataset_dir in sorted(root.iterdir()):
            info_path = dataset_dir / "meta" / "info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            datasets.append({
                "name": dataset_dir.name,
                "episodes": info.get("total_episodes", 0),
                "frames": info.get("total_frames", 0),
                "fps": info.get("fps", 0),
            })
        if not datasets:
            return "No datasets found."
        return json.dumps(datasets, indent=2, ensure_ascii=False)

    def list_policies(self, manifest: Manifest | None = None) -> str:
        if manifest is None:
            manifest = self._parent.manifest
            manifest.ensure()
        root = Path(manifest.snapshot.get("policies", {}).get("root", ""))
        if not root.exists():
            return "No policies found."
        policies = []
        for policy_dir in sorted(root.iterdir()):
            if not policy_dir.is_dir():
                continue
            last_checkpoint = policy_dir / "checkpoints" / "last" / "pretrained_model"
            if not last_checkpoint.exists():
                continue
            entry = {"name": policy_dir.name, "checkpoint": str(last_checkpoint)}
            train_config = last_checkpoint / "train_config.json"
            if train_config.exists():
                try:
                    cfg = json.loads(train_config.read_text())
                except (json.JSONDecodeError, OSError):
                    cfg = {}
                entry["dataset"] = cfg.get("dataset", {}).get("repo_id", "")
                entry["steps"] = cfg.get("steps", 0)
            policies.append(entry)
        if not policies:
            return "No policies found."
        return json.dumps(policies, indent=2, ensure_ascii=False)
