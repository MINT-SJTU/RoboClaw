"""TrainSession — detached policy training and job inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.command import CommandBuilder, logs_dir, validate_dataset_name

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TrainSession:
    """Detached training — NOT a Session subclass.

    Uses runner.run_detached() for background execution.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    async def train(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.executor import SubprocessExecutor

        dataset_name = kwargs.get("dataset_name", "default")
        validate_dataset_name(dataset_name)
        argv = CommandBuilder.train(
            manifest,
            dataset_name=dataset_name,
            steps=kwargs.get("steps", 100_000),
            device=kwargs.get("device", "cuda"),
        )
        job_id = await SubprocessExecutor().run_detached(argv=argv, log_dir=logs_dir())
        return f"Training started. Job ID: {job_id}"

    async def job_status(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.executor import SubprocessExecutor

        job_id = kwargs.get("job_id", "")
        status = await SubprocessExecutor().job_status(job_id=job_id, log_dir=logs_dir())
        return "\n".join(f"{key}: {value}" for key, value in status.items())

    # ── Listing utilities ────────────────────────────────────────────────

    def list_datasets(self, manifest: Manifest | None = None) -> str:
        if manifest is None:
            manifest = self._parent.manifest
            manifest.ensure()
        root = Path(manifest.snapshot.get("datasets", {}).get("root", "")) / "local"
        if not root.exists():
            return "No datasets found."
        datasets = _scan_datasets(root)
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
        policies = _scan_policies(root)
        if not policies:
            return "No policies found."
        return json.dumps(policies, indent=2, ensure_ascii=False)


# ── Private scanning helpers ─────────────────────────────────────────────


def _scan_datasets(root: Path) -> list[dict[str, Any]]:
    """Scan dataset directories under *root* and return summary dicts."""
    datasets: list[dict[str, Any]] = []
    for dataset_dir in sorted(root.iterdir()):
        info_path = dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        datasets.append({
            "name": dataset_dir.name,
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "fps": info.get("fps", 0),
        })
    return datasets


def _scan_policies(root: Path) -> list[dict[str, Any]]:
    """Scan policy directories under *root* and return summary dicts."""
    policies: list[dict[str, Any]] = []
    for policy_dir in sorted(root.iterdir()):
        if not policy_dir.is_dir():
            continue
        last_checkpoint = policy_dir / "checkpoints" / "last" / "pretrained_model"
        if not last_checkpoint.exists():
            continue
        entry: dict[str, Any] = {
            "name": policy_dir.name,
            "checkpoint": str(last_checkpoint),
        }
        _enrich_policy_entry(entry, last_checkpoint)
        policies.append(entry)
    return policies


def _enrich_policy_entry(entry: dict[str, Any], checkpoint_dir: Path) -> None:
    """Add dataset and steps info from train_config.json if present."""
    train_config = checkpoint_dir / "train_config.json"
    if not train_config.exists():
        return
    cfg = json.loads(train_config.read_text())
    entry["dataset"] = cfg.get("dataset", {}).get("repo_id", "")
    entry["steps"] = cfg.get("steps", 0)
