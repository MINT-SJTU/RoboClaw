"""TrainSession — detached policy training and job inspection."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.command import ActionError, CommandBuilder, logs_dir
from roboclaw.embodied.training import (
    JobResources,
    TrainingJobRecord,
    TrainingJobState,
    TrainingJobStatus,
    TrainingJobStore,
    TrainingProvider,
    TrainingRequest,
    build_training_backend,
)
from roboclaw.embodied.training.common import arg_value, repo_root

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService
    from roboclaw.data.datasets import DatasetRuntimeRef


class TrainSession:
    """Detached training — NOT a Session subclass.

    Uses runner.run_detached() for background execution.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent
        self._jobs = TrainingJobStore()

    async def train(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        dataset_name = kwargs.get("dataset_name", "default")
        dataset = self._parent.datasets.resolve_runtime_dataset(dataset_name)
        argv = CommandBuilder.train(
            manifest,
            dataset=dataset.runtime,
            policy_type=kwargs.get("policy_type", "act"),
            steps=kwargs.get("steps", 100_000),
            device=kwargs.get("device", "cuda"),
        )
        request = self._build_request(dataset.runtime, argv, kwargs)
        backend = build_training_backend(request.provider)
        submit = await backend.submit(request)
        record = TrainingJobRecord(
            job_id=submit.job_id,
            provider=request.provider,
            created_at=time.time(),
            job_name=request.job_name,
            dataset_name=request.dataset_name,
            dataset_repo_id=request.dataset_repo_id,
            output_dir=str(request.output_dir),
            log_path=submit.log_path,
            remote_job_id=submit.remote_job_id,
            provider_data=dict(submit.provider_data),
        )
        self._jobs.save(record)

        message = submit.message
        if request.wait:
            status = await backend.wait(
                record,
                poll_interval_s=request.poll_interval_s,
                timeout_s=request.timeout_s,
            )
            message += f"\nFinal status: {status.state.value}"
            if request.auto_collect and request.provider is not TrainingProvider.LOCAL:
                if status.state is TrainingJobState.SUCCEEDED:
                    collected = await backend.collect(record, output_dir=str(request.output_dir))
                    if collected:
                        message += (
                            f"\nCollected artifacts to {request.output_dir} "
                            f"({len(collected)} downloaded paths)."
                        )
                else:
                    message += "\nSkipping artifact collection because the remote job did not succeed."
        return message

    async def stop_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.executor import SubprocessExecutor

        job_id = kwargs.get("job_id", "")
        record = self._jobs.load(job_id)
        if record is not None:
            backend = build_training_backend(record.provider)
            return _format_status(await backend.stop(record))
        status = await SubprocessExecutor().stop_job(job_id=job_id, log_dir=logs_dir())
        return "\n".join(f"{key}: {value}" for key, value in status.items())

    async def job_status(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        payload = await self.job_status_payload(manifest, kwargs, tty_handoff)
        return "\n".join(f"{key}: {value}" for key, value in payload.items())

    async def job_status_payload(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> dict[str, Any]:
        from roboclaw.embodied.executor import SubprocessExecutor

        job_id = kwargs.get("job_id", "")
        record = self._jobs.load(job_id)
        if record is not None:
            backend = build_training_backend(record.provider)
            return (await backend.status(record)).to_dict()
        return await SubprocessExecutor().job_status(job_id=job_id, log_dir=logs_dir())

    async def current_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> dict[str, str | int | bool | None]:
        from roboclaw.embodied.executor import SubprocessExecutor

        for record in self._jobs.list():
            backend = build_training_backend(record.provider)
            status = await backend.status(record)
            if status.running:
                return status.to_dict()
        return await SubprocessExecutor().latest_running_job(log_dir=logs_dir())

    async def collect_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        job_id = kwargs.get("job_id", "")
        output_dir = kwargs.get("output_dir", "")
        record = self._jobs.load(job_id)
        if record is None:
            raise ActionError(f"Training job '{job_id}' not found.")
        backend = build_training_backend(record.provider)
        written = await backend.collect(record, output_dir=output_dir or None)
        if not written:
            return f"No artifacts collected for job {job_id}."
        target = output_dir or record.output_dir
        return f"Collected artifacts for job {job_id} into {target}"

    def curve_data(self, job_id: str) -> dict[str, Any]:
        job_id = job_id.strip()
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ValueError("Invalid job_id.")

        from roboclaw.embodied.executor import SubprocessExecutor
        record = self._jobs.load(job_id)
        if record is not None and record.provider is not TrainingProvider.LOCAL:
            raise ValueError("Curve data is only available for local training jobs.")
        if record is not None and record.log_path:
            log_path = Path(record.log_path)
        else:
            log_path = SubprocessExecutor()._job_log_path(job_id, logs_dir())

        try:
            mtime: float | None = log_path.stat().st_mtime
        except FileNotFoundError:
            mtime = None

        best, points = _parse_training_curve(job_id, log_path)
        return {
            "job_id": job_id,
            "log_path": str(log_path),
            "exists": mtime is not None,
            "points": points,
            "last_epoch": points[-1]["epoch"] if points else None,
            "last_loss": points[-1]["loss"] if points else None,
            "best_ep": best["ep"] if best else None,
            "best_loss": best["loss"] if best else None,
            "updated_at": mtime,
        }

    # ── Listing utilities ────────────────────────────────────────────────

    def list_datasets(self, manifest: Manifest | None = None) -> str:
        datasets = [
            ref.to_dict()
            for ref in self._parent.datasets.list_local_datasets()
            if ref.capabilities.can_train
        ]
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

    def capabilities(self) -> dict[str, Any]:
        providers = {
            TrainingProvider.LOCAL.value: {
                "id": TrainingProvider.LOCAL.value,
                "display_name": "Current machine",
                "kind": "current_machine",
                "configured": True,
                "presets": [],
                "supports_image_override": False,
                "supports_resource_overrides": False,
            },
            TrainingProvider.ALIYUN.value: _aliyun_provider_capability(),
            TrainingProvider.AUTODL.value: _autodl_provider_capability(),
        }
        has_remote = any(
            config["configured"]
            for provider, config in providers.items()
            if provider != TrainingProvider.LOCAL.value
        )
        remote_backend = _remote_backend_capability(has_remote)
        return {
            "locations": {
                "current_machine": {"configured": True},
                "remote_backend": remote_backend,
            },
            "providers": providers,
        }

    def _build_request(
        self,
        dataset: DatasetRuntimeRef,
        argv: list[str],
        kwargs: dict[str, Any],
    ) -> TrainingRequest:
        provider = _parse_provider(kwargs.get("provider", TrainingProvider.LOCAL.value))
        output_dir_raw = arg_value(argv, "--output_dir=")
        if not output_dir_raw:
            raise ActionError("Training command is missing --output_dir.")
        dataset_root_raw = arg_value(argv, "--dataset.root=")
        if not dataset_root_raw:
            raise ActionError("Training command is missing --dataset.root.")
        output_dir = Path(output_dir_raw).expanduser()
        dataset_root = Path(dataset_root_raw).expanduser()
        resources = _resolve_resources(provider, kwargs)
        env = _coerce_env(kwargs.get("env", {}))
        env = {**_default_remote_env(), **env}
        dataset_name = kwargs.get("dataset_name", dataset.name)
        return TrainingRequest(
            provider=provider,
            dataset_name=dataset_name,
            dataset_repo_id=dataset.repo_id,
            dataset_local_path=dataset_root,
            train_argv=tuple(argv),
            output_dir=output_dir,
            policy_type=str(kwargs.get("policy_type", "act")),
            steps=int(kwargs.get("steps", 100_000)),
            device=str(kwargs.get("device", "cuda")),
            job_name=str(kwargs.get("job_name", "")) or _default_job_name(dataset_name, provider),
            code_dir=Path(str(kwargs.get("code_dir", repo_root()))).expanduser(),
            entrypoint=str(kwargs.get("entrypoint", "")),
            resources=resources,
            env=env,
            wait=bool(kwargs.get("wait", False)),
            timeout_s=float(kwargs["timeout_s"]) if kwargs.get("timeout_s") is not None else None,
            poll_interval_s=float(kwargs.get("poll_interval_s", 30.0)),
            auto_collect=bool(kwargs.get("auto_collect", True)),
            remote_workdir=str(kwargs.get("remote_workdir", "")),
        )

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


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")
_TRAIN_LOG_RE = re.compile(
    r"step:(?P<step>\S+).*?"
    r"ep:(?P<ep>\d+).*?"
    r"epch:(?P<epch>-?\d+(?:\.\d+)?).*?"
    r"loss:(?P<loss>-?\d+(?:\.\d+)?)"
)
_MAX_CURVE_POINTS = 1000
_TAIL_READ_BLOCK_BYTES = 65_536
_MAX_CACHED_JOBS = 50
_BEST_LOSS_BY_JOB: dict[str, dict[str, float | int]] = {}
_ALIYUN_DEFAULT_PRESET = "aliyun-a10-recommended"
_ALIYUN_PRESETS: dict[str, dict[str, Any]] = {
    _ALIYUN_DEFAULT_PRESET: {
        "label": "Aliyun A10 (Recommended)",
        "summary": "A10 · 1 GPU · 16 CPU · 128 GiB",
        "gpu_type": "A10",
        "gpu_count": 1,
        "cpu_cores": 16,
        "memory_gb": 128,
        "node_count": 1,
    },
}


def _parse_provider(raw: str) -> TrainingProvider:
    try:
        return TrainingProvider(str(raw or TrainingProvider.LOCAL.value))
    except ValueError as exc:
        allowed = ", ".join(provider.value for provider in TrainingProvider)
        raise ActionError(f"Unsupported training provider '{raw}'. Expected one of: {allowed}.") from exc


def _coerce_env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ActionError("train env must be a mapping of string keys to string values.")
    return {str(key): str(value) for key, value in raw.items()}


def _default_remote_env() -> dict[str, str]:
    keys = ("HF_ENDPOINT", "HF_TOKEN", "HTTPS_PROXY", "HTTP_PROXY")
    return {key: value for key in keys if (value := os.environ.get(key))}


def _required_env_names(prefix: str, keys: tuple[str, ...]) -> list[str]:
    return [f"{prefix}{key}" for key in keys]


def _has_required_env(names: list[str]) -> bool:
    return all(os.environ.get(name, "").strip() for name in names)


def _remote_backend_capability(has_remote: bool) -> dict[str, Any]:
    if not has_remote:
        return {
            "configured": False,
            "mode": "unavailable",
            "notice": "",
        }

    raw_mode = os.environ.get("ROBOCLAW_REMOTE_TRAINING_MODE", "").strip().lower()
    mode = raw_mode if raw_mode in {"self_hosted", "managed"} else "self_hosted"
    return {
        "configured": True,
        "mode": mode,
        "notice": os.environ.get("ROBOCLAW_REMOTE_TRAINING_NOTICE", "").strip(),
    }


def _aliyun_provider_capability() -> dict[str, Any]:
    required = _required_env_names(
        "ROBOCLAW_ALIYUN_",
        (
            "ACCESS_KEY_ID",
            "ACCESS_KEY_SECRET",
            "REGION_ID",
            "WORKSPACE_ID",
            "OSS_BUCKET",
            "OSS_ENDPOINT",
        ),
    )
    configured = _has_required_env(required)
    default_image_configured = bool(os.environ.get("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", "").strip())
    presets = [
        {
            "id": preset_id,
            "backend_preset": preset_id,
            "label": str(preset["label"]),
            "summary": str(preset["summary"]),
            "gpu_type": str(preset["gpu_type"]),
            "gpu_count": int(preset["gpu_count"]),
            "cpu_cores": int(preset["cpu_cores"]),
            "memory_gb": int(preset["memory_gb"]),
            "node_count": int(preset["node_count"]),
        }
        for preset_id, preset in sorted(_ALIYUN_PRESETS.items())
    ]
    return {
        "id": TrainingProvider.ALIYUN.value,
        "display_name": "Aliyun",
        "kind": "remote_backend",
        "configured": configured,
        "default_image_configured": default_image_configured,
        "presets": presets,
        "supports_image_override": True,
        "supports_resource_overrides": True,
    }


def _autodl_provider_capability() -> dict[str, Any]:
    configured = bool(os.environ.get("ROBOCLAW_AUTODL_HOST", "").strip())
    return {
        "id": TrainingProvider.AUTODL.value,
        "display_name": "AutoDL",
        "kind": "remote_backend",
        "configured": configured,
        "presets": [],
        "supports_image_override": False,
        "supports_resource_overrides": False,
    }


def _resolve_resources(provider: TrainingProvider, kwargs: dict[str, Any]) -> JobResources:
    if provider is not TrainingProvider.ALIYUN:
        return JobResources(
            gpu_count=int(kwargs.get("gpu_count", 1)),
            gpu_type=str(kwargs.get("gpu_type", "A100")),
            cpu_cores=int(kwargs.get("cpu_cores", 16)),
            memory_gb=int(kwargs.get("memory_gb", 128)),
            node_count=int(kwargs.get("node_count", 1)),
            image=str(kwargs.get("image", "")),
            ecs_spec=str(kwargs.get("ecs_spec", "")),
        )

    preset_name = str(kwargs.get("preset", "")).strip() or _ALIYUN_DEFAULT_PRESET
    preset = _ALIYUN_PRESETS.get(preset_name)
    if preset is None:
        allowed = ", ".join(sorted(_ALIYUN_PRESETS))
        raise ActionError(f"Unsupported Aliyun training preset '{preset_name}'. Expected one of: {allowed}.")

    image = str(kwargs.get("image", "")).strip() or os.environ.get("ROBOCLAW_ALIYUN_DEFAULT_IMAGE", "").strip()
    if not image:
        raise ActionError(
            "Aliyun training image is not configured. "
            "Set ROBOCLAW_ALIYUN_DEFAULT_IMAGE on the RoboClaw server or provide an image override."
        )

    return JobResources(
        gpu_count=int(kwargs.get("gpu_count") or preset["gpu_count"]),
        gpu_type=str(kwargs.get("gpu_type") or preset["gpu_type"]),
        cpu_cores=int(kwargs.get("cpu_cores") or preset["cpu_cores"]),
        memory_gb=int(kwargs.get("memory_gb") or preset["memory_gb"]),
        node_count=int(kwargs.get("node_count") or preset["node_count"]),
        image=image,
        ecs_spec=str(kwargs.get("ecs_spec", "")),
    )


def _default_job_name(dataset_name: str, provider: TrainingProvider) -> str:
    return f"{dataset_name}-{provider.value}-{int(time.time())}"


def _format_status(status: TrainingJobStatus) -> str:
    return "\n".join(f"{key}: {value}" for key, value in status.to_dict().items())


def _update_best(
    best: dict[str, float | int] | None, loss: float, ep: int,
) -> dict[str, float | int]:
    if best is None or loss < best["loss"] or (loss == best["loss"] and ep < best["ep"]):
        return {"loss": loss, "ep": ep}
    return best


def _parse_training_curve(job_id: str, log_path: Path) -> tuple[dict[str, float | int] | None, list[dict[str, Any]]]:
    if not log_path.exists():
        return _BEST_LOSS_BY_JOB.get(job_id), []

    points: deque[dict[str, Any]] = deque()
    best = _BEST_LOSS_BY_JOB.get(job_id)
    with log_path.open("rb") as handle:
        file_size = handle.seek(0, 2)
        position = file_size
        remainder = b""

        while position > 0 and len(points) < _MAX_CURVE_POINTS:
            read_size = min(_TAIL_READ_BLOCK_BYTES, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)

            data = block + remainder
            lines = data.split(b"\n")

            if position > 0:
                remainder = lines[0]
                lines = lines[1:]
            else:
                remainder = b""

            for raw_line in reversed(lines):
                point = _parse_training_curve_line(raw_line.decode("utf-8", errors="replace"))
                if point is None:
                    continue
                points.appendleft(point)
                best = _update_best(best, point["loss"], point["ep"])
                if len(points) >= _MAX_CURVE_POINTS:
                    break

        if remainder and len(points) < _MAX_CURVE_POINTS:
            point = _parse_training_curve_line(remainder.decode("utf-8", errors="replace"))
            if point is not None:
                points.appendleft(point)
                best = _update_best(best, point["loss"], point["ep"])

    points_list = list(points)
    if best is not None:
        if len(_BEST_LOSS_BY_JOB) >= _MAX_CACHED_JOBS:
            oldest = next(iter(_BEST_LOSS_BY_JOB))
            del _BEST_LOSS_BY_JOB[oldest]
        _BEST_LOSS_BY_JOB[job_id] = best

    return best, points_list


def _parse_training_curve_line(line: str) -> dict[str, Any] | None:
    match = _TRAIN_LOG_RE.search(line)
    if not match:
        return None

    try:
        epoch = float(match.group("epch"))
        loss = float(match.group("loss"))
        ep = int(match.group("ep"))
    except ValueError:
        return None

    return {
        "step": match.group("step"),
        "ep": ep,
        "epoch": epoch,
        "loss": loss,
    }
