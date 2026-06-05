"""TrainSession — detached policy training and job inspection."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.command import CommandBuilder, logs_dir
from roboclaw.http.dashboard_policies import list_policies as list_policy_entries
from roboclaw.training import TrainingJobStatus, TrainingPolicyEntry, TrainingStartSpec

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TrainSession:
    """Detached training — NOT a Session subclass.

    Uses runner.run_detached() for background execution.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    async def start_job(
        self,
        manifest: Manifest,
        spec: TrainingStartSpec,
    ) -> TrainingJobStatus:
        from roboclaw.embodied.executor import SubprocessExecutor

        dataset = self._parent.datasets.resolve_runtime_dataset(spec.dataset_name)
        argv = CommandBuilder.train(
            manifest,
            dataset=dataset.runtime,
            policy_type=spec.policy_type,
            steps=spec.steps,
            device=spec.device,
        )
        job_id = await SubprocessExecutor().run_detached(argv=argv, log_dir=logs_dir())
        return TrainingJobStatus(
            job_id=job_id,
            status="running",
            running=True,
            message=f"Training started. Job ID: {job_id}",
            mode="local",
        )

    async def stop_job_state(self, job_id: str) -> TrainingJobStatus:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().stop_job(job_id=job_id, log_dir=logs_dir())
        return TrainingJobStatus.from_payload(status, mode="local")

    async def job_status_state(self, job_id: str) -> TrainingJobStatus:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().job_status(job_id=job_id, log_dir=logs_dir())
        return TrainingJobStatus.from_payload(status, mode="local")

    async def current_job_state(self) -> TrainingJobStatus:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().latest_running_job(log_dir=logs_dir())
        return TrainingJobStatus.from_payload(status, mode="local")

    def list_policy_entries(self, manifest: Manifest | None = None) -> list[TrainingPolicyEntry]:
        if manifest is None:
            manifest = self._parent.manifest
            manifest.ensure()
        configured_root = manifest.snapshot.get("policies", {}).get("root", "")
        if configured_root:
            root = Path(configured_root).expanduser()
        else:
            from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home

            root = get_roboclaw_home() / "workspace" / "embodied" / "policies"
        return [
            TrainingPolicyEntry.from_payload(entry, source="local", deployable=True)
            for entry in list_policy_entries(root)
        ]

    async def stop_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        job_id = kwargs.get("job_id", "")
        status = await self.stop_job_state(job_id)
        return status.message

    async def job_status(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        job_id = kwargs.get("job_id", "")
        status = await self.job_status_state(job_id)
        return status.message

    async def current_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> dict[str, str | int | bool | None]:
        return (await self.current_job_state()).to_dict()

    def curve_data(self, job_id: str) -> dict[str, Any]:
        job_id = job_id.strip()
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ValueError("Invalid job_id.")

        from roboclaw.embodied.executor import SubprocessExecutor
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
        policies = [entry.to_dict() for entry in self.list_policy_entries(manifest)]
        if not policies:
            return "No policies found."
        return json.dumps(policies, indent=2, ensure_ascii=False)


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
