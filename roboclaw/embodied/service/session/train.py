"""TrainSession — detached policy training and job inspection."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.agent.experience import ExperienceRecord, ExperienceStore
from roboclaw.embodied.command import CommandBuilder, logs_dir

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TrainSession:
    """Detached training — NOT a Session subclass.

    Uses runner.run_detached() for background execution.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent
        self._experiences = ExperienceStore(parent.manifest._path.parent.parent)
        self._job_specs: dict[str, dict[str, str]] = {}

    async def start_job_state(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
    ) -> dict[str, str | int | bool | None]:
        from roboclaw.embodied.executor import SubprocessExecutor

        dataset_name = str(kwargs.get("dataset_name", "default") or "default")
        policy_type = str(kwargs.get("policy_type", "act") or "act")
        steps = int(kwargs.get("steps", 100_000) or 100_000)
        device = str(kwargs.get("device", "cuda") or "cuda")
        dataset = self._parent.datasets.resolve_runtime_dataset(dataset_name)
        experience_hint = self._build_experience_hint(dataset_name=dataset_name, policy_type=policy_type)
        argv = CommandBuilder.train(
            manifest,
            dataset=dataset.runtime,
            policy_type=policy_type,
            steps=steps,
            device=device,
        )
        job_id = await SubprocessExecutor().run_detached(argv=argv, log_dir=logs_dir())
        self._job_specs[job_id] = {
            "dataset_name": dataset_name,
            "policy_type": policy_type,
            "dataset_path": str(dataset.runtime.local_path),
            "provider": "local",
        }
        state: dict[str, str | int | bool | None] = {
            "job_id": job_id,
            "status": "running",
            "running": True,
            "pid": None,
            "log_path": str(SubprocessExecutor()._job_log_path(job_id, logs_dir())),
            "log_tail": "",
            "dataset_name": dataset_name,
            "policy_type": policy_type,
            "dataset_path": str(dataset.runtime.local_path),
            "provider": "local",
            "experience_hint": experience_hint,
        }
        state["message"] = self._format_status_message(state)
        self._record_experience(
            job_id=job_id,
            status="submitted",
            log_tail="",
            log_path=str(state.get("log_path") or ""),
        )
        return state

    async def train(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        state = await self.start_job_state(manifest, kwargs)
        return str(state["message"])

    async def stop_job_state(self, job_id: str) -> dict[str, str | int | bool | None]:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().stop_job(job_id=job_id, log_dir=logs_dir())
        return self._enrich_state(job_id, status)

    async def stop_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        job_id = kwargs.get("job_id", "")
        status = await self.stop_job_state(str(job_id))
        return str(status["message"])

    async def job_status_state(self, job_id: str) -> dict[str, str | int | bool | None]:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().job_status(job_id=job_id, log_dir=logs_dir())
        return self._enrich_state(job_id, status)

    async def job_status(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        job_id = kwargs.get("job_id", "")
        status = await self.job_status_state(str(job_id))
        return str(status["message"])

    async def current_job_state(self) -> dict[str, str | int | bool | None]:
        from roboclaw.embodied.executor import SubprocessExecutor

        status = await SubprocessExecutor().latest_running_job(log_dir=logs_dir())
        job_id = str(status.get("job_id") or "")
        if not job_id:
            enriched = self._enrich_state("", status)
            enriched["message"] = self._format_status_message(enriched)
            return enriched
        return self._enrich_state(job_id, status)

    async def current_job(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> dict[str, str | int | bool | None]:
        return await self.current_job_state()

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

    def _build_experience_hint(self, *, dataset_name: str, policy_type: str) -> str:
        records = self._experiences.search(
            task_type="train",
            dataset=dataset_name,
            policy=policy_type,
            provider="local",
            limit=2,
        )
        if not records:
            return ""
        lines = []
        for record in records:
            lesson = record.lesson or record.summary
            lines.append(f"{record.outcome}: {lesson}")
        return "\n".join(lines)

    def _enrich_state(
        self,
        job_id: str,
        state: dict[str, str | int | bool | None],
    ) -> dict[str, str | int | bool | None]:
        enriched = dict(state)
        metadata = self._job_specs.get(job_id, {})
        enriched.setdefault("job_id", job_id)
        enriched.setdefault("provider", metadata.get("provider", "local"))
        enriched.setdefault("dataset_name", metadata.get("dataset_name", ""))
        enriched.setdefault("policy_type", metadata.get("policy_type", ""))
        enriched.setdefault("dataset_path", metadata.get("dataset_path", ""))
        enriched.setdefault("experience_hint", "")
        enriched["message"] = self._format_status_message(enriched)

        status_text = str(enriched.get("status") or "").lower()
        if status_text in _TERMINAL_TRAIN_STATUSES and job_id:
            self._record_experience(
                job_id=job_id,
                status=status_text,
                log_tail=str(enriched.get("log_tail") or ""),
                log_path=str(enriched.get("log_path") or ""),
            )
        return enriched

    def _format_status_message(self, state: dict[str, str | int | bool | None]) -> str:
        order = (
            "job_id",
            "status",
            "running",
            "pid",
            "dataset_name",
            "policy_type",
            "provider",
            "dataset_path",
            "log_path",
            "experience_hint",
        )
        lines: list[str] = []
        seen: set[str] = set()
        for key in order:
            value = state.get(key)
            if value in {None, ""}:
                continue
            seen.add(key)
            lines.append(f"{key}: {value}")
        for key, value in state.items():
            if key in seen or value in {None, ""}:
                continue
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _record_experience(
        self,
        *,
        job_id: str,
        status: str,
        log_tail: str,
        log_path: str,
    ) -> None:
        metadata = self._job_specs.get(job_id, {})
        dataset_name = metadata.get("dataset_name", "")
        policy_type = metadata.get("policy_type", "")
        provider = metadata.get("provider", "local")
        if not dataset_name and not policy_type and not job_id:
            return
        lesson = _status_lesson(status, log_tail)
        summary = (
            f"Local training for dataset '{dataset_name or '<unknown>'}' "
            f"with policy '{policy_type or '<unknown>'}' ended as {status}"
        )
        self._experiences.append(ExperienceRecord.create(
            task_type="train",
            summary=summary,
            outcome=status,
            lesson=lesson,
            dataset=dataset_name,
            policy=policy_type,
            provider=provider,
            job_id=job_id,
            source="train_session",
            error=log_tail if status in {"failed", "error"} else "",
            dataset_path=metadata.get("dataset_path", ""),
            task_name=job_id,
            checkpoint_path=log_path,
        ))

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
_TERMINAL_TRAIN_STATUSES = {"finished", "stopped", "failed", "error", "missing", "idle"}
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


def _status_lesson(status: str, log_tail: str) -> str:
    normalized = status.strip().lower()
    if normalized == "submitted":
        return "A similar run was submitted successfully."
    if normalized == "finished":
        return "A similar run finished previously."
    if normalized == "stopped":
        return "A similar run had to be stopped manually."
    if normalized in {"failed", "error"}:
        tail = log_tail.strip()
        if tail:
            return f"Recent failure signal: {tail.splitlines()[-1]}"
        return "A similar run failed previously."
    if normalized == "missing":
        return "A similar run lost its local process metadata before completion."
    return f"A similar run reached status '{status}'."
