"""TCP bridge for EVO_Train cloud training tasks."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


TERMINAL_STATUSES = {"completed", "deleted", "failed", "stopped", "succeeded", "success"}


class EvoTrainBridgeError(RuntimeError):
    """Raised when RoboClaw cannot talk to EVO_Train."""


@dataclass(frozen=True)
class EvoTrainSettings:
    host: str
    port: int
    timeout_s: float
    provider: str
    username: str
    region: str
    env_file: str
    dataset_root: str
    checkpoint_root: str
    checkpoint_frequency: int
    gpu_count: int
    steps_per_epoch: int

    @classmethod
    def from_env(cls) -> "EvoTrainSettings":
        return cls(
            host=os.environ.get("ROBOCLAW_EVO_TRAIN_HOST", "").strip(),
            port=_env_int("ROBOCLAW_EVO_TRAIN_PORT", 9000),
            timeout_s=_env_float("ROBOCLAW_EVO_TRAIN_TIMEOUT", 10.0),
            provider=os.environ.get("ROBOCLAW_EVO_TRAIN_PROVIDER", "aliyun").strip() or "aliyun",
            username=os.environ.get("ROBOCLAW_EVO_TRAIN_USERNAME", "").strip(),
            region=os.environ.get("ROBOCLAW_EVO_TRAIN_REGION", "cn-hangzhou").strip() or "cn-hangzhou",
            env_file=os.environ.get("ROBOCLAW_EVO_TRAIN_ENV_FILE", "").strip(),
            dataset_root=os.environ.get("ROBOCLAW_EVO_TRAIN_DATASET_ROOT", "").strip(),
            checkpoint_root=os.environ.get("ROBOCLAW_EVO_TRAIN_CHECKPOINT_ROOT", "").strip(),
            checkpoint_frequency=max(1, _env_int("ROBOCLAW_EVO_TRAIN_CHECKPOINT_FREQUENCY", 1)),
            gpu_count=max(1, _env_int("ROBOCLAW_EVO_TRAIN_GPU_COUNT", 1)),
            steps_per_epoch=max(1, _env_int("ROBOCLAW_EVO_TRAIN_STEPS_PER_EPOCH", 10_000)),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.host)


class EvoTrainBridge:
    """High-level bridge between RoboClaw routes and EVO_Train's TCP API."""

    def __init__(self, settings: EvoTrainSettings | None = None) -> None:
        self.settings = settings or EvoTrainSettings.from_env()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def resolve_username(self, explicit_username: str = "") -> str:
        username = explicit_username.strip() or self.settings.username
        if not username:
            raise EvoTrainBridgeError(
                "EVO_Train username is required. Pass `username` from the web client or set "
                "`ROBOCLAW_EVO_TRAIN_USERNAME`."
            )
        return username

    def start_training(
        self,
        service: "EmbodiedService",
        *,
        dataset_name: str = "",
        policy_type: str = "act",
        steps: int = 100_000,
        device: str,
        username: str = "",
        workflow: str = "",
        params: dict[str, Any] | None = None,
        sku_id: str = "",
        image_id: str = "",
        task_name: str = "",
    ) -> dict[str, Any]:
        params = dict(params or {})
        if workflow:
            task_name = task_name or _task_name(workflow.replace("_", "-"), policy_type)
            payload = {
                "username": self.resolve_username(username),
                "taskName": task_name,
                "action": "开始训练",
                "provider": self.settings.provider,
                "workflow": workflow,
                "params": params,
                "skuId": sku_id,
                "imageId": image_id,
            }
        else:
            dataset = service.datasets.resolve_runtime_dataset(dataset_name)
            task_name = task_name or _task_name(dataset.name, policy_type)
            epochs = max(1, round(steps / self.settings.steps_per_epoch))
            payload = {
                "username": self.resolve_username(username),
                "taskName": task_name,
                "action": "开始训练",
                "provider": self.settings.provider,
                "region": self.settings.region,
                "envFile": self.settings.env_file,
                "datasetPath": self._dataset_path(dataset.name, str(dataset.runtime.local_path)),
                "checkpointPath": self._checkpoint_path(service, dataset.name, policy_type),
                "epochs": epochs,
                "checkpointFrequency": self.settings.checkpoint_frequency,
                "gpuCount": self.settings.gpu_count,
                "description": (
                    f"RoboClaw cloud training ({device}, policy={policy_type}, "
                    f"steps={steps}, mapped_epochs={epochs})"
                ),
            }
        if sku_id:
            payload["skuId"] = sku_id
        if image_id:
            payload["imageId"] = image_id
        response = self._request(payload)
        task = _find_task_by_name(response.get("tasks", []), task_name)
        return _task_to_status_payload(
            task,
            fallback_message=str(response.get("message", "")),
            fallback_name=task_name,
        )

    def training_plan(
        self,
        *,
        username: str = "",
        message: str = "",
        workflow: str = "",
        params: dict[str, Any] | None = None,
        provider: str = "",
        sku_id: str = "",
        image_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "username": self.resolve_username(username),
            "action": "AI配置训练",
            "message": message,
            "workflow": workflow,
            "params": dict(params or {}),
            "provider": provider or self.settings.provider,
            "skuId": sku_id,
            "imageId": image_id,
        }
        return self._request(payload)

    def gpu_skus(self, *, provider: str = "", include_incomplete: bool = False) -> dict[str, Any]:
        return self._request(
            {
                "action": "GPU规格查询",
                "provider": provider or self.settings.provider,
                "includeIncomplete": include_incomplete,
            }
        )

    def images(self, *, include_incomplete: bool = False) -> dict[str, Any]:
        return self._request({"action": "AutoDL镜像查询", "includeIncomplete": include_incomplete})

    def runtime_match(
        self,
        *,
        username: str = "",
        provider: str = "",
        params: dict[str, Any] | None = None,
        sku_id: str = "",
        image_id: str = "",
    ) -> dict[str, Any]:
        return self._request(
            {
                "username": self.resolve_username(username),
                "action": "训练运行时匹配",
                "provider": provider or self.settings.provider,
                "params": dict(params or {}),
                "skuId": sku_id,
                "imageId": image_id,
            }
        )

    def stop_training(self, *, job_id: str, username: str = "") -> dict[str, Any]:
        task = self.task_status(job_id=job_id, username=username)
        task_name = str(task.get("task_name") or "")
        if not task_name:
            raise EvoTrainBridgeError(f"EVO_Train task not found for job_id '{job_id}'.")
        response = self._request(
            {
                "username": self.resolve_username(username),
                "taskName": task_name,
                "action": "结束训练",
                "region": self.settings.region,
                "envFile": self.settings.env_file,
            }
        )
        refreshed = _find_task_by_name(response.get("tasks", []), task_name)
        payload = _task_to_status_payload(
            refreshed,
            fallback_message=str(response.get("message", "")),
            fallback_name=task_name,
        )
        payload["running"] = False
        return payload

    def current_task(self, *, username: str = "") -> dict[str, Any]:
        tasks = self.sync_tasks(username=username)
        running_tasks = [task for task in tasks if _is_running_task(task)]
        if not running_tasks:
            return {
                "job_id": "",
                "status": "idle",
                "running": False,
                "message": "",
                "task_name": "",
                "checkpoint_path": "",
                "dataset_path": "",
                "provider": self.settings.provider,
            }
        task = sorted(
            running_tasks,
            key=lambda item: (item.get("updatedAt", ""), item.get("createdAt", ""), item.get("taskName", "")),
            reverse=True,
        )[0]
        return _task_to_status_payload(task)

    def task_status(self, *, job_id: str, username: str = "") -> dict[str, Any]:
        tasks = self.sync_tasks(username=username)
        task = _find_task_by_job_id(tasks, job_id)
        if task is None:
            return {
                "job_id": job_id,
                "status": "missing",
                "running": False,
                "message": "status: missing",
                "task_name": "",
                "checkpoint_path": "",
                "dataset_path": "",
                "provider": self.settings.provider,
            }
        return _task_to_status_payload(task)

    def sync_tasks(self, *, username: str = "") -> list[dict[str, Any]]:
        response = self._request(
            {
                "username": self.resolve_username(username),
                "action": "任务同步",
                "region": self.settings.region,
                "envFile": self.settings.env_file,
            }
        )
        tasks = response.get("tasks", [])
        return [task for task in tasks if isinstance(task, dict)]

    def list_policy_entries(self, *, username: str = "") -> list[dict[str, Any]]:
        tasks = self.sync_tasks(username=username)
        entries: list[dict[str, Any]] = []
        for task in tasks:
            checkpoint_path = str(task.get("checkpointPath") or "").strip()
            if not checkpoint_path:
                continue
            checkpoint = _resolve_policy_checkpoint(Path(checkpoint_path).expanduser())
            deployable = _is_checkpoint_available(checkpoint)
            entry = {
                "name": str(task.get("taskName") or checkpoint.name or "cloud-task"),
                "checkpoint": str(checkpoint),
                "dataset": str(task.get("datasetPath") or ""),
                "steps": None,
                "source": "cloud",
                "deployable": deployable,
                "provider": str(task.get("provider") or self.settings.provider),
                "status": str(task.get("status") or ""),
                "job_id": str(task.get("jobId") or ""),
                "task_name": str(task.get("taskName") or ""),
                "updated_at": str(task.get("updatedAt") or ""),
            }
            entries.append(entry)
        entries.sort(key=lambda item: (item.get("updated_at", ""), item.get("name", "")), reverse=True)
        return entries

    def _dataset_path(self, dataset_name: str, fallback_local_path: str) -> str:
        if self.settings.dataset_root:
            return str(Path(self.settings.dataset_root).expanduser() / dataset_name)
        return fallback_local_path

    def _checkpoint_path(self, service: "EmbodiedService", dataset_name: str, policy_type: str) -> str:
        root = self.settings.checkpoint_root
        if not root:
            root = str(service.manifest.snapshot.get("policies", {}).get("root", "") or "")
        if not root:
            raise EvoTrainBridgeError(
                "Missing checkpoint root. Set `ROBOCLAW_EVO_TRAIN_CHECKPOINT_ROOT` or configure "
                "the RoboClaw policies root before starting cloud training."
            )
        output_dir_name = dataset_name if policy_type == "act" else f"{dataset_name}_{policy_type}"
        return str(Path(root).expanduser() / output_dir_name)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise EvoTrainBridgeError("EVO_Train bridge is not enabled.")
        request_text = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            with socket.create_connection(
                (self.settings.host, self.settings.port),
                timeout=self.settings.timeout_s,
            ) as sock:
                sock.settimeout(self.settings.timeout_s)
                sock.sendall(request_text.encode("utf-8"))
                response = _recv_until_newline(sock)
        except OSError as exc:
            raise EvoTrainBridgeError(
                f"Unable to reach EVO_Train at {self.settings.host}:{self.settings.port}: {exc}"
            ) from exc

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise EvoTrainBridgeError(f"Invalid EVO_Train response: {response!r}") from exc
        if not isinstance(data, dict):
            raise EvoTrainBridgeError(f"Unexpected EVO_Train response payload: {data!r}")
        return data


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _recv_until_newline(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return text.split("\n", 1)[0].strip()


def _task_name(dataset_name: str, policy_type: str) -> str:
    base = dataset_name if policy_type == "act" else f"{dataset_name}_{policy_type}"
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{base}-{stamp}"


def _find_task_by_name(tasks: list[dict[str, Any]], task_name: str) -> dict[str, Any] | None:
    for task in tasks:
        if str(task.get("taskName") or "") == task_name:
            return task
    return None


def _find_task_by_job_id(tasks: list[dict[str, Any]], job_id: str) -> dict[str, Any] | None:
    for task in tasks:
        if str(task.get("jobId") or "") == job_id:
            return task
    return None


def _is_running_task(task: dict[str, Any]) -> bool:
    status = str(task.get("status") or "").strip().lower()
    if not status:
        return False
    return status not in TERMINAL_STATUSES


def _task_to_status_payload(
    task: dict[str, Any] | None,
    *,
    fallback_message: str = "",
    fallback_name: str = "",
) -> dict[str, Any]:
    if task is None:
        return {
            "job_id": "",
            "status": "missing",
            "running": False,
            "message": fallback_message or "status: missing",
            "task_name": fallback_name,
            "checkpoint_path": "",
            "dataset_path": "",
            "provider": "",
        }
    status = str(task.get("status") or "")
    running = _is_running_task(task)
    message_lines = [
        f"task_name: {task.get('taskName') or fallback_name}",
        f"job_id: {task.get('jobId') or ''}",
        f"status: {status}",
        f"running: {running}",
    ]
    if task.get("provider"):
        message_lines.append(f"provider: {task.get('provider')}")
    if task.get("datasetPath"):
        message_lines.append(f"dataset_path: {task.get('datasetPath')}")
    if task.get("checkpointPath"):
        message_lines.append(f"checkpoint_path: {task.get('checkpointPath')}")
    if task.get("error"):
        message_lines.append(f"error: {task.get('error')}")
    return {
        "job_id": str(task.get("jobId") or ""),
        "status": status or "missing",
        "running": running,
        "message": "\n".join(message_lines),
        "task_name": str(task.get("taskName") or fallback_name),
        "checkpoint_path": str(task.get("checkpointPath") or ""),
        "dataset_path": str(task.get("datasetPath") or ""),
        "provider": str(task.get("provider") or ""),
        "error": str(task.get("error") or ""),
    }


def _is_checkpoint_available(checkpoint: Path) -> bool:
    if checkpoint.is_dir():
        return True
    if checkpoint.is_file():
        return True
    parent = checkpoint.parent
    return parent.is_dir() and any(parent.glob("*.safetensors"))


def _resolve_policy_checkpoint(checkpoint_root: Path) -> Path:
    candidates = [
        checkpoint_root / "checkpoints" / "last" / "pretrained_model",
        checkpoint_root / "pretrained_model",
        checkpoint_root,
    ]
    for candidate in candidates:
        if _is_checkpoint_available(candidate):
            return candidate
    return candidates[0]
