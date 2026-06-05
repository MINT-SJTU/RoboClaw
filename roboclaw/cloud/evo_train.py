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
    billing_mode: str = "external"
    client_token: str = ""
    admin_token: str = ""

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
            billing_mode=os.environ.get("ROBOCLAW_EVO_TRAIN_BILLING_MODE", "external").strip() or "external",
            client_token=os.environ.get("ROBOCLAW_EVO_TRAIN_CLIENT_TOKEN", "").strip(),
            admin_token=(
                os.environ.get("ROBOCLAW_EVO_TRAIN_ADMIN_TOKEN", "").strip()
                or os.environ.get("EVO_TRAIN_ADMIN_TOKEN", "").strip()
            ),
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
        provider: str = "",
        workflow: str = "",
        params: dict[str, Any] | None = None,
        sku_id: str = "",
        image_id: str = "",
        task_name: str = "",
        wait_for_submit: bool = True,
    ) -> dict[str, Any]:
        params = _normalize_bridge_model_source_contract(dict(params or {}))
        selected_provider = (provider or self.settings.provider).strip() or self.settings.provider
        if not workflow and not dataset_name:
            raise ValueError(
                "Cloud training requires either a workflow (e.g. 'rlinf_vla') "
                "or a dataset_name. Neither was provided."
            )
        if workflow:
            task_name = task_name or _task_name(workflow.replace("_", "-"), policy_type)
            payload = {
                "username": self.resolve_username(username),
                "taskName": task_name,
                "action": "开始训练",
                "provider": selected_provider,
                "workflow": workflow,
                "params": params,
                "skuId": sku_id,
                "imageId": image_id,
                "forceRefresh": True,
                "waitForSubmit": wait_for_submit,
            }
        else:
            dataset = service.datasets.resolve_runtime_dataset(dataset_name)
            task_name = task_name or _task_name(dataset.name, policy_type)
            epochs = max(1, round(steps / self.settings.steps_per_epoch))
            payload = {
                "username": self.resolve_username(username),
                "taskName": task_name,
                "action": "开始训练",
                "provider": selected_provider,
                "region": self.settings.region,
                "envFile": self.settings.env_file,
                "datasetPath": self._dataset_path(dataset.name, str(dataset.runtime.local_path)),
                "checkpointPath": self._checkpoint_path(service, dataset.name, policy_type),
                "epochs": epochs,
                "checkpointFrequency": self.settings.checkpoint_frequency,
                "gpuCount": self.settings.gpu_count,
                "forceRefresh": True,
                "waitForSubmit": wait_for_submit,
                "description": (
                    f"RoboClaw cloud training ({device}, policy={policy_type}, "
                    f"steps={steps}, mapped_epochs={epochs})"
                ),
            }
        if self.settings.billing_mode:
            payload["billingMode"] = self.settings.billing_mode
        if sku_id:
            payload["skuId"] = sku_id
        if image_id:
            payload["imageId"] = image_id
        response = self._request(payload)
        if response.get("ok") is False and _is_duplicate_task_error(response):
            task_name = _retry_task_name(task_name)
            payload["taskName"] = task_name
            response = self._request(payload)
        if response.get("ok") is False:
            message = str(response.get("message") or response.get("errorCode") or "EVO_Train start failed")
            raise EvoTrainBridgeError(message)
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
            "params": _normalize_bridge_model_source_contract(dict(params or {})),
            "provider": provider or self.settings.provider,
            "skuId": sku_id,
            "imageId": image_id,
        }
        return self._request(payload)

    def gpu_skus(self, *, provider: str = "", include_incomplete: bool = False, force_refresh: bool = False) -> dict[str, Any]:
        return self._request(
            {
                "action": "GPU规格查询",
                "provider": provider or self.settings.provider,
                "includeIncomplete": include_incomplete,
                "forceRefresh": force_refresh,
            }
        )

    def images(self, *, provider: str = "", include_incomplete: bool = False) -> dict[str, Any]:
        return self._request(
            {
                "action": "AutoDL镜像查询",
                "provider": provider or self.settings.provider,
                "includeIncomplete": include_incomplete,
            }
        )

    def configuration_check(self, *, provider: str = "") -> dict[str, Any]:
        return self._request(
            {
                "action": "配置检查",
                "provider": provider or self.settings.provider,
            }
        )

    def provider_balance(self, *, provider: str = "", minimum_assets: int = 0) -> dict[str, Any]:
        return self._request(
            {
                "action": "平台余额查询",
                "provider": provider or self.settings.provider,
                "minimumAssets": minimum_assets,
                "adminToken": self.settings.admin_token,
            }
        )

    def runtime_match(
        self,
        *,
        username: str = "",
        provider: str = "",
        params: dict[str, Any] | None = None,
        sku_id: str = "",
        image_id: str = "",
        force_refresh: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            {
                "username": self.resolve_username(username),
                "action": "训练运行时匹配",
                "provider": provider or self.settings.provider,
                "params": _normalize_bridge_model_source_contract(dict(params or {})),
                "skuId": sku_id,
                "imageId": image_id,
                "forceRefresh": force_refresh,
            }
        )

    def source_preflight(
        self,
        *,
        username: str = "",
        provider: str = "",
        source: dict[str, Any] | None = None,
        role: str = "dataset",
    ) -> dict[str, Any]:
        return self._request(
            {
                "username": self.resolve_username(username),
                "action": "公开源预检查",
                "provider": provider or self.settings.provider,
                "role": role,
                "source": dict(source or {}),
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
        if running_tasks:
            task = sorted(
                running_tasks,
                key=lambda item: (item.get("updatedAt", ""), item.get("createdAt", ""), item.get("taskName", "")),
                reverse=True,
            )[0]
            return _task_to_status_payload(task)

        if not tasks:
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
            tasks,
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
        request_payload = dict(payload)
        if self.settings.client_token and not any(key in request_payload for key in ("apiToken", "token", "adminToken")):
            request_payload["apiToken"] = self.settings.client_token
        request_text = json.dumps(request_payload, ensure_ascii=False) + "\n"
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


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _nested_model_source(params: dict[str, Any]) -> dict[str, Any]:
    source_contract = _dict_value(params.get("sourceContract"))
    model_source = _dict_value(source_contract.get("modelSource"))
    if model_source:
        return model_source
    training_contract = _dict_value(params.get("trainingContract"))
    sources = _dict_value(training_contract.get("sources"))
    return _dict_value(sources.get("model"))


def _infer_bridge_model_source_type(model_source: dict[str, Any], params: dict[str, Any]) -> str:
    explicit = _first_text(
        model_source.get("sourceType"),
        model_source.get("type"),
        params.get("modelSourceKind"),
    )
    if explicit:
        return _canonical_bridge_model_source_type(explicit)

    model_uri = _first_text(
        model_source.get("uri"),
        model_source.get("modelUri"),
        model_source.get("repoId"),
        model_source.get("repository"),
        model_source.get("checkpoint"),
        model_source.get("path"),
        params.get("checkpointPath"),
        params.get("sftModelPath"),
    ).lower()
    checkpoint_format = _first_text(params.get("checkpointFormat"), model_source.get("format")).lower()
    config_name = _first_text(
        params.get("configName"),
        params.get("rlinfConfigName"),
        params.get("builtinTrainingProfile"),
    )

    if checkpoint_format == "rlinf_config" or (config_name and not model_uri):
        return "builtin_policy"
    if not model_uri:
        return "builtin_policy"
    if model_uri.startswith(("hf://", "huggingface://", "modelscope://", "https://huggingface.co/", "http://", "https://")):
        return "public_model_repo"
    if model_uri.startswith(("s3://", "oss://", "r2://", "gs://", "cos://")):
        return "user_object_storage"
    if model_uri.startswith("/"):
        return "evo_studio_checkpoint"
    return "public_model_repo"


def _canonical_bridge_model_source_type(value: str) -> str:
    text = value.strip().lower().replace("-", "_")
    aliases = {
        "auto": "builtin_policy",
        "catalog_lookup_required": "builtin_policy",
        "rlinf_config": "builtin_policy",
        "rlinf_config_default": "builtin_policy",
        "hf": "public_model_repo",
        "huggingface": "public_model_repo",
        "huggingface_model": "public_model_repo",
        "public_reference": "public_model_repo",
        "public_repo": "public_model_repo",
        "public_model": "public_model_repo",
        "pretrained_model": "public_model_repo",
        "checkpoint_repo": "public_model_repo",
        "object_storage": "user_object_storage",
        "s3": "user_object_storage",
        "oss": "user_object_storage",
        "r2": "user_object_storage",
    }
    return aliases.get(text, text)


def _normalize_bridge_model_source_contract(params: dict[str, Any]) -> dict[str, Any]:
    """Make model-source contracts complete before hitting the EVO_Train TCP API."""

    normalized = dict(params or {})
    raw_model_source = normalized.get("modelSource")
    if isinstance(raw_model_source, dict):
        model_source = dict(raw_model_source)
    elif isinstance(raw_model_source, str) and raw_model_source.strip():
        model_source = {"uri": raw_model_source.strip()}
    else:
        model_source = _nested_model_source(normalized)

    if not _first_text(model_source.get("uri")):
        model_uri = _first_text(model_source.get("modelUri"), model_source.get("repoId"), model_source.get("repository"))
        if model_uri:
            model_source["uri"] = model_uri
    if not _first_text(model_source.get("checkpoint")):
        checkpoint = _first_text(model_source.get("checkpointName"))
        if checkpoint:
            model_source["checkpoint"] = checkpoint
    if not _first_text(model_source.get("modelFamily")):
        model_family = _first_text(
            normalized.get("modelFamily"),
            normalized.get("policyType"),
            normalized.get("policyFamily"),
            normalized.get("modelRegistryName"),
        )
        if model_family:
            model_source["modelFamily"] = model_family

    model_source["sourceType"] = _infer_bridge_model_source_type(model_source, normalized)
    normalized["modelSource"] = model_source
    normalized["modelSourceKind"] = model_source["sourceType"]

    source_contract = _dict_value(normalized.get("sourceContract"))
    if source_contract:
        source_contract["modelSource"] = dict(model_source)
        source_contract["modelSourceKind"] = model_source["sourceType"]
        if _first_text(model_source.get("format")):
            source_contract["checkpointFormat"] = model_source["format"]
        normalized["sourceContract"] = source_contract

    training_contract = _dict_value(normalized.get("trainingContract"))
    if training_contract:
        sources = _dict_value(training_contract.get("sources"))
        sources["model"] = dict(model_source)
        training_contract["sources"] = sources
        normalized["trainingContract"] = training_contract
    return normalized


def _task_name(dataset_name: str, policy_type: str) -> str:
    base = dataset_name if policy_type == "act" else f"{dataset_name}_{policy_type}"
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{base}-{stamp}"


def _retry_task_name(task_name: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{task_name}-retry-{stamp}"


def _is_duplicate_task_error(response: dict[str, Any]) -> bool:
    message = str(response.get("message") or "")
    return "task already exists" in message or "task row already exists" in message


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
    log_path = str(task.get("logPath") or task.get("log_path") or "")
    log_tail = str(task.get("logTail") or task.get("log_tail") or "")
    stage = _latest_evo_stage(log_tail)
    if stage:
        message_lines.append(f"stage: {stage}")
    if log_path:
        message_lines.append(f"log_path: {log_path}")
    if log_tail:
        message_lines.append("log_tail:")
        message_lines.append(log_tail)
    if task.get("error"):
        message_lines.append(f"error: {task.get('error')}")
    failure_remediation = task.get("failureRemediation") if isinstance(task.get("failureRemediation"), dict) else {}
    if failure_remediation:
        message_lines.append(f"failure_remediation: {failure_remediation.get('code') or 'available'}")
    return {
        "job_id": str(task.get("jobId") or ""),
        "status": status or "missing",
        "running": running,
        "message": "\n".join(message_lines),
        "task_name": str(task.get("taskName") or fallback_name),
        "checkpoint_path": str(task.get("checkpointPath") or ""),
        "dataset_path": str(task.get("datasetPath") or ""),
        "provider": str(task.get("provider") or ""),
        "log_path": log_path,
        "log_tail": log_tail,
        "error": str(task.get("error") or ""),
        "failureRemediation": failure_remediation,
    }


def _latest_evo_stage(log_tail: str) -> str:
    stage = ""
    for line in log_tail.splitlines():
        if line.startswith("__EVO_STAGE_START__="):
            stage = line.split("=", 1)[1].strip()
        elif line.startswith("__EVO_STAGE_DONE__="):
            done = line.split("=", 1)[1].strip()
            if done and done == stage:
                stage = ""
    if stage == "prepare_code":
        return "准备代码"
    if stage == "bootstrap_runtime":
        return "准备环境/安装依赖"
    if stage == "resolve_sources":
        return "下载/检查数据和模型"
    if stage == "healthcheck_runtime":
        return "检查运行环境"
    if stage == "preflight":
        return "启动前检查"
    if stage == "write_contract":
        return "写入任务记录"
    if stage == "train_vla_rl_backend":
        return "正在训练/评测"
    if stage == "collect_artifacts":
        return "收集结果"
    return stage


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
