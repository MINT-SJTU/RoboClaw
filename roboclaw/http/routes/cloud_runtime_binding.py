"""Local SSH runtime binding handlers for cloud training."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import HTTPException

from roboclaw.training import TrainingService

from .cloud_ssh import (
    _clear_ssh_runtime_env,
    _listening_on_local_port,
    _local_runtime_bind_enabled,
    _parse_ssh_command,
    _probe_ssh_banner,
    _read_ssh_runtime_endpoint,
    _restart_local_evo_train_bridge,
    _ssh_runtime_env_path,
    _write_ssh_runtime_env,
)
from .cloud_supervisor import clear_cloud_supervisor_runtime_for_tests
from .train_cloud_helpers import _runtime_configuration_ready
from .train_cloud_schema import CloudSshRuntimeBindRequest

_log = logging.getLogger(__name__)


async def bind_ssh_runtime(
    body: CloudSshRuntimeBindRequest,
    *,
    training: TrainingService,
) -> dict[str, Any]:
    if not _local_runtime_bind_enabled():
        raise HTTPException(status_code=403, detail="local runtime binding is disabled for this deployment")
    if not body.restart_bridge:
        raise HTTPException(
            status_code=400,
            detail="restartBridge must be true so the new SSH runtime is verified before use",
        )
    try:
        host, port, user = _parse_ssh_command(body.ssh_command)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    password = body.password
    key_path = body.key_path.strip()
    if not password and not key_path:
        raise HTTPException(status_code=400, detail="password or keyPath is required")
    previous_endpoint = _read_ssh_runtime_endpoint()
    endpoint = f"{user}@{host}:{port}"
    try:
        banner_attempts = max(3, min(20, int(os.environ.get("ROBOCLAW_SSH_BIND_ATTEMPTS", "5"))))
    except ValueError:
        banner_attempts = 5
    try:
        banner_interval_s = max(1.0, min(5.0, float(os.environ.get("ROBOCLAW_SSH_BIND_INTERVAL_S", "2"))))
    except ValueError:
        banner_interval_s = 2.0
    banner_ready, banner_error = _probe_ssh_banner(
        host,
        port,
        timeout=4.0,
        attempts=banner_attempts,
        interval_s=banner_interval_s,
    )
    if not banner_ready:
        cleared_stale_binding = endpoint == previous_endpoint.get("endpoint", "")
        clear_restart_result: dict[str, Any] = {}
        if cleared_stale_binding:
            _clear_ssh_runtime_env()
            clear_restart_result = _restart_local_evo_train_bridge(_ssh_runtime_env_path())
            clear_cloud_supervisor_runtime_for_tests()
        return {
            "ok": False,
            "saved": False,
            "rolledBack": False,
            "clearedStaleBinding": cleared_stale_binding,
            "host": host,
            "port": port,
            "user": user,
            "endpoint": endpoint,
            "previousEndpoint": previous_endpoint.get("endpoint", ""),
            "activeEndpoint": "" if cleared_stale_binding else previous_endpoint.get("endpoint", ""),
            "authMode": "key" if key_path else "password",
            "envPath": previous_endpoint.get("envPath", str(_ssh_runtime_env_path())),
            "bridge": clear_restart_result or {"restarted": False, "listening": _listening_on_local_port(9000)},
            "runtimeReady": False,
            "gpuReady": False,
            "validation": {},
            "validationError": banner_error,
            "rollback": {
                "restored": False,
                "reason": "current stale binding was cleared" if cleared_stale_binding else "candidate endpoint was rejected before saving",
            },
            "message": (
                f"{endpoint} 未连上，已自动清除这个旧绑定：{banner_error}"
                if cleared_stale_binding
                else f"{endpoint} 未连上，未保存为新的当前实例：{banner_error}"
            ),
        }
    previous_env_path = _ssh_runtime_env_path()
    previous_env_text = previous_env_path.read_text(encoding="utf-8") if previous_env_path.exists() else None
    env_path = _write_ssh_runtime_env(
        host=host,
        port=port,
        user=user,
        password=password,
        key_path=key_path,
    )
    restart_result = _restart_local_evo_train_bridge(env_path)
    validation: dict[str, Any] = {}
    validation_error = ""
    runtime_ready = False
    if restart_result.get("listening"):
        try:
            validation = dict(await training.configuration_check(provider="autodl"))
            runtime_ready, validation_error = _runtime_configuration_ready(validation, require_gpu=False)
            if runtime_ready:
                clear_cloud_supervisor_runtime_for_tests()
        except RuntimeError as exc:
            validation_error = str(exc)
    else:
        validation_error = "EVO_Train bridge did not start after rebinding the SSH runtime"
    rollback_result: dict[str, Any] = {}
    saved = bool(runtime_ready)
    if not runtime_ready:
        if previous_env_text is not None:
            previous_env_path.write_text(previous_env_text, encoding="utf-8")
            try:
                previous_env_path.chmod(0o600)
            except OSError:
                pass
            rollback_result = _restart_local_evo_train_bridge(previous_env_path)
        else:
            try:
                env_path.unlink()
            except FileNotFoundError:
                pass
            rollback_result = {"restored": False, "reason": "no previous runtime binding"}
    gpu_ready = bool(validation.get("gpuReady", validation.get("sshGpuReady", False)))
    _log.info(
        "SSH runtime rebind attempted: endpoint=%s runtime_ready=%s gpu_ready=%s saved=%s error=%s",
        endpoint,
        runtime_ready,
        gpu_ready,
        saved,
        validation_error,
    )
    return {
        "ok": bool(runtime_ready),
        "saved": saved,
        "rolledBack": not saved,
        "host": host,
        "port": port,
        "user": user,
        "endpoint": endpoint,
        "previousEndpoint": previous_endpoint.get("endpoint", ""),
        "activeEndpoint": endpoint if saved else previous_endpoint.get("endpoint", ""),
        "authMode": "key" if key_path else "password",
        "envPath": str(env_path),
        "bridge": restart_result,
        "runtimeReady": runtime_ready,
        "gpuReady": gpu_ready,
        "validation": validation,
        "validationError": validation_error,
        "rollback": rollback_result,
        "message": (
            f"{endpoint} 已连接，GPU 可用"
            if runtime_ready and gpu_ready
            else f"{endpoint} 已连接，可先无卡准备；GPU 任务需开卡或重新绑定有卡实例"
            if runtime_ready
            else f"{endpoint} 未连上，已保留之前的实例配置：{validation_error}"
        ),
    }


async def unbind_ssh_runtime() -> dict[str, Any]:
    if not _local_runtime_bind_enabled():
        raise HTTPException(status_code=403, detail="local runtime binding is disabled for this deployment")
    previous_endpoint = _clear_ssh_runtime_env()
    restart_result = _restart_local_evo_train_bridge(_ssh_runtime_env_path())
    clear_cloud_supervisor_runtime_for_tests()
    _log.info("SSH runtime binding cleared: previous_endpoint=%s", previous_endpoint.get("endpoint", ""))
    return {
        "ok": True,
        "cleared": True,
        "previousEndpoint": previous_endpoint.get("endpoint", ""),
        "activeEndpoint": "",
        "envPath": previous_endpoint.get("envPath", str(_ssh_runtime_env_path())),
        "bridge": restart_result,
        "message": "已清除旧实例绑定。请粘贴当前实例最新 SSH 命令后重新连接。",
    }
