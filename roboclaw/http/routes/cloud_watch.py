"""Cloud training supervisor watch loop."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from roboclaw.training import TrainingService, TrainingStopSpec

from .cloud_billing import _release_failed_cloud_hold
from .cloud_supervisor import (
    _auto_repair_policy,
    _cloud_failure_signal,
    _cloud_supervisor_runtime_state,
    _cloud_supervisor_runtime_lock,
    _cloud_supervisor_task_key,
    _cloud_supervisor_tasks,
    _cloud_training_active,
    _normalize_cloud_failure_payload,
    _repair_start_request,
    _set_cloud_supervisor_state,
    _training_intervention_start_request,
    _training_time_intervention,
)
from .train_cloud_helpers import (
    _cloud_failure_fingerprint,
    _cloud_supervisor_max_repairs,
    _runtime_binding_failure_message,
)
from .train_cloud_schema import CloudTrainStartRequest

_log = logging.getLogger(__name__)


def _watch_root_job_id(job_id: str, payload: dict[str, Any]) -> str:
    supervisor = payload.get("supervisor") if isinstance(payload.get("supervisor"), dict) else {}
    runtime = supervisor.get("runtime") if isinstance(supervisor.get("runtime"), dict) else {}
    root_job_id = str(runtime.get("rootJobId") or "").strip()
    if root_job_id:
        return root_job_id
    for marker in ("-intervention-", "-repair-", "-restart-"):
        if marker in job_id:
            return job_id.rsplit(marker, 1)[0]
    return job_id


async def run_cloud_supervisor_watch(
    *,
    root_job_id: str,
    username: str,
    automation_policy: dict[str, Any],
    initial_payload: dict[str, Any],
    training: TrainingService,
    start_cloud_training: Callable[[CloudTrainStartRequest], Awaitable[dict[str, Any]]],
    deployment_mode_for_payload: Callable[[dict[str, Any]], Awaitable[str]],
    llm_provider: Any = None,
) -> None:
    policy = _auto_repair_policy(automation_policy)
    if not policy["autoRetrySameRuntime"]:
        return
    provider = str(initial_payload.get("provider") or "")
    current_job_id = root_job_id
    runtime_seed = _cloud_supervisor_runtime_state(
        username,
        {
            "job_id": root_job_id,
            "supervisor": {
                "runtime": {
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                }
            },
        },
    )
    seeded_current_job_id = str(runtime_seed.get("currentJobId") or "").strip()
    if seeded_current_job_id:
        current_job_id = seeded_current_job_id
    try:
        repair_count = max(
            0,
            int(runtime_seed.get("repairAttempts", runtime_seed.get("repairCount", 0)) or 0),
        )
    except (TypeError, ValueError):
        repair_count = 0
    seen_failure_fingerprints: set[str] = set()
    seed_interventions = runtime_seed.get("appliedInterventions")
    seen_training_interventions: set[str] = (
        {str(item) for item in seed_interventions if str(item).strip()}
        if isinstance(seed_interventions, list)
        else set()
    )
    max_repairs = _cloud_supervisor_max_repairs()
    interval = max(1.0, float(os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_INTERVAL_SECONDS", "6") or "6"))
    initial_delay = max(0.0, float(os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_INITIAL_DELAY_SECONDS", "4") or "4"))
    _set_cloud_supervisor_state(username, root_job_id, {
        "state": "watching",
        "rootJobId": root_job_id,
        "currentJobId": current_job_id,
        "repairCount": repair_count,
        "repairAttempts": repair_count,
        "maxRepairs": max_repairs,
        "appliedInterventions": sorted(seen_training_interventions),
        "message": "后端总控正在观察云端任务。",
    })
    if initial_delay:
        await asyncio.sleep(initial_delay)
    while True:
        try:
            result = await training.status(job_id=current_job_id, username=username)
            payload = _normalize_cloud_failure_payload(result.to_dict())
            payload = _release_failed_cloud_hold(
                payload,
                username,
                reason="release hold before autonomous supervisor retry",
            )
            deployment_mode = await deployment_mode_for_payload({
                **payload,
                "provider": payload.get("provider") or provider,
            })
            if _cloud_training_active(payload):
                intervention = _training_time_intervention(payload)
                intervention_key = "|".join(
                    part
                    for part in (
                        current_job_id,
                        str(intervention.get("code") or ""),
                        str(intervention.get("strategy") or ""),
                    )
                    if part
                )
                if intervention and intervention_key not in seen_training_interventions:
                    seen_training_interventions.add(intervention_key)
                    intervention_payload = _training_intervention_start_request(
                        payload,
                        username,
                        automation_policy,
                        intervention,
                        deployment_mode=deployment_mode,
                    )
                    if intervention_payload is not None:
                        _set_cloud_supervisor_state(username, root_job_id, {
                            "state": "repairing",
                            "rootJobId": root_job_id,
                            "currentJobId": current_job_id,
                            "repairCount": repair_count,
                            "repairAttempts": repair_count,
                            "maxRepairs": max_repairs,
                            "appliedInterventions": sorted(seen_training_interventions),
                            "interventionCode": intervention.get("code") or "",
                            "trainingIntervention": intervention,
                            "message": "后端总控检测到训练异常，正在调整参数并续跑。",
                        })
                        _log.warning(
                            "Training intervention triggered for job %s: %s",
                            current_job_id,
                            intervention.get("summary") or intervention.get("code") or "",
                        )
                        await training.stop(TrainingStopSpec(job_id=current_job_id, username=username))
                        started = await start_cloud_training(CloudTrainStartRequest(**intervention_payload))
                        next_job_id = str(started.get("job_id") or intervention_payload.get("task_name") or "").strip()
                        if not next_job_id:
                            _set_cloud_supervisor_state(username, root_job_id, {
                                "state": "needs_review",
                                "rootJobId": root_job_id,
                                "currentJobId": current_job_id,
                                "repairCount": repair_count,
                                "repairAttempts": repair_count,
                                "maxRepairs": max_repairs,
                                "appliedInterventions": sorted(seen_training_interventions),
                                "interventionCode": intervention.get("code") or "",
                                "trainingIntervention": intervention,
                                "message": "训练异常干预请求没有返回任务 ID，需要人工确认。",
                            })
                            return
                        current_job_id = next_job_id
                        _set_cloud_supervisor_state(username, root_job_id, {
                            "state": "repairing",
                            "rootJobId": root_job_id,
                            "currentJobId": current_job_id,
                            "repairOfJobId": str(payload.get("job_id") or ""),
                            "repairCount": repair_count,
                            "repairAttempts": repair_count,
                            "maxRepairs": max_repairs,
                            "appliedInterventions": sorted(seen_training_interventions),
                            "interventionCode": intervention.get("code") or "",
                            "trainingIntervention": intervention,
                            "message": "训练异常已自动处理，后端总控继续观察。",
                        })
                        await asyncio.sleep(interval)
                        continue
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "watching",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "message": "后端总控正在观察云端任务。",
                })
                await asyncio.sleep(interval)
                continue
            if not _cloud_failure_signal(payload):
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "completed",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "message": "任务已结束，后端总控停止观察。",
                })
                return
            runtime_binding_message = _runtime_binding_failure_message(payload)
            if runtime_binding_message:
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "needs_rebind",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "failureRemediation": payload.get("failureRemediation") or {},
                    "message": runtime_binding_message,
                })
                return
            failure_fingerprint = _cloud_failure_fingerprint(payload)
            if failure_fingerprint in seen_failure_fingerprints:
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "needs_review",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "failureRemediation": payload.get("failureRemediation") or {},
                    "failureFingerprint": failure_fingerprint,
                    "message": "同一错误已经自动修复过，已暂停连续重试，避免反复生成新的 repair 任务。",
                })
                return
            seen_failure_fingerprints.add(failure_fingerprint)
            if max_repairs >= 0 and repair_count >= max_repairs:
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "needs_review",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "failureRemediation": payload.get("failureRemediation") or {},
                    "message": "自动修复已暂停，避免在同一错误上无限重试。请确认下一步处理方式。",
                })
                return
            repair_payload = await _repair_start_request(
                payload,
                username,
                training,
                automation_policy,
                deployment_mode=deployment_mode,
                llm_provider=llm_provider,
            )
            if repair_payload is None:
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "needs_review",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "status": payload.get("status") or "",
                    "failureRemediation": payload.get("failureRemediation") or {},
                    "message": "这次修复涉及未确认风险，已停下等待确认。",
                })
                return
            repair_count += 1
            repair_request = CloudTrainStartRequest(**repair_payload)
            _set_cloud_supervisor_state(username, root_job_id, {
                "state": "repairing",
                "rootJobId": root_job_id,
                "currentJobId": current_job_id,
                "repairCount": repair_count,
                "repairAttempts": repair_count,
                "maxRepairs": max_repairs,
                "appliedInterventions": sorted(seen_training_interventions),
                "failureRemediation": payload.get("failureRemediation") or {},
                "message": "后端总控正在同一实例内自动修复并续跑。",
            })
            started = await start_cloud_training(repair_request)
            next_job_id = str(started.get("job_id") or repair_request.task_name or "").strip()
            if not next_job_id:
                _set_cloud_supervisor_state(username, root_job_id, {
                    "state": "needs_review",
                    "rootJobId": root_job_id,
                    "currentJobId": current_job_id,
                    "repairCount": repair_count,
                    "repairAttempts": repair_count,
                    "maxRepairs": max_repairs,
                    "appliedInterventions": sorted(seen_training_interventions),
                    "message": "续跑请求没有返回任务 ID，需要人工确认。",
                })
                return
            current_job_id = next_job_id
            _log.info(
                "Auto-repair submitted for job %s -> %s (strategy=%s)",
                payload.get("job_id") or "",
                next_job_id,
                repair_request.params.get("repairStrategy") if isinstance(repair_request.params, dict) else "",
            )
            _set_cloud_supervisor_state(username, root_job_id, {
                "state": "repair_submitted",
                "rootJobId": root_job_id,
                "currentJobId": current_job_id,
                "repairOfJobId": str(payload.get("job_id") or ""),
                "repairCount": repair_count,
                "repairAttempts": repair_count,
                "maxRepairs": max_repairs,
                "appliedInterventions": sorted(seen_training_interventions),
                "message": "修复任务已提交，后端总控继续观察。",
            })
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            _set_cloud_supervisor_state(username, root_job_id, {
                "state": "cancelled",
                "rootJobId": root_job_id,
                "currentJobId": current_job_id,
                "repairCount": repair_count,
                "repairAttempts": repair_count,
                "maxRepairs": max_repairs,
                "appliedInterventions": sorted(seen_training_interventions),
                "message": "后端总控已停止观察。",
            })
            raise
        except Exception as exc:  # noqa: BLE001 - keep supervisor failures contained
            _set_cloud_supervisor_state(username, root_job_id, {
                "state": "needs_review",
                "rootJobId": root_job_id,
                "currentJobId": current_job_id,
                "repairCount": repair_count,
                "repairAttempts": repair_count,
                "maxRepairs": max_repairs,
                "appliedInterventions": sorted(seen_training_interventions),
                "message": f"后端总控观察失败：{exc}",
            })
            return


def schedule_cloud_supervisor(
    *,
    username: str,
    payload: dict[str, Any],
    automation_policy: dict[str, Any],
    training: TrainingService,
    start_cloud_training: Callable[[CloudTrainStartRequest], Awaitable[dict[str, Any]]],
    deployment_mode_for_payload: Callable[[dict[str, Any]], Awaitable[str]],
    llm_provider: Any = None,
) -> None:
    policy = _auto_repair_policy(automation_policy)
    if not username.strip() or not policy["autoRetrySameRuntime"]:
        return
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return
    root_job_id = _watch_root_job_id(job_id, payload)
    key = _cloud_supervisor_task_key(username, root_job_id)
    with _cloud_supervisor_runtime_lock:
        existing = _cloud_supervisor_tasks.get(key)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            run_cloud_supervisor_watch(
                root_job_id=root_job_id,
                username=username,
                automation_policy=automation_policy,
                initial_payload=dict(payload),
                training=training,
                start_cloud_training=start_cloud_training,
                deployment_mode_for_payload=deployment_mode_for_payload,
                llm_provider=llm_provider,
            )
        )
        _cloud_supervisor_tasks[key] = task
