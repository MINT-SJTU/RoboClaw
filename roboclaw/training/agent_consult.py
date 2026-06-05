"""Evo Studio agent-consult orchestration.

This is the product-facing control layer: a thin, deterministic backend agent
that turns a natural-language training request into the cloud-control actions
RoboClaw/EVO_Train can actually execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from roboclaw.providers.base import LLMProvider
from roboclaw.training.embodied_intent import classify_embodied_intent
from roboclaw.training.schema import TrainingStartSpec
from roboclaw.training.service import TrainingService
from roboclaw.training.vla_rl import VLAPlanRequest, VLARLService


@dataclass(frozen=True)
class EvoStudioAgentConsultRequest:
    """A delegated Evo Studio product task."""

    task: str
    mode: str = "plan"
    username: str = ""
    provider: str = ""
    workflow: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""
    job_id: str = ""
    confirmed: bool = False
    automation_policy: Mapping[str, Any] = field(default_factory=dict)
    automation_mode: str = ""


class EvoStudioAgentConsultService:
    """OpenClaw-style inner agent for Evo Studio training workflows.

    The outer LLM or OpenAI-compatible relay should call only this consult
    surface. The service then performs real backend checks through
    ``TrainingService`` and returns a structured, confirmable result.
    """

    def __init__(
        self,
        training: TrainingService,
        *,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._training = training
        self._llm_provider = llm_provider

    async def consult(self, request: EvoStudioAgentConsultRequest) -> dict[str, Any]:
        mode = _normalize_mode(request.mode)
        context = dict(request.context or {})
        params = _merge_context_params(context, request.params)
        username = _first_string(request.username, context.get("username"))
        provider = _first_string(request.provider, context.get("provider"))
        workflow = _first_string(request.workflow, context.get("workflow"))
        sku_id = _first_string(request.sku_id, context.get("sku_id"), context.get("skuId"))
        image_id = _first_string(request.image_id, context.get("image_id"), context.get("imageId"))
        job_id = _first_string(request.job_id, context.get("job_id"), context.get("jobId"))
        automation_policy = _automation_policy(request.automation_policy, context, params)
        automation_mode = _first_string(request.automation_mode, automation_policy.get("mode")).lower()
        confirmed = request.confirmed or automation_mode == "full_auto"
        embodied_intent = _embodied_intent(request.task, context)

        actions: list[str] = []
        response: dict[str, Any] = {
            "kind": "evo_studio_agent_consult/v1",
            "consultTool": "evo_studio_agent_consult",
            "mode": mode,
            "task": request.task,
            "delegated": True,
            "outerModelContract": {
                "description": "Outer providers delegate to this one stable backend contract; skills/tools run inside RoboClaw/EVO_Train.",
                "directToolSurface": False,
            },
            "executionPolicy": {
                **automation_policy,
                "confirmedStartRequired": automation_policy["paidStartRequiresConfirmation"],
                "startsPaidCompute": mode == "execute" and confirmed,
                "fullAuto": automation_mode == "full_auto",
                "secretsInChatAllowed": False,
            },
            "actions": actions,
            "completedChecks": actions,
        }
        response["embodiedIntent"] = embodied_intent

        bridge = self._training.cloud_bridge_status()
        response["bridge"] = bridge
        actions.append("bridge_status")

        if embodied_intent.get("route") not in {"", "none", "cloud_training"}:
            response.update(_non_cloud_training_response(embodied_intent))
            response["completedChecks"] = actions
            return response

        cloud_provider = str(bridge.get("provider") or "").strip()
        effective_provider = _effective_cloud_provider(provider, cloud_provider)
        configuration = await self._configuration_check(provider=effective_provider)
        if configuration:
            response["configuration"] = configuration
            actions.append("configuration_check")
        ssh_existing_runtime = _is_ssh_existing_runtime(
            provider=provider,
            context=context,
            configuration=configuration,
        )
        if ssh_existing_runtime:
            response["runtimeMode"] = "ssh_existing_instance"
            response["provider"] = effective_provider

        if mode in {"status", "repair"}:
            response["status"] = await self._status(job_id=job_id, username=username)
            actions.append("status")
            if mode == "repair":
                response["repair"] = _repair_plan_from_status(response["status"], automation_policy)
            if mode == "status":
                response["nextAction"] = "inspect_logs_or_repair" if _status_failed(response["status"]) else "wait_or_stop"
                response["completedChecks"] = actions
                return response

        plan_response = await self._plan(
            task=request.task,
            username=username,
            workflow=workflow,
            params=params,
            provider=effective_provider,
            sku_id=sku_id,
            image_id=image_id,
        )
        response.update({
            "plan": plan_response.get("plan"),
            "vlaPlan": plan_response.get("vlaPlan"),
            "aiPlan": plan_response.get("aiPlan"),
            "plannerMessage": plan_response.get("message", ""),
            "wallet": plan_response.get("wallet"),
            "executorWallet": plan_response.get("executorWallet"),
            "billingMode": plan_response.get("billingMode"),
        })
        actions.append("plan")

        plan = dict(plan_response.get("vlaPlan") or plan_response.get("plan") or {})
        planned_params = dict(plan.get("params") or params)
        source_checks = await self._source_preflights(
            username=username,
            provider=effective_provider,
            params=planned_params,
        )
        if source_checks:
            response["sourcePreflight"] = source_checks
            actions.append("source_preflight")

        runtime_match = await self._runtime_match(
            username=username,
            provider=effective_provider,
            params=planned_params,
            sku_id=sku_id,
            image_id=image_id,
            ssh_existing_runtime=ssh_existing_runtime,
            configuration=configuration,
        )
        response["runtimeMatch"] = runtime_match
        actions.append("runtime_match")

        ready_for_confirmation = bool(plan.get("readyToStart")) and bool(runtime_match.get("readyToStart"))
        response["readyForConfirmation"] = ready_for_confirmation
        if mode == "repair":
            response["nextAction"] = "confirm_repaired_start" if ready_for_confirmation else "adjust_repair_plan_or_runtime"
        else:
            response["nextAction"] = (
                "confirm_start"
                if ready_for_confirmation
                else "adjust_plan_or_runtime"
            )

        if mode == "execute":
            if confirmed and ready_for_confirmation:
                try:
                    started = await self._start_from_plan(
                        plan=plan,
                        params=planned_params,
                        username=username,
                        provider=effective_provider,
                        workflow=workflow,
                        sku_id=sku_id,
                        image_id=image_id,
                        automation_policy=automation_policy,
                    )
                    response["start"] = {"started": True, **started}
                    response["started"] = True
                    response["nextAction"] = "watch_status"
                    actions.append("start")
                except RuntimeError as exc:
                    response["start"] = {"started": False, "error": str(exc)}
                    response["started"] = False
                    response["nextAction"] = "start_failed"
            else:
                response["start"] = {
                    "started": False,
                    "reason": (
                        "confirmed=true or automationMode=full_auto is required before starting paid compute."
                        if not confirmed
                        else "plan/runtime checks are not ready for start."
                    ),
                }
                response["started"] = False
        response["completedChecks"] = actions
        return response

    async def _start_from_plan(
        self,
        *,
        plan: dict[str, Any],
        params: dict[str, Any],
        username: str,
        provider: str,
        workflow: str,
        sku_id: str,
        image_id: str,
        automation_policy: dict[str, Any],
    ) -> dict[str, Any]:
        policy_type = _first_string(
            params.get("policyType"),
            params.get("policyFamily"),
            params.get("modelFamily"),
            params.get("modelRegistryName"),
            "act",
        )
        steps_raw = params.get("steps") or params.get("maxSteps") or 100_000
        try:
            steps = int(steps_raw)
        except (TypeError, ValueError):
            steps = 100_000
        start_params = dict(params)
        start_params.setdefault("automationPolicy", automation_policy)
        result = await self._training.start(
            TrainingStartSpec(
                dataset_name=_first_string(params.get("datasetName"), params.get("dataset_name")),
                policy_type=policy_type,
                steps=steps,
                device=_first_string(params.get("device"), "cuda"),
                username=username,
                provider=provider,
                workflow=_first_string(plan.get("workflow"), workflow),
                params=start_params,
                sku_id=_first_string(plan.get("sku_id"), plan.get("skuId"), sku_id),
                image_id=_first_string(plan.get("image_id"), plan.get("imageId"), image_id),
                task_name=_first_string(params.get("taskName"), params.get("task_name")),
                wait_for_submit=True,
            )
        )
        return result.to_dict()

    async def _plan(
        self,
        *,
        task: str,
        username: str,
        workflow: str,
        params: dict[str, Any],
        provider: str,
        sku_id: str,
        image_id: str,
    ) -> dict[str, Any]:
        vla_rl = VLARLService(self._training, llm_provider=self._llm_provider)
        return await vla_rl.plan(
            VLAPlanRequest(
                username=username,
                message=task,
                workflow=workflow,
                params=params,
                provider=provider,
                sku_id=sku_id,
                image_id=image_id,
            )
        )

    async def _runtime_match(
        self,
        *,
        username: str,
        provider: str,
        params: dict[str, Any],
        sku_id: str,
        image_id: str,
        ssh_existing_runtime: bool = False,
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._training.cloud_enabled:
            return {"readyToStart": False, "error": "EVO_Train bridge is not enabled."}
        if ssh_existing_runtime:
            ready = _configuration_ready(configuration)
            return {
                "readyToStart": ready,
                "skipped": True,
                "provider": provider,
                "runtimeMode": "ssh_existing_instance",
                "message": (
                    "Using the already configured SSH backend; GPU SKU/image matching is not required."
                    if ready
                    else "SSH backend is selected, but deployment configuration is not ready."
                ),
                "configuration": configuration or {},
            }
        try:
            return await self._training.runtime_match(
                username=username,
                provider=provider,
                params=params,
                sku_id=sku_id,
                image_id=image_id,
                force_refresh=True,
            )
        except RuntimeError as exc:
            return {"readyToStart": False, "error": str(exc)}

    async def _configuration_check(self, *, provider: str) -> dict[str, Any]:
        if not self._training.cloud_enabled:
            return {}
        try:
            return dict(await self._training.configuration_check(provider=provider))
        except RuntimeError as exc:
            return {"ready": False, "error": str(exc)}

    async def _source_preflights(
        self,
        *,
        username: str,
        provider: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._training.cloud_enabled:
            return {}
        checks: dict[str, Any] = {}
        for role, key in (("dataset", "datasetSource"), ("model", "modelSource")):
            source = params.get(key)
            if not isinstance(source, dict) or not source:
                continue
            try:
                checks[role] = await self._training.source_preflight(
                    username=username,
                    provider=provider,
                    role=role,
                    source=source,
                )
            except RuntimeError as exc:
                checks[role] = {"ok": False, "error": str(exc)}
        return checks

    async def _status(self, *, job_id: str, username: str) -> dict[str, Any]:
        try:
            if job_id:
                return (await self._training.status(job_id=job_id, username=username)).to_dict()
            return (await self._training.current(username=username)).to_dict()
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}


def _normalize_mode(mode: str) -> str:
    value = (mode or "").strip().lower()
    return value if value in {"plan", "execute", "repair", "status"} else "plan"


def _merge_context_params(context: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (context.get("params"), context.get("currentParams"), params):
        if isinstance(source, Mapping):
            merged.update(dict(source))
    for key in ("datasetSource", "modelSource", "trainingContract"):
        item = context.get(key)
        if isinstance(item, Mapping) and key not in merged:
            merged[key] = dict(item)
    return merged


def _embodied_intent(task: str, context: Mapping[str, Any]) -> dict[str, Any]:
    item = context.get("embodiedIntent")
    if isinstance(item, Mapping) and item:
        return dict(item)
    return classify_embodied_intent(task, context).to_dict()


def _non_cloud_training_response(intent: Mapping[str, Any]) -> dict[str, Any]:
    route = str(intent.get("route") or "")
    if route == "dataset_pipeline":
        next_action = "open_dataset_push_or_prepare_upload"
        missing = ["dataset upload runner is not wired to this consult executor yet"]
    elif route == "robotics_runtime":
        next_action = "connect_robot_runtime_or_select_executor"
        missing = ["robotics runtime executor is not wired to this consult executor yet"]
    else:
        next_action = "choose_supported_executor"
        missing = ["unsupported embodied route"]
    return {
        "capabilityRoute": route,
        "readyForConfirmation": False,
        "started": False,
        "start": {"started": False, "reason": "selected capability route is not launchable by cloud training"},
        "nextAction": next_action,
        "missingFields": missing,
        "plannerMessage": (
            "该需求已识别为具身能力任务，但当前 consult 执行器还没有把它接到可启动的运行器；"
            "不会误走 VLA 云训练。"
        ),
    }


def _automation_policy(
    request_policy: Mapping[str, Any],
    context: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    raw: Mapping[str, Any] = {}
    for candidate in (
        request_policy,
        context.get("automationPolicy"),
        context.get("automation_policy"),
        params.get("automationPolicy"),
        params.get("automation_policy"),
    ):
        if isinstance(candidate, Mapping) and candidate:
            raw = candidate
            break

    mode = str(raw.get("mode") or "").strip().lower()
    if mode not in {"ask", "safe_auto", "full_auto"}:
        mode = "ask"
    auto_enabled = mode in {"safe_auto", "full_auto"}
    full_auto = mode == "full_auto"
    return {
        "mode": mode,
        "autoInspectLogs": bool(raw.get("autoInspectLogs", auto_enabled)),
        "autoRepairPlan": bool(raw.get("autoRepairPlan", auto_enabled)),
        "autoRetrySameRuntime": bool(raw.get("autoRetrySameRuntime", full_auto)),
        "allowAgentRepairSameRuntime": bool(raw.get("allowAgentRepairSameRuntime", full_auto)),
        "paidStartRequiresConfirmation": bool(raw.get("paidStartRequiresConfirmation", not full_auto)),
        "allowRuntimeChangeWithoutConfirmation": bool(raw.get("allowRuntimeChangeWithoutConfirmation", False)),
        "allowSecretEditingInChat": False,
    }


def _first_string(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _effective_cloud_provider(requested_provider: str, bridge_provider: str) -> str:
    provider = (requested_provider or "").strip()
    if provider.lower() in {"ssh", "existing_ssh", "ssh_existing", "debug_ssh"}:
        return bridge_provider or ""
    return provider or bridge_provider or ""


def _is_ssh_existing_runtime(
    *,
    provider: str,
    context: Mapping[str, Any],
    configuration: Mapping[str, Any] | None,
) -> bool:
    provider_value = (provider or "").strip().lower()
    backend = str(context.get("backend") or context.get("runtimeMode") or "").strip().lower()
    mode = str((configuration or {}).get("mode") or (configuration or {}).get("deploymentMode") or "").strip().lower()
    return provider_value in {"ssh", "existing_ssh", "ssh_existing", "debug_ssh"} or backend in {
        "ssh",
        "existing_ssh",
        "ssh_existing",
        "debug_ssh",
    } or mode in {"ssh", "existing_ssh", "ssh_existing"}


def _configuration_ready(configuration: Mapping[str, Any] | None) -> bool:
    if not configuration:
        return True
    if "ready" in configuration:
        return bool(configuration.get("ready"))
    missing = configuration.get("missingFields") or configuration.get("missingDeploymentFields") or []
    if isinstance(missing, list) and missing:
        return False
    return not configuration.get("error")


def _status_failed(status: Mapping[str, Any]) -> bool:
    text = f"{status.get('status', '')} {status.get('error', '')}".lower()
    return any(token in text for token in ("failed", "error", "stopped"))


def _repair_plan_from_status(
    status: Mapping[str, Any],
    automation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    remediation = status.get("failureRemediation")
    if not isinstance(remediation, Mapping):
        remediation = _first_task_remediation(status)
    auto_repair = remediation.get("autoRepair") if isinstance(remediation, Mapping) else {}
    policy = dict(automation_policy or {})
    paid_start_requires_confirmation = bool(policy.get("paidStartRequiresConfirmation", True))
    return {
        "available": bool(remediation),
        "summary": (
            str(remediation.get("summary") or "")
            if isinstance(remediation, Mapping)
            else ""
        ),
        "autoRepair": dict(auto_repair) if isinstance(auto_repair, Mapping) else {},
        "preserveCaches": {
            "workdir": True,
            "sourceCache": True,
            "successfulStages": True,
        },
        "automationPolicy": policy,
        "autoRetrySameRuntimeAllowed": bool(policy.get("autoRetrySameRuntime", False)),
        "requiresUserConfirmationBeforeStart": paid_start_requires_confirmation,
        "userFacingSummary": (
            "已根据失败原因生成续跑计划；后端会复用源码、数据/权重缓存和已完成阶段，"
            + (
                "但重新启动云端任务前仍需要用户确认。"
                if paid_start_requires_confirmation
                else "当前权限允许在同一后端和预算内自动续跑。"
            )
        ),
    }


def _first_task_remediation(status: Mapping[str, Any]) -> Mapping[str, Any]:
    tasks = status.get("tasks")
    if not isinstance(tasks, list):
        return {}
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        remediation = item.get("failureRemediation")
        if isinstance(remediation, Mapping):
            return remediation
    return {}
