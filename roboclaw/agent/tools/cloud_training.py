"""Agent-facing Evo Studio cloud training tool."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.account import (
    AccountLedger,
    estimate_training_hold_cents,
    hourly_cost_from_params,
)
from roboclaw.agent.tools.base import Tool
from roboclaw.data.auth_refs import validate_training_auth_refs
from roboclaw.training import TrainingPlanSpec, TrainingStartSpec, TrainingStopSpec
from roboclaw.training.service import TrainingService


class EvoStudioCloudTrainTool(Tool):
    """Plan, preflight, and start EVO_Train cloud jobs from the agent."""

    def __init__(
        self,
        *,
        embodied_service: Any = None,
        ledger: AccountLedger | None = None,
    ) -> None:
        self.embodied_service = embodied_service
        self._ledger = ledger

    @property
    def name(self) -> str:
        return "evo_studio_cloud_train"

    @property
    def description(self) -> str:
        return (
            "Use Evo Studio cloud training via EVO_Train. Supports planning, runtime matching, "
            "source preflight, wallet checks, status, stop, and confirmed cloud job start. "
            "Use this instead of the legacy local train tool for VLA/RL cloud workflows. "
            "Starting a paid job requires either confirmed=true or automation_mode=full_auto "
            "under the configured automation policy after the plan, runtime, estimated hold, "
            "and artifact paths are available."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "balance",
                        "backend_probe",
                        "repair_backend",
                        "provider_balance",
                        "plan",
                        "runtime_match",
                        "source_preflight",
                        "start",
                        "current",
                        "status",
                        "stop",
                    ],
                },
                "username": {"type": "string"},
                "message": {"type": "string"},
                "provider": {"type": "string"},
                "workflow": {"type": "string"},
                "params": {"type": "object"},
                "source": {"type": "object"},
                "role": {"type": "string", "enum": ["dataset", "model", "artifact"]},
                "sku_id": {"type": "string"},
                "image_id": {"type": "string"},
                "task_name": {"type": "string"},
                "job_id": {"type": "string"},
                "dataset_name": {"type": "string"},
                "policy_type": {"type": "string"},
                "steps": {"type": "integer", "minimum": 1},
                "device": {"type": "string"},
                "hourly_cost_cents": {"type": "integer", "minimum": 0},
                "service_fee_bps": {"type": "integer", "minimum": 0},
                "confirmed": {"type": "boolean"},
                "automation_mode": {"type": "string", "enum": ["ask", "safe_auto", "full_auto"]},
                "automationMode": {"type": "string", "enum": ["ask", "safe_auto", "full_auto"]},
                "require_billing": {"type": "boolean"},
                "minimum_assets": {"type": "integer", "minimum": 0},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "").strip()
        username = str(kwargs.get("username") or "").strip()
        if action == "balance":
            return self._wallet_response(username)

        training = self._training_service()
        if action == "backend_probe":
            return await self._backend_probe(training, kwargs)
        if action == "repair_backend":
            return await self._repair_backend(training, kwargs)
        if action == "provider_balance":
            return await self._provider_balance(training, kwargs)
        if action == "plan":
            return await self._plan(training, kwargs)
        if action == "runtime_match":
            return await self._runtime_match(training, kwargs)
        if action == "source_preflight":
            return await self._source_preflight(training, kwargs)
        if action == "start":
            return await self._start(training, kwargs)
        if action == "current":
            return _json((await training.current(username=username)).to_dict())
        if action == "status":
            job_id = str(kwargs.get("job_id") or "").strip()
            if not job_id:
                return "Error: job_id is required for status."
            return _json((await training.status(job_id=job_id, username=username)).to_dict())
        if action == "stop":
            job_id = str(kwargs.get("job_id") or "").strip()
            if not job_id:
                return "Error: job_id is required for stop."
            return _json((await training.stop(TrainingStopSpec(job_id=job_id, username=username))).to_dict())
        return f"Error: unknown cloud training action: {action}"

    def _training_service(self) -> TrainingService:
        if self.embodied_service is not None:
            return TrainingService(self.embodied_service)
        from roboclaw.embodied.service import EmbodiedService

        return TrainingService(EmbodiedService())

    def _ledger_service(self) -> AccountLedger:
        if self._ledger is None:
            self._ledger = AccountLedger()
        return self._ledger

    def _wallet_response(self, username: str) -> str:
        if not username:
            return "Error: username is required for balance."
        return _json({"wallet": self._ledger_service().wallet(username).to_dict()})

    async def _backend_probe(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        return _json(await self._collect_backend_probe(training, kwargs))

    async def _collect_backend_probe(self, training: TrainingService, kwargs: dict[str, Any]) -> dict[str, Any]:
        provider = str(kwargs.get("provider") or "")
        username = str(kwargs.get("username") or "")
        payload: dict[str, Any] = {
            "bridge": training.cloud_bridge_status(),
            "probeScope": "evo_train_cloud_backend",
            "note": (
                "This probes the EVO_Train cloud/SSH backend contract. It does not run "
                "local shell commands on the RoboClaw host."
            ),
        }
        checks: dict[str, Any] = {}
        for key, call in (
            ("configuration", lambda: training.configuration_check(provider=provider)),
            (
                "gpuSkus",
                lambda: training.gpu_skus(provider=provider, include_incomplete=True, force_refresh=True),
            ),
            ("images", lambda: training.images(provider=provider, include_incomplete=True)),
        ):
            try:
                checks[key] = await call()
            except Exception as exc:
                checks[key] = {"ok": False, "error": str(exc)}
        if username:
            try:
                checks["current"] = (await training.current(username=username)).to_dict()
            except Exception as exc:
                checks["current"] = {"ok": False, "error": str(exc)}
        payload["checks"] = checks
        return payload

    async def _repair_backend(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        """Return allowlisted backend repair actions the agent can explain or apply through ops.

        This deliberately does not execute arbitrary shell commands. Secrets and paid starts stay
        outside the repair path; it only diagnoses known deployment problems and gives a concrete
        operator action.
        """
        probe = await self._collect_backend_probe(training, kwargs)
        configuration = dict((probe.get("checks") or {}).get("configuration") or {})
        missing = {str(item) for item in configuration.get("missing") or []}
        warnings = [str(item) for item in configuration.get("warnings") or []]
        warning_text = "\n".join(warnings).lower()
        repairs: list[dict[str, Any]] = []

        if "AUTODL_SSH_PARAMIKO" in missing or "paramiko" in warning_text:
            command = _paramiko_install_command(configuration)
            repairs.append(
                {
                    "id": "autodl_ssh_paramiko",
                    "title": "Install AutoDL SSH dependency in EVO_Train runtime",
                    "reason": (
                        "EVO_Train uses SSH to run jobs on an existing AutoDL/SeetaCloud instance. "
                        "The Python process running EVO_Train must have paramiko installed before start."
                    ),
                    "safeToAutoApply": False,
                    "commands": [
                        command,
                        "restart EVO_Train with the same Python interpreter after installation",
                    ],
                    "userFacingSummary": "后端 SSH 依赖缺失：需要在运行 EVO_Train 的 Python 环境安装 paramiko 后重启。",
                }
            )

        if "AUTODL_SSH_CONNECTION" in missing:
            repairs.append(
                {
                    "id": "autodl_ssh_connection",
                    "title": "Configure managed SSH connection for existing GPU instance",
                    "reason": "Existing-instance mode needs host, port, user, and password/key in backend deployment env.",
                    "safeToAutoApply": False,
                    "commands": [
                        "set AUTODL_HOST/AUTODL_PORT/AUTODL_USER and AUTODL_PASSWORD or AUTODL_KEY_PATH in the backend deployment",
                        "restart EVO_Train after updating the deployment env",
                    ],
                    "userFacingSummary": "后端还没有配置 SSH 实例连接；这些应由部署环境保存，不能让用户在聊天里粘贴。",
                }
            )

        status = "repair_available" if repairs else "no_known_repair_needed"
        return _json(
            {
                "action": "repair_backend",
                "status": status,
                "repairScope": "allowlisted_backend_repair",
                "autoApplied": False,
                "repairs": repairs,
                "probe": probe,
            }
        )

    async def _plan(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        username = str(kwargs.get("username") or "")
        workflow, params = _normalize_workflow_and_params(
            str(kwargs.get("workflow") or ""),
            dict(kwargs.get("params") or {}),
        )
        result = await training.plan(
            TrainingPlanSpec(
                username=username,
                message=str(kwargs.get("message") or ""),
                workflow=workflow,
                params=params,
                provider=str(kwargs.get("provider") or ""),
                sku_id=str(kwargs.get("sku_id") or ""),
                image_id=str(kwargs.get("image_id") or ""),
            )
        )
        return _json(_attach_user_wallet(result, username, self._ledger_service()))

    async def _provider_balance(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        result = await training.provider_balance(
            provider=str(kwargs.get("provider") or ""),
            minimum_assets=int(kwargs.get("minimum_assets") or kwargs.get("minimumAssets") or 0),
        )
        payload = dict(result)
        payload["balanceScope"] = "provider_pool"
        payload["description"] = (
            "Operator AutoDL/provider balance for managed compute capacity. "
            "Use action=balance for the user's Evo Studio training balance."
        )
        return _json(payload)

    async def _runtime_match(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        _workflow, params = _normalize_workflow_and_params(
            str(kwargs.get("workflow") or ""),
            dict(kwargs.get("params") or {}),
        )
        result = await training.runtime_match(
            username=str(kwargs.get("username") or ""),
            provider=str(kwargs.get("provider") or ""),
            params=params,
            sku_id=str(kwargs.get("sku_id") or ""),
            image_id=str(kwargs.get("image_id") or ""),
        )
        return _json(result)

    async def _source_preflight(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        result = await training.source_preflight(
            username=str(kwargs.get("username") or ""),
            provider=str(kwargs.get("provider") or ""),
            role=str(kwargs.get("role") or "dataset"),
            source=dict(kwargs.get("source") or {}),
        )
        return _json(result)

    async def _start(self, training: TrainingService, kwargs: dict[str, Any]) -> str:
        automation_mode = _automation_mode(kwargs)
        confirmed = bool(kwargs.get("confirmed", False)) or automation_mode == "full_auto"
        if not confirmed:
            return (
                "Error: confirmed=true or automation_mode=full_auto is required before "
                "starting a paid cloud training job. First show the user the plan, runtime "
                "match, source preflight, estimated hold, and artifact path, then ask for "
                "confirmation unless the configured automation policy allows full_auto."
            )
        if not training.cloud_enabled:
            return "Error: EVO_Train bridge is not enabled in this backend deployment."

        username = str(kwargs.get("username") or "").strip()
        workflow, params = _normalize_workflow_and_params(
            str(kwargs.get("workflow") or ""),
            dict(kwargs.get("params") or {}),
        )
        if automation_mode:
            automation_policy = dict(params.get("automationPolicy") or {})
            automation_policy.setdefault("mode", automation_mode)
            params["automationPolicy"] = automation_policy
        params = _normalize_cloud_params(params, dataset_name=str(kwargs.get("dataset_name") or ""))
        auth_errors = validate_training_auth_refs(params, username=username)
        if auth_errors:
            return _json({"error": "training_auth_ref_invalid", "errors": auth_errors})

        service_fee_bps = int(kwargs.get("service_fee_bps") or 1000)
        hourly_cost_cents = int(kwargs.get("hourly_cost_cents") or 0) or hourly_cost_from_params(params)
        require_billing = kwargs.get("require_billing", True)
        if require_billing and not username:
            return "Error: username is required so Evo Studio can freeze the user's training balance."
        if require_billing and hourly_cost_cents <= 0:
            return (
                "Error: hourly_cost_cents is required before starting so Evo Studio can reserve "
                "at least the first hour of cloud training cost."
            )

        hold_cents = 0
        freeze_record = None
        task_name = str(kwargs.get("task_name") or "")
        if username and hourly_cost_cents:
            hold_cents = estimate_training_hold_cents(
                hourly_cost_cents=hourly_cost_cents,
                service_fee_bps=service_fee_bps,
            )
            try:
                _wallet, freeze_record = self._ledger_service().freeze(
                    username,
                    hold_cents,
                    reason="cloud training first-hour hold",
                    task_name=task_name or str(kwargs.get("dataset_name") or ""),
                    job_id=task_name or str(kwargs.get("dataset_name") or "") or "pending-cloud-train",
                )
            except ValueError as exc:
                return f"Error: {exc}"

        try:
            status = await training.start(
                TrainingStartSpec(
                    dataset_name=str(kwargs.get("dataset_name") or ""),
                    policy_type=str(kwargs.get("policy_type") or "act"),
                    steps=int(kwargs.get("steps") or 100_000),
                    device=str(kwargs.get("device") or "cuda"),
                    username=username,
                    provider=str(kwargs.get("provider") or ""),
                    workflow=workflow,
                    params=params,
                    sku_id=str(kwargs.get("sku_id") or ""),
                    image_id=str(kwargs.get("image_id") or ""),
                    task_name=task_name,
                    wait_for_submit=str(
                        kwargs.get("wait_for_submit") if "wait_for_submit" in kwargs else kwargs.get("waitForSubmit", True)
                    ).strip().lower() not in {"0", "false", "no", "off"},
                )
            )
        except RuntimeError as exc:
            if username and freeze_record is not None:
                try:
                    self._ledger_service().release_job_hold(
                        username,
                        freeze_record.job_id,
                        reason="release hold after cloud training start failure",
                        task_name=task_name,
                    )
                except ValueError:
                    pass
            return f"Error: {exc}"

        payload = status.to_dict()
        if username and freeze_record is not None:
            job_id = payload.get("job_id") or freeze_record.job_id
            if job_id and str(job_id) != freeze_record.job_id:
                try:
                    freeze_record = self._ledger_service().reassign_job_hold(
                        username,
                        freeze_record.job_id,
                        str(job_id),
                    )
                except ValueError:
                    pass
            payload["billing"] = {
                "holdCents": hold_cents,
                "hourlyCostCents": hourly_cost_cents,
                "serviceFeeBps": service_fee_bps,
                "record": freeze_record.to_dict(),
            }
            payload["wallet"] = self._ledger_service().wallet(username).to_dict()
        return _json(payload)


def _normalize_cloud_params(params: dict[str, Any], *, dataset_name: str = "") -> dict[str, Any]:
    from roboclaw.http.routes.train_cloud import _normalize_cloud_training_params

    return _normalize_cloud_training_params(params, dataset_name=dataset_name)


def _normalize_workflow_and_params(workflow: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    workflow = workflow.strip()
    normalized = dict(params or {})
    if _looks_like_vla_backend_intent(workflow, normalized):
        workflow = "vla_rl_backend"
    return workflow, normalized


def _looks_like_vla_backend_intent(workflow: str, params: dict[str, Any]) -> bool:
    if workflow in {"", "rlinf_vla", "vla_rl_backend"}:
        return workflow == "vla_rl_backend"
    if any(params.get(key) not in (None, "", {}, []) for key in ("backendKind", "backendInterface", "builtinTrainingProfile")):
        return True
    if any(params.get(key) not in (None, "", {}, []) for key in ("datasetSource", "modelSource", "codeSource")):
        return True
    if any(params.get(key) not in (None, "", {}, []) for key in ("policyType", "modelFamily", "policyFamily", "checkpointPath")):
        return True
    text = " ".join(
        str(item)
        for item in (
            workflow,
            params.get("recipe"),
            params.get("trainingMode"),
        )
        if item not in (None, "")
    ).lower()
    return any(token in text for token in ("vla", "lerobot", "rlinf", "openpi", "robomimic", "isaaclab"))


def _automation_mode(kwargs: dict[str, Any]) -> str:
    explicit = str(kwargs.get("automation_mode") or kwargs.get("automationMode") or "").strip().lower()
    if explicit:
        return explicit if explicit in {"ask", "safe_auto", "full_auto"} else "ask"
    params = kwargs.get("params")
    if isinstance(params, dict):
        policy = params.get("automationPolicy") or params.get("automation_policy")
        if isinstance(policy, dict):
            mode = str(policy.get("mode") or "").strip().lower()
            if mode in {"ask", "safe_auto", "full_auto"}:
                return mode
    return "ask"


def _source_uri(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("uri", "url", "repo", "repoId", "repoUrl", "gitUrl", "path"):
            item = value.get(key)
            if item not in (None, ""):
                return str(item)
        return ""
    return str(value or "")


def _paramiko_install_command(configuration: dict[str, Any]) -> str:
    for check in configuration.get("checks") or []:
        if check.get("name") != "AUTODL_SSH_PARAMIKO":
            continue
        detail = str(check.get("detail") or "")
        marker = "interpreter="
        if marker in detail:
            interpreter = detail.split(marker, 1)[1].strip()
            if interpreter:
                return f"{interpreter} -m pip install paramiko"
    return "python3 -m pip install paramiko"


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _attach_user_wallet(payload: dict[str, Any], username: str, ledger: AccountLedger) -> dict[str, Any]:
    result = dict(payload)
    username = username.strip()
    executor_wallet = result.get("wallet")
    if executor_wallet is not None:
        result["executorWallet"] = executor_wallet
    if username:
        result["wallet"] = ledger.wallet(username).to_dict()
        result["billingMode"] = "external"
    return result
