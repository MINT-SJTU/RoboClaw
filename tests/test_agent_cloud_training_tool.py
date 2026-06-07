from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from roboclaw.account import AccountLedger
from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.cloud_training import EvoStudioCloudTrainTool
from roboclaw.agent.tools.evo_studio_agent import EvoStudioAgentConsultTool
from roboclaw.bus.queue import MessageBus
from roboclaw.training import TrainingJobStatus
from roboclaw.training.embodied_intent import classify_embodied_intent


class FakeTrainingService:
    last: "FakeTrainingService | None" = None

    def __init__(self, service: Any) -> None:
        self.service = service
        self.cloud_enabled = True
        self.start_calls: list[Any] = []
        self.plan_calls: list[Any] = []
        FakeTrainingService.last = self

    async def plan(self, spec: Any) -> dict[str, Any]:
        self.plan_calls.append(spec)
        return {
            "message": "plan generated",
            "wallet": {"username": spec.username, "balanceCents": "10000", "availableBalanceCents": "10000"},
            "readyToStart": True,
            "workflow": spec.workflow or "rlinf_vla",
            "params": dict(spec.params or {}),
            "plan": {
                "readyToStart": True,
                "workflow": spec.workflow or "rlinf_vla",
                "params": dict(spec.params or {}),
            },
        }

    async def provider_balance(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "message": "platform balance query success",
            "provider": kwargs.get("provider") or "autodl",
            "balance": {"assets": "20340"},
            "minimumAssets": str(kwargs.get("minimum_assets") or 0),
            "lowBalance": False,
        }

    def cloud_bridge_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": "autodl",
            "deploymentMode": "ssh",
            "configurationReady": True,
        }

    async def configuration_check(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "provider": kwargs.get("provider") or "autodl"}

    async def gpu_skus(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "skus": [
                {
                    "skuId": "seetacloud-4090d-1x",
                    "readyToStart": True,
                    "capabilities": ["ssh-existing", "vla"],
                }
            ]
        }

    async def images(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "images": [
                {
                    "imageId": "seetacloud-current-vla",
                    "readyToStart": True,
                    "capabilities": ["ssh-existing", "vla"],
                }
            ]
        }

    async def runtime_match(self, **kwargs: Any) -> dict[str, Any]:
        return {"readyToStart": True, "matches": [{"sku": {"skuId": "4090d"}, "image": {"imageId": "vla"}}]}

    async def source_preflight(self, **kwargs: Any) -> dict[str, Any]:
        return {"source": kwargs.get("source", {}), "requiresUserConfirmation": True}

    async def start(self, spec: Any) -> TrainingJobStatus:
        self.start_calls.append(spec)
        return TrainingJobStatus(
            job_id="job-1",
            status="Submitted",
            running=True,
            mode="cloud",
            task_name=spec.task_name or "smoke",
            provider=spec.provider or "autodl",
        )

    async def current(self, *, username: str = "") -> TrainingJobStatus:
        return TrainingJobStatus(status="idle", mode="cloud")

    async def status(self, *, job_id: str, username: str = "") -> TrainingJobStatus:
        return TrainingJobStatus(job_id=job_id, status="Running", running=True, mode="cloud")

    async def stop(self, spec: Any) -> TrainingJobStatus:
        return TrainingJobStatus(job_id=spec.job_id, status="STOPPED", running=False, mode="cloud")


class DummyProvider:
    def get_default_model(self) -> str:
        return "dummy"


def _ledger(tmp_path: Path) -> AccountLedger:
    account_ledger = AccountLedger(tmp_path / "ledger.json")
    account_ledger.admin_recharge("pearl", 10_000)
    return account_ledger


def test_cloud_training_tool_plans_with_structured_sources(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="plan",
        username="pearl",
        workflow="rlinf_vla",
        params={
            "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero"},
            "modelSource": {"sourceType": "builtin_policy", "modelFamily": "smolvla"},
        },
    ))

    payload = json.loads(result)
    assert payload["readyToStart"] is True
    assert payload["wallet"]["balanceCents"] == 10_000
    assert payload["executorWallet"]["balanceCents"] == "10000"
    assert payload["billingMode"] == "external"
    assert FakeTrainingService.last is not None
    assert FakeTrainingService.last.plan_calls[0].params["datasetSource"]["uri"] == "hf://HuggingFaceVLA/libero"


def test_cloud_training_tool_normalizes_custom_git_openvla_workflow(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="plan",
        username="pearl",
        workflow="ai_named_openvla_git_eval_job",
        params={
            "policyType": "openvla_oft",
            "datasetSource": {"sourceType": "huggingface", "uri": "hf://lerobot/libero-assets"},
            "modelSource": {"sourceType": "huggingface", "uri": "hf://openvla/openvla-7b"},
        },
    ))

    payload = json.loads(result)
    assert payload["workflow"] == "vla_rl_backend"
    assert FakeTrainingService.last is not None
    spec = FakeTrainingService.last.plan_calls[0]
    assert spec.workflow == "vla_rl_backend"
    assert "backendKind" not in spec.params
    assert "builtinTrainingProfile" not in spec.params
    assert spec.params["policyType"] == "openvla_oft"


def test_cloud_training_tool_provider_balance_is_separate_from_user_wallet(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="provider_balance",
        provider="autodl",
        minimum_assets=1000,
    ))

    payload = json.loads(result)
    assert payload["balanceScope"] == "provider_pool"
    assert payload["balance"]["assets"] == "20340"
    assert payload["minimumAssets"] == "1000"
    assert ledger.wallet("pearl").balance_cents == 10_000


def test_cloud_training_tool_backend_probe_uses_evo_train_contract(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="backend_probe",
        username="pearl",
        provider="autodl",
    ))

    payload = json.loads(result)
    assert payload["probeScope"] == "evo_train_cloud_backend"
    assert payload["bridge"]["deploymentMode"] == "ssh"
    assert payload["checks"]["configuration"]["ok"] is True
    assert payload["checks"]["gpuSkus"]["skus"][0]["skuId"] == "seetacloud-4090d-1x"
    assert payload["checks"]["images"]["images"][0]["imageId"] == "seetacloud-current-vla"
    assert payload["checks"]["current"]["status"] == "idle"


def test_cloud_training_tool_repair_backend_reports_paramiko_dependency(monkeypatch, tmp_path: Path) -> None:
    async def missing_paramiko(self: FakeTrainingService, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": False,
            "missing": ["AUTODL_SSH_PARAMIKO"],
            "warnings": ["AutoDL remote execution requires paramiko"],
            "checks": [
                {
                    "name": "AUTODL_SSH_PARAMIKO",
                    "configured": False,
                    "required": True,
                    "detail": "Python SSH dependency for AutoDL remote execution; interpreter=/opt/evo/bin/python",
                }
            ],
        }

    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    monkeypatch.setattr(FakeTrainingService, "configuration_check", missing_paramiko)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="repair_backend",
        username="pearl",
        provider="autodl",
    ))

    payload = json.loads(result)
    assert payload["status"] == "repair_available"
    assert payload["repairScope"] == "allowlisted_backend_repair"
    assert payload["repairs"][0]["id"] == "autodl_ssh_paramiko"
    assert payload["repairs"][0]["commands"][0] == "/opt/evo/bin/python -m pip install paramiko"
    assert payload["autoApplied"] is False


def test_cloud_training_tool_start_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="start",
        username="pearl",
        workflow="rlinf_vla",
        params={"hourlyPriceCents": 1000},
        confirmed=False,
    ))

    assert "confirmed=true or automation_mode=full_auto is required" in result
    assert ledger.wallet("pearl").frozen_cents == 0


def test_cloud_training_tool_full_auto_can_start_without_confirm(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="start",
        username="pearl",
        provider="autodl",
        workflow="rlinf_vla",
        task_name="cloud-auto",
        params={
            "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero"},
            "modelSource": {"sourceType": "builtin_policy", "modelFamily": "smolvla"},
            "hourlyPriceCents": 1000,
        },
        confirmed=False,
        automationMode="full_auto",
    ))

    payload = json.loads(result)
    assert payload["job_id"] == "job-1"
    assert FakeTrainingService.last is not None
    start_spec = FakeTrainingService.last.start_calls[0]
    assert start_spec.params["automationPolicy"]["mode"] == "full_auto"


def test_cloud_training_tool_start_freezes_balance_and_submits(monkeypatch, tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    monkeypatch.setattr("roboclaw.agent.tools.cloud_training.TrainingService", FakeTrainingService)
    tool = EvoStudioCloudTrainTool(embodied_service=object(), ledger=ledger)

    result = asyncio.run(tool.execute(
        action="start",
        username="pearl",
        provider="autodl",
        workflow="rlinf_vla",
        task_name="cloud-smoke",
        params={
            "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero"},
            "modelSource": {"sourceType": "builtin_policy", "modelFamily": "smolvla"},
            "hourlyPriceCents": 1000,
        },
        confirmed=True,
    ))

    payload = json.loads(result)
    assert payload["job_id"] == "job-1"
    assert payload["billing"]["holdCents"] == 1100
    assert payload["wallet"]["frozenBalanceCents"] == 1100
    assert FakeTrainingService.last is not None
    start_spec = FakeTrainingService.last.start_calls[0]
    assert start_spec.workflow == "rlinf_vla"
    assert start_spec.params["datasetSource"]["uri"] == "hf://HuggingFaceVLA/libero"


def test_agent_loop_registers_cloud_training_tool(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=tmp_path,
        embodied_service=object(),
    )

    assert loop.tools.has("evo_studio_agent_consult")
    assert loop.tools.has("evo_studio_cloud_train")


def test_evo_studio_agent_consult_tool_passes_full_auto_policy(monkeypatch) -> None:
    monkeypatch.setattr("roboclaw.agent.tools.evo_studio_agent.TrainingService", FakeTrainingService)
    monkeypatch.setattr(
        "roboclaw.training.vla_rl.generate_ai_training_plan",
        AsyncMock(return_value={
            "source": "llm",
            "providerModel": "dummy",
            "workflow": "vla_rl_backend",
            "readyToStart": True,
            "params": {
                "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero"},
                "modelSource": {"sourceType": "builtin_policy", "modelFamily": "openvla-oft"},
                "modelFamily": "openvla-oft",
            },
        }),
    )
    tool = EvoStudioAgentConsultTool(embodied_service=object())

    result = asyncio.run(tool.execute(
        task="用 OpenVLA-OFT 评测 LIBERO",
        mode="execute",
        username="pearl",
        provider="ssh",
        workflow="vla_rl_backend",
        context={"backend": "ssh"},
        params={
            "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero"},
            "modelSource": {"sourceType": "builtin_policy", "modelFamily": "openvla-oft"},
            "modelFamily": "openvla-oft",
        },
        automationMode="full_auto",
    ))

    payload = json.loads(result)
    assert payload["started"] is True
    assert payload["executionPolicy"]["fullAuto"] is True
    assert payload["executionPolicy"]["startsPaidCompute"] is True
    assert payload["executionPolicy"]["allowAgentRepairSameRuntime"] is True
    assert FakeTrainingService.last is not None
    start_spec = FakeTrainingService.last.start_calls[0]
    assert start_spec.params["automationPolicy"]["mode"] == "full_auto"


def test_embodied_intent_routes_starvla_xlerobot_to_cloud_training() -> None:
    intent = classify_embodied_intent("用 StarVLA 在 XLeRobot 数据上做评测并调测")

    assert intent.route == "cloud_training"
    assert intent.workflow == "vla_rl_backend"
    assert "vla" in intent.domains
    assert "starvla" in intent.frameworks
    assert "xlerobot" in intent.frameworks
    assert intent.params["modelFamily"] == "starvla"
    assert intent.params["robotAdapter"] == "xlerobot"


def test_evo_studio_agent_consult_does_not_misroute_ros2_slam_to_vla(monkeypatch) -> None:
    monkeypatch.setattr("roboclaw.agent.tools.evo_studio_agent.TrainingService", FakeTrainingService)
    tool = EvoStudioAgentConsultTool(embodied_service=object())

    result = asyncio.run(tool.execute(
        task="用 KISS-ICP 跑 ROS2 lidar odometry，并检查 topic 和 tf",
        mode="execute",
        username="pearl",
        automationMode="full_auto",
    ))

    payload = json.loads(result)
    assert payload["capabilityRoute"] == "robotics_runtime"
    assert payload["readyForConfirmation"] is False
    assert payload["started"] is False
    assert payload["embodiedIntent"]["params"]["recommendedPackage"] == "kiss-icp"
    assert FakeTrainingService.last is not None
    assert FakeTrainingService.last.plan_calls == []
    assert FakeTrainingService.last.start_calls == []


def test_agent_loop_auto_delegates_vla_ssh_requests(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("roboclaw.agent.tools.evo_studio_agent.TrainingService", FakeTrainingService)
    monkeypatch.setattr(
        "roboclaw.training.vla_rl.generate_ai_training_plan",
        AsyncMock(return_value={
            "source": "llm",
            "providerModel": "dummy",
            "workflow": "vla_rl_backend",
            "params": {
                "backendKind": "openvla_oft",
                "modelFamily": "openvla",
                "policyType": "openvla",
                "repoUrl": "https://github.com/moojink/openvla-oft.git",
                "datasetSource": {"sourceType": "public_reference", "uri": "hf://HuggingFaceVLA/libero", "format": "libero"},
                "modelSource": {
                    "sourceType": "public_model_repo",
                    "uri": "hf://moojink/openvla-7b-oft-finetuned-libero-spatial",
                    "format": "huggingface_transformers",
                },
            },
            "humanSummary": "复现 OpenVLA-OFT baseline。",
        }),
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=tmp_path,
        embodied_service=object(),
    )

    result = asyncio.run(loop.process_direct(
        "在SSH后端复现OpenVLA-OFT baseline，libero pro仿真平台的",
        session_key="web:test",
        channel="web",
        chat_id="test",
    ))

    assert "已由 Evo Studio 后端总控接管" in result
    assert "我不会要求你在聊天里粘贴 SSH 密码或 key" in result
    assert FakeTrainingService.last is not None
    assert FakeTrainingService.last.plan_calls
    spec = FakeTrainingService.last.plan_calls[0]
    assert spec.provider == "autodl"
    assert spec.workflow == "rlinf_vla"
    assert spec.params["datasetSource"]["uri"] == "hf://HuggingFaceVLA/libero"
    assert spec.params["modelSource"]["sourceType"] == "public_model_repo"
    assert spec.params["modelSource"]["uri"] == "hf://moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    assert spec.params["repoUrl"] == "https://github.com/RLinf/RLinf.git"
    assert spec.params["backendKind"] == "rlinf"
    assert spec.params["builtinTrainingProfile"] == "rlinf_openvla_oft_libero_pro_eval"
    assert FakeTrainingService.last.start_calls
    start_spec = FakeTrainingService.last.start_calls[0]
    assert start_spec.params["automationPolicy"]["mode"] == "full_auto"


def test_agent_loop_does_not_auto_delegate_explain_only_questions(tmp_path: Path) -> None:
    assert AgentLoop._should_auto_delegate_evo_studio("为什么这个 API 不能调用各种 skill 完成训练计划啊") is False
    assert AgentLoop._should_auto_delegate_evo_studio("在SSH后端复现OpenVLA-OFT baseline") is True
    assert AgentLoop._should_auto_delegate_evo_studio("用 KISS-ICP 跑 ROS2 lidar odometry") is True
    assert AgentLoop._should_auto_delegate_evo_studio("为什么 KISS-ICP 和 SLAM 有关") is False
