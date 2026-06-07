"""VLA-RL planning, artifact review, and deployability checks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from roboclaw.embodied.policy import policy_registry
from roboclaw.providers.base import LLMProvider
from roboclaw.training.ai_planner import generate_ai_training_plan, merge_ai_plan
from roboclaw.training.rlinf_catalog import (
    apply_rlinf_config_contract,
    discover_rlinf_catalog,
    match_rlinf_config_name,
    rlinf_shell_backend_interface,
)
from roboclaw.training.schema import TrainingPlanSpec
from roboclaw.training.service import TrainingService


_MODEL_ALIASES = {
    "uni-navid": "uni-navid",
    "uninavid": "uni-navid",
    "gr00tn1": "gr00tn1",
    "gr00t n1": "gr00tn1",
    "gr00t": "gr00t",
    "pi0.5": "pi0.5",
    "pi05": "pi0.5",
    "pi0": "pi0",
    "dm0": "dm0",
    "cogact": "cogact",
    "openvla-oft": "openvla-oft",
    "openvla oft": "openvla-oft",
    "openvla": "openvla",
    "oft": "oft",
    "starvla": "starvla",
    "star vla": "starvla",
    "navila": "navila",
}

_ROBOT_ALIASES = {
    "so-101": "so-101",
    "so101": "so-101",
    "xlerobot": "xlerobot",
    "xle robot": "xlerobot",
}

_DEFAULT_TRAINING_PROFILES = {
    "pi0": "dexbotic_pi0_rlinf",
    "gr00tn1": "roboclaw_rlinf_backend",
    "gr00t": "roboclaw_rlinf_backend",
    "dm0": "dexbotic_dm0_rlinf",
    "cogact": "roboclaw_rlinf_backend",
    "oft": "openvla_oft_libero_eval",
    "openvla": "openvla_oft_libero_eval",
    "openvla-oft": "openvla_oft_libero_eval",
    "starvla": "roboclaw_rlinf_backend",
    "navila": "roboclaw_rlinf_backend",
    "uni-navid": "roboclaw_rlinf_backend",
}

_BACKEND_INTERFACE_CATALOG = {
    "rlinf": {
        "interfaceVersion": "vla-rl-backend/v1",
        "workflow": "rlinf_vla",
        "launchModes": ["project_backend", "rlinf_frontend"],
        "launcherKinds": ["python_module", "python_script", "deepspeed_script"],
        "requiredParams": ["repoUrl", "workdir", "configName", "launcherModule", "datasetPath", "artifactPath"],
        "registryInjection": {
            "field": "rlinfExtModule",
            "env": "RLINF_EXT_MODULE",
            "preflightImport": True,
        },
        "envExports": [
            "VLA_RL_BACKEND_KIND",
            "VLA_RL_BACKEND_EXT_MODULE",
            "RLINF_EXT_MODULE",
            "RLINF_DATASET_PATH",
            "RLINF_CHECKPOINT_PATH",
            "VLA_RL_CONTRACT_PATH",
            "VLA_RL_MODEL_FAMILY",
            "VLA_RL_TRAINING_MODE",
        ],
        "preflightChecks": [
            "import rlinf",
            "import launcherModule",
            "import rlinfExtModule when provided",
            "import envModule/rewardModule when provided",
        ],
        "launcherContract": {
            "python_module": "python -m {launcherModule} --config-name {configName}",
            "python_script": "python {scriptPath} --config-name {configName}",
            "deepspeed_script": "deepspeed {scriptPath}",
        },
        "algorithmToLauncherKind": {
            "ppo": "python_module",
            "sac": "python_module",
            "grpo": "python_module",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "datasetPath", "checkpointPath", "artifactPath", "metricPaths"],
            "successMetricField": "successMetric",
        },
    },
    "custom": {
        "interfaceVersion": "vla-rl-backend/v1",
        "workflow": "vla_rl_backend",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_module", "python_script", "deepspeed_script"],
        "requiredParams": ["repoUrl", "workdir", "launcherModule", "artifactPath"],
        "registryInjection": {
            "field": "backendExtModule",
            "env": "VLA_RL_BACKEND_EXT_MODULE",
            "preflightImport": True,
        },
        "envExports": [
            "VLA_RL_BACKEND_KIND",
            "VLA_RL_BACKEND_EXT_MODULE",
            "VLA_RL_CONTRACT_PATH",
            "VLA_RL_MODEL_FAMILY",
            "VLA_RL_TRAINING_MODE",
        ],
        "preflightChecks": [
            "import launcherModule",
            "import backendExtModule when provided",
            "import envModule/rewardModule when provided",
        ],
        "launcherContract": {
            "python_module": "python -m {launcherModule} --config-name {configName}",
            "python_script": "python {scriptPath} --config-name {configName}",
            "deepspeed_script": "deepspeed {scriptPath}",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "datasetPath", "checkpointPath", "artifactPath", "metricPaths"],
            "successMetricField": "successMetric",
        },
    },
    "openvla_oft": {
        "interfaceVersion": "vla-rl-backend/v1",
        "workflow": "vla_rl_backend",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_script"],
        "requiredParams": ["repoUrl", "workdir", "scriptPath", "checkpointPath", "artifactPath"],
        "registryInjection": {
            "field": "",
            "env": "",
            "preflightImport": False,
        },
        "envExports": [
            "TRANSFORMERS_CACHE",
            "HF_HOME",
            "MUJOCO_GL",
            "VLA_RL_CONTRACT_PATH",
            "VLA_RL_MODEL_FAMILY",
            "VLA_RL_TRAINING_MODE",
        ],
        "preflightChecks": [
            "repo has experiments/robot/libero/run_libero_eval.py",
            "runtime image provides OpenVLA-OFT dependencies",
            "LIBERO/MuJoCo assets are installed",
        ],
        "launcherContract": {
            "python_script": "python {scriptPath} --pretrained_checkpoint {checkpointPath} --task_suite_name {suite}",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "checkpointPath", "artifactPath", "metricPaths"],
            "successMetricField": "success_rate",
        },
    },
    "lerobot": {
        "interfaceVersion": "vla-rl-backend/v1",
        "workflow": "vla_rl_backend",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_module", "python_script"],
        "requiredParams": ["repoUrl", "workdir", "launcherModule", "datasetPath", "checkpointPath", "policyFamily"],
        "registryInjection": {
            "field": "backendExtModule",
            "env": "VLA_RL_BACKEND_EXT_MODULE",
            "preflightImport": True,
        },
        "envExports": [
            "VLA_RL_BACKEND_KIND",
            "VLA_RL_BACKEND_EXT_MODULE",
            "RLINF_DATASET_PATH",
            "RLINF_CHECKPOINT_PATH",
            "VLA_RL_CONTRACT_PATH",
            "VLA_RL_MODEL_FAMILY",
        ],
        "preflightChecks": [
            "import lerobot",
            "import launcherModule",
            "dataset path exists or is resolvable by the launcher",
        ],
        "launcherContract": {
            "python_module": "python -m {launcherModule} --config-name {configName} --dataset_path {datasetPath} --checkpoint_path {checkpointPath}",
            "python_script": "python {scriptPath} --config-name {configName} --dataset_path {datasetPath} --checkpoint_path {checkpointPath}",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "policyFamily", "datasetPath", "checkpointPath", "artifactPath", "metricPaths"],
            "successMetricField": "eval/loss",
        },
    },
    "dexbotic": {
        "interfaceVersion": "vla-rl-backend/v1",
        "workflow": "vla_rl_backend",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_module", "python_script", "deepspeed_script"],
        "requiredParams": ["repoUrl", "workdir", "launcherModule", "datasetPath", "artifactPath"],
        "registryInjection": {"field": "backendExtModule", "env": "VLA_RL_BACKEND_EXT_MODULE", "preflightImport": True},
        "envExports": ["VLA_RL_BACKEND_KIND", "VLA_RL_BACKEND_EXT_MODULE", "VLA_RL_CONTRACT_PATH"],
        "preflightChecks": ["import launcherModule", "import backendExtModule when provided", "Dexbotic project path is importable"],
        "launcherContract": {
            "python_module": "python -m {launcherModule} --config-name {configName}",
            "deepspeed_script": "deepspeed {scriptPath}",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "datasetPath", "artifactPath", "metricPaths"],
            "successMetricField": "success_rate",
        },
    },
}

_PROFILE_CATALOG = {
    "dexbotic_pi0_rlinf": {
        "title": "Pi0 RL post-training",
        "backendKind": "rlinf",
        "modelFamily": "pi0",
        "policyTypes": ["pi0"],
        "trainingMode": "rl_post_train",
        "launchMode": "project_backend",
        "status": "experimental",
        "requiredParams": ["repoUrl", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "rlinf",
    },
    "dexbotic_dm0_rlinf": {
        "title": "DM0 RL post-training",
        "backendKind": "rlinf",
        "modelFamily": "dm0",
        "policyTypes": [],
        "trainingMode": "rl_post_train",
        "launchMode": "project_backend",
        "status": "experimental",
        "requiredParams": ["repoUrl", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "rlinf",
    },
    "dexbotic_simplevla_rl": {
        "title": "SimpleVLA-RL style script launcher",
        "backendKind": "dexbotic",
        "modelFamily": "simplevla",
        "policyTypes": [],
        "trainingMode": "rl_post_train",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "scriptPath", "datasetPath", "checkpointPath"],
        "recommendedBackend": "rlinf",
    },
    "roboclaw_rlinf_backend": {
        "title": "Generic project-owned VLA-RL backend",
        "backendKind": "rlinf",
        "modelFamily": "custom",
        "policyTypes": [],
        "trainingMode": "rl_post_train",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "workdir", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "rlinf",
    },
    "roboclaw_grpo_backend": {
        "title": "RoboClaw GRPO post-training backend",
        "backendKind": "rlinf",
        "modelFamily": "custom",
        "policyTypes": ["pi0", "pi05", "groot"],
        "trainingMode": "rl_post_train",
        "algorithm": "grpo",
        "groupSize": 8,
        "placementStrategy": "single_node",
        "launchMode": "project_backend",
        "status": "template",
        "requiredParams": ["repoUrl", "workdir", "configName", "configPath", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "rlinf",
    },
    "roboclaw_lerobot_backend": {
        "title": "Generic project-owned LeRobot fine-tuning backend",
        "backendKind": "lerobot",
        "modelFamily": "custom",
        "policyTypes": ["act", "diffusion", "pi0", "pi05", "groot", "smolvla", "xvla"],
        "trainingMode": "supervised_finetune",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "lerobot",
    },
    "roboclaw_dexbotic_backend": {
        "title": "Dexbotic-style project backend",
        "backendKind": "dexbotic",
        "modelFamily": "custom",
        "policyTypes": ["pi0", "pi05", "groot", "smolvla", "xvla"],
        "trainingMode": "rl_post_train",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "launcherModule", "datasetPath", "checkpointPath"],
        "recommendedBackend": "dexbotic",
    },
    "roboclaw_custom_backend": {
        "title": "Custom project-owned VLA/RL backend",
        "backendKind": "custom",
        "modelFamily": "custom",
        "policyTypes": [],
        "trainingMode": "custom",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "launcherModule", "artifactPath"],
        "recommendedBackend": "custom",
    },
    "openvla_oft_libero_eval": {
        "title": "OpenVLA-OFT LIBERO baseline evaluation",
        "backendKind": "openvla_oft",
        "modelFamily": "openvla",
        "policyTypes": ["openvla", "oft", "openvla-oft"],
        "trainingMode": "baseline_reproduction",
        "launchMode": "project_backend",
        "status": "adapter",
        "requiredParams": ["repoUrl", "workdir", "scriptPath", "checkpointPath", "artifactPath"],
        "recommendedBackend": "openvla_oft",
    },
    "rlinf_openvla_oft_libero_pro_eval": {
        "title": "RLinf OpenVLA-OFT LIBERO-Pro evaluation",
        "backendKind": "rlinf",
        "modelFamily": "openvla-oft",
        "policyTypes": ["openvla", "openvla-oft", "oft"],
        "trainingMode": "simulation_eval",
        "launchMode": "project_backend",
        "status": "official",
        "requiredParams": ["repoUrl", "workdir", "scriptPath", "configName", "artifactPath"],
        "recommendedBackend": "rlinf",
    },
}


@dataclass(frozen=True)
class VLAPlanRequest:
    username: str = ""
    message: str = ""
    workflow: str = ""
    params: Mapping[str, Any] | None = None
    provider: str = "autodl"
    sku_id: str = ""
    image_id: str = ""


class VLARLService:
    """RoboClaw control-plane helpers for VLA-RL workflows."""

    def __init__(self, training: TrainingService, *, llm_provider: LLMProvider | None = None) -> None:
        self._training = training
        self._llm_provider = llm_provider

    async def plan(self, request: VLAPlanRequest) -> dict[str, Any]:
        params = normalize_capabilities(request.message, dict(request.params or {}))
        params.setdefault("launchMode", "project_backend")
        workflow = _workflow_for_request(request.workflow, params)
        spec = TrainingPlanSpec(
            username=request.username,
            message=request.message,
            workflow=workflow,
            params=params,
            provider=request.provider,
            sku_id=request.sku_id,
            image_id=request.image_id,
        )
        ai_plan = await generate_ai_training_plan(
            self._llm_provider,
            spec,
            workflow=workflow,
            params=params,
        )
        workflow, params = merge_ai_plan(ai_plan=ai_plan, workflow=workflow, params=params)
        params = normalize_capabilities(request.message, params)
        workflow = _workflow_for_request(workflow, params)
        spec = TrainingPlanSpec(
            username=request.username,
            message=request.message,
            workflow=workflow,
            params=params,
            provider=request.provider,
            sku_id=request.sku_id,
            image_id=request.image_id,
        )
        if self._training.cloud_enabled:
            response = await self._training.plan(spec)
        else:
            warnings = ["EVO_Train bridge is not connected; generated plan only."]
            missing_fields = ["EVO_Train bridge"]
            if ai_plan:
                missing_fields.extend(str(item) for item in ai_plan.get("missingFields") or [])
                warnings.extend(str(item) for item in ai_plan.get("warnings") or [])
            response = {
                "message": "AI plan generated locally; cloud execution is not connected.",
                "plan": {
                    "workflow": workflow,
                    "params": params,
                    "readyToStart": False,
                    "missingFields": missing_fields,
                    "warnings": warnings,
                },
            }
        plan = dict(response.get("plan") or {})
        hints = _deployability_hints(plan.get("params") if isinstance(plan.get("params"), dict) else params)
        response["vlaPlan"] = {
            "workflow": plan.get("workflow") or workflow,
            "params": plan.get("params") or params,
            "readyToStart": bool(plan.get("readyToStart")),
            "missingFields": list(plan.get("missingFields") or []),
            "warnings": list(plan.get("warnings") or []),
            "deployabilityHints": hints,
        }
        if ai_plan:
            ai_source = str(ai_plan.get("source") or "")
            response["aiPlan"] = ai_plan
            response["vlaPlan"]["aiSummary"] = ai_plan.get("humanSummary", "")
            response["vlaPlan"]["planSteps"] = list(ai_plan.get("planSteps") or [])
            response["vlaPlan"]["evaluationPlan"] = list(ai_plan.get("evaluationPlan") or [])
            response["vlaPlan"]["resourceHints"] = list(ai_plan.get("resourceHints") or [])
            response["vlaPlan"]["safetyChecks"] = list(ai_plan.get("safetyChecks") or [])
            response["vlaPlan"]["clarifyingQuestions"] = list(ai_plan.get("clarifyingQuestions") or [])
            response["vlaPlan"]["intentUnderstanding"] = dict(ai_plan.get("intentUnderstanding") or {})
            response["vlaPlan"]["planner"] = {
                "source": ai_plan.get("source"),
                "providerModel": ai_plan.get("providerModel"),
            }
            if ai_source and ai_source != "llm":
                response["vlaPlan"]["readyToStart"] = False
                response["vlaPlan"]["missingFields"] = [
                    "AI planner did not complete",
                    *response["vlaPlan"]["missingFields"],
                ]
                response["vlaPlan"]["warnings"] = [
                    "AI planner did not produce a usable structured plan; do not treat the deterministic fallback as a launch-ready plan.",
                    *response["vlaPlan"]["warnings"],
                ]
        return response

    def profiles(self) -> dict[str, Any]:
        return vla_profile_catalog()

    def rlinf_catalog(self) -> dict[str, Any]:
        return discover_rlinf_catalog()

    def playground(self) -> dict[str, Any]:
        return vla_playground_spec()

    def review_artifact(self, *, contract_path: str = "", contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
        loaded = _load_contract(contract_path, contract)
        metrics = _load_first_json_path(loaded.get("metricPaths"))
        success_metric = str(loaded.get("successMetric") or "success_rate")
        success_value = _as_float(metrics.get(success_metric)) if isinstance(metrics, dict) else None
        checkpoint_path = str(loaded.get("checkpointPath") or "")
        artifact_path = str(loaded.get("artifactPath") or "")
        checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).expanduser().exists())
        artifact_exists = bool(artifact_path and Path(artifact_path).expanduser().exists())
        return {
            "contract": loaded,
            "metrics": metrics,
            "successMetric": success_metric,
            "successValue": success_value,
            "checkpointExists": checkpoint_exists,
            "artifactExists": artifact_exists,
            "deployability": deployability_gate(loaded),
        }


def normalize_capabilities(message: str, params: dict[str, Any]) -> dict[str, Any]:
    lowered = message.lower()
    enriched = dict(params)
    explicit_rlinf = (
        bool(str(enriched.get("rlinfConfigName") or "").strip())
        or "rlinf" in lowered
        or str(enriched.get("backendKind") or "").lower() == "rlinf"
        or str(enriched.get("workflow") or "").lower() == "rlinf_vla"
        or str(enriched.get("repoUrl") or "").rstrip("/") == "https://github.com/RLinf/RLinf.git"
    )
    requested_rlinf_config = str(enriched.get("rlinfConfigName") or "").strip()
    if explicit_rlinf and not requested_rlinf_config:
        requested_rlinf_config = str(enriched.get("configName") or "").strip()
    if not requested_rlinf_config and ("rlinf" in lowered or str(enriched.get("backendKind") or "") == "rlinf"):
        requested_rlinf_config = match_rlinf_config_name(message)
    for token, model_family in _MODEL_ALIASES.items():
        if token in lowered and not enriched.get("modelFamily"):
            enriched["modelFamily"] = model_family
            break
    for token, robot_adapter in _ROBOT_ALIASES.items():
        if token in lowered and not enriched.get("robotAdapter"):
            enriched["robotAdapter"] = robot_adapter
            break
    if _contains_any(lowered, ("co-training", "cotrain", "共训练", "联合优化")):
        enriched.setdefault("trainingMode", "co_training")
    elif _contains_any(lowered, ("后训练", "post-training", "rl post", "vla+rl")):
        enriched.setdefault("trainingMode", "rl_post_train")
    if _contains_any(lowered, ("action expert", "动作专家")):
        enriched.setdefault("coTrainingTargets", ["action_expert", "llm"])
    if "grpo" in lowered:
        enriched.setdefault("algorithm", "grpo")
    elif "ppo" in lowered:
        enriched.setdefault("algorithm", "ppo")
    if "blackwell" in lowered:
        enriched.setdefault("imageProfile", "blackwell")
    if "dexbotic" in lowered:
        enriched.setdefault("repoUrl", "https://github.com/dexmal/dexbotic.git")
        enriched.setdefault("workdir", "/root/autodl-tmp/dexbotic")
        enriched.setdefault("launcherModule", "dexbotic.rl.model_rl_libero_pi0")
        enriched.setdefault("rlinfExtModule", "dexbotic.rl.rlinf_registry")
    if requested_rlinf_config:
        enriched = apply_rlinf_config_contract(
            enriched,
            config_name=requested_rlinf_config,
            message=message,
        )
        return enriched
    if "libero" in lowered:
        suite = "libero_10" if _contains_any(lowered, ("libero pro", "libero-pro", "liberopro", "libero plus", "libero-plus", "liberoplus")) else "libero_spatial"
        if "libero_object" in lowered or "object" in lowered:
            suite = "libero_object"
        elif "libero_goal" in lowered or "goal" in lowered:
            suite = "libero_goal"
        elif "libero_10" in lowered or "libero 10" in lowered:
            suite = "libero_10"
        enriched.setdefault("benchmark", "libero")
        enriched.setdefault("suite", suite)
        enriched.setdefault(
            "datasetSource",
            {
                "sourceType": "public_reference",
                "datasetId": "libero",
                "uri": "hf://HuggingFaceVLA/libero",
                "format": "libero",
                "benchmark": "libero",
                "suite": suite,
            },
        )
        enriched.setdefault("datasetFormat", "libero")
    if _contains_any(lowered, ("openvla", "openvla-oft", "oft")):
        suite = str(enriched.get("suite") or "libero_spatial")
        if _contains_any(lowered, ("rlinf", "libero pro", "libero-pro", "liberopro", "libero plus", "libero-plus", "liberoplus")):
            libero_type = "plus" if _contains_any(lowered, ("libero plus", "libero-plus", "liberoplus")) else "pro"
            config_name = "libero_10_grpo_openvlaoft_eval" if suite == "libero_10" else f"{suite}_grpo_openvlaoft"
            artifact_path = f"/root/autodl-tmp/RLinf/outputs/{config_name}_{libero_type}"
            enriched.update(
                {
                    "backendKind": "rlinf",
                    "workflow": "rlinf_vla",
                    "modelFamily": "openvla-oft",
                    "policyType": "openvla-oft",
                    "policyFamily": "openvla-oft",
                    "algorithm": "grpo",
                    "trainingMode": "simulation_eval",
                    "builtinTrainingProfile": "rlinf_openvla_oft_libero_pro_eval",
                    "repoUrl": "https://github.com/RLinf/RLinf.git",
                    "branch": "main",
                    "workdir": "/root/autodl-tmp/RLinf",
                    "launchMode": "project_backend",
                    "launcherKind": "python_script",
                    "launcherModule": "",
                    "scriptPath": "examples/embodiment/eval_embodied_agent.py",
                    "entrypoint": "examples/embodiment/eval_embodied_agent.py",
                    "configName": config_name,
                    "configPath": f"examples/embodiment/config/{config_name}.yaml",
                    "artifactPath": artifact_path,
                    "contractPath": f"{artifact_path}/run_contract.json",
                    "successMetric": "success_rate",
                    "metricPaths": [f"{artifact_path}/metrics.json", f"{artifact_path}/rollout_summary.json"],
                    "resultFiles": ["run_contract.json", "metrics.json", "rollout_summary.json", "logs"],
                    "evalEpisodes": int(enriched.get("evalEpisodes") or 2),
                    "bootstrapProfile": "",
                    "backendInterface": rlinf_shell_backend_interface(
                        libero_type=libero_type,
                        robot_platform="LIBERO",
                        direct_python=True,
                    ),
                }
            )
            if not isinstance(enriched.get("modelSource"), dict) or str(
                (enriched.get("modelSource") or {}).get("modelFamily")
                or (enriched.get("modelSource") or {}).get("uri")
                or ""
            ).strip().lower() in {"", "auto", "unknown"}:
                enriched["modelSource"] = {
                    "sourceType": "rlinf_config_default",
                    "modelFamily": "openvla-oft",
                    "format": "rlinf_config",
                }
                enriched.setdefault("checkpointFormat", "rlinf_config")
            enriched.setdefault("launcherArgs", [])
            enriched.setdefault(
                "trainingContract",
                {
                    "interfaceKind": "rlinf_runner",
                    "framework": "rlinf",
                    "sources": {
                        "code": {"uri": "https://github.com/RLinf/RLinf.git"},
                        "dataset": enriched.get("datasetSource"),
                        "model": enriched.get("modelSource"),
                    },
                    "runner": "EmbodiedRunner",
                    "algorithm": {"name": "grpo"},
                    "env": {"benchmark": "libero", "suite": suite, "liberoType": libero_type},
                    "runtime": {"placementStrategy": enriched.get("placementStrategy", "single_node")},
                    "artifacts": {"path": artifact_path},
                },
            )
            return enriched
        enriched.setdefault("backendKind", "openvla_oft")
        enriched.setdefault("modelFamily", "openvla")
        enriched.setdefault("policyType", "openvla")
        enriched.setdefault("policyFamily", "openvla")
        enriched.setdefault("trainingMode", "baseline_reproduction")
        enriched.setdefault("builtinTrainingProfile", "openvla_oft_libero_eval")
        enriched.setdefault("repoUrl", "https://github.com/moojink/openvla-oft.git")
        enriched.setdefault("branch", "main")
        enriched.setdefault("workdir", "/root/autodl-tmp/openvla-oft")
        enriched.setdefault("scriptPath", "experiments/robot/libero/run_libero_eval.py")
        enriched.setdefault("configName", "libero_oft_eval")
        enriched.setdefault("artifactPath", "/workspace/outputs")
        enriched.setdefault(
            "modelSource",
            {
                "sourceType": "catalog_lookup_required",
                "modelFamily": "openvla",
                "format": "auto",
            },
        )
        enriched.setdefault("launcherArgs", ["--center_crop", "True", "--num_trials_per_task", "2"])
    model_family = str(enriched.get("modelFamily") or "")
    if model_family in _DEFAULT_TRAINING_PROFILES:
        enriched.setdefault("builtinTrainingProfile", _DEFAULT_TRAINING_PROFILES[model_family])
    return enriched


def _rlinf_libero_shell_backend_interface(libero_type: str) -> dict[str, Any]:
    return {
        "interfaceVersion": "vla-rl-backend/v1",
        "backendKind": "rlinf",
        "workflow": "rlinf_vla",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_script"],
        "requiredParams": ["repoUrl", "workdir", "scriptPath", "configName", "artifactPath"],
        "registryInjection": {"field": "rlinfExtModule", "env": "RLINF_EXT_MODULE", "preflightImport": False},
        "envExports": {
            "LIBERO_TYPE": f"literal:{libero_type}",
            "ROBOT_PLATFORM": "literal:LIBERO",
            "VLA_RL_CONTRACT_PATH": "contractPath",
            "VLA_RL_MODEL_FAMILY": "modelFamily",
            "VLA_RL_TRAINING_MODE": "trainingMode",
        },
        "preflightImports": ["rlinf"],
        "preflightCommands": [
            "test -f {scriptPath}",
            "test -f {configPath}",
        ],
        "usePreflightCommands": True,
        "useLauncherContract": True,
        "launcherContract": {
            "python_script": "bash {scriptPath} {configName} LIBERO",
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "artifactPath", "metricPaths"],
            "successMetricField": "success_rate",
        },
    }


def vla_profile_catalog() -> dict[str, Any]:
    supported_policy_types = sorted(policy_registry.supported_types())
    backend_interfaces = _configured_backend_interfaces()
    rlinf_catalog = discover_rlinf_catalog()
    profiles = []
    for profile_id, profile in _PROFILE_CATALOG.items():
        item = dict(profile)
        item["id"] = profile_id
        item["availableInPolicyRegistry"] = all(
            policy_type in supported_policy_types for policy_type in item.get("policyTypes", [])
        )
        profiles.append(item)
    return {
        "workflow": "vla_rl_backend",
        "compatibleWorkflows": ["vla_rl_backend", "rlinf_vla"],
        "supportedPolicyTypes": supported_policy_types,
        "backendKindExtensible": True,
        "backendInterfaces": backend_interfaces,
        "backendInterfaceConfigurable": True,
        "backendInterfaceConfigSources": [
            "ROBOCLAW_VLA_BACKEND_INTERFACES_JSON",
            "ROBOCLAW_VLA_BACKEND_INTERFACES_FILE",
        ],
        "rlinfCatalog": {
            "configured": rlinf_catalog["configured"],
            "repoUrl": rlinf_catalog["repoUrl"],
            "repoPath": rlinf_catalog["repoPath"],
            "configCount": rlinf_catalog["configCount"],
            "benchmarks": rlinf_catalog["benchmarks"],
            "algorithms": rlinf_catalog["algorithms"],
            "modelFamilies": rlinf_catalog["modelFamilies"],
            "entrypoint": "/api/vla-rl/rlinf-catalog",
        },
        "defaultTrainingProfiles": dict(_DEFAULT_TRAINING_PROFILES),
        "profiles": profiles,
    }


def vla_playground_spec() -> dict[str, Any]:
    profiles = vla_profile_catalog()
    return {
        "kind": "roboclaw_vla_training_playground/v1",
        "title": "VLA/RL Training Playground",
        "description": "Guide a robot training request from intent to runtime match, cloud execution, artifact review, and deployability checks.",
        "defaultProvider": "autodl",
        "entrypoints": {
            "profiles": "/api/vla-rl/profiles",
            "plan": "/api/vla-rl/plan",
            "runtimeMatch": "/api/train/runtime-match",
            "gpuSkus": "/api/train/gpu-skus",
            "images": "/api/train/images",
            "start": "/api/train/cloud/start",
            "status": "/api/train/cloud/status/{job_id}",
            "artifactReview": "/api/vla-rl/artifact-review",
            "deployability": "/api/vla-rl/deployability",
        },
        "stages": [
            {
                "id": "intent",
                "title": "Training intent",
                "inputs": ["message", "workflow", "params.modelFamily", "params.algorithm", "params.trainingMode"],
                "next": "plan",
            },
            {
                "id": "plan",
                "title": "Structured workflow plan",
                "api": "/api/vla-rl/plan",
                "checks": ["missingFields", "warnings", "deployabilityHints"],
                "next": "runtime_match",
            },
            {
                "id": "runtime_match",
                "title": "Runtime match",
                "api": "/api/train/runtime-match",
                "checks": ["backendKind", "modelFamily", "benchmark", "requiredCapabilities", "gpuMemoryGb"],
                "next": "confirm",
            },
            {
                "id": "confirm",
                "title": "Cost and launch confirmation",
                "checks": ["firstHourCost", "walletAvailable", "skuId", "imageId"],
                "next": "execute",
            },
            {
                "id": "execute",
                "title": "Cloud execution",
                "api": "/api/train/cloud/start",
                "checks": ["taskName", "jobId", "status"],
                "next": "review",
            },
            {
                "id": "review",
                "title": "Artifact review",
                "api": "/api/vla-rl/artifact-review",
                "checks": ["run_contract.json", "metricPaths", "checkpointPath", "artifactPath"],
                "next": "deployability",
            },
            {
                "id": "deployability",
                "title": "Deployability gate",
                "api": "/api/vla-rl/deployability",
                "checks": ["robotEmbodiment", "observationSchema", "actionSchema"],
            },
        ],
        "inputSchema": {
            "requiredUserFields": ["username", "message"],
            "normalUserFields": ["message", "modelFamily", "algorithm", "benchmark", "datasetPath", "budgetCents"],
            "advancedFields": ["workflow", "params", "sku_id", "image_id", "backendInterface"],
            "adminOnlyFields": ["providerToken", "autodlGpuSpecUuid", "autodlImageUuid", "sshHost", "sshKeyPath"],
        },
        "guardrails": [
            "Do not ask normal users for provider tokens, SSH details, or AutoDL UUIDs.",
            "Do not start paid compute until runtime-match returns a compatible SKU/image pair and the user confirms cost.",
            "Prefer a smoke test when repo, dataset, image, or metric is uncertain.",
            "Treat backend names as supported only when a backendInterface contract declares launcher, preflight, and artifact behavior.",
            "Treat RoboClaw-owned RLinf launcher as experimental until validated inside a live RLinf image.",
        ],
        "backendInterfaces": profiles["backendInterfaces"],
        "profiles": profiles["profiles"],
        "recommendedSmokeParams": {
            "workflow": "rlinf_vla",
            "params": {
                "builtinTrainingProfile": "roboclaw_grpo_backend",
                "modelFamily": "pi0",
                "algorithm": "grpo",
                "benchmark": "libero",
                "evalEpisodes": 1,
                "maxSteps": 10,
            },
        },
    }


def _configured_backend_interfaces() -> dict[str, Any]:
    interfaces = {key: dict(value) for key, value in _BACKEND_INTERFACE_CATALOG.items()}
    for source in (_load_backend_interfaces_file(), _load_backend_interfaces_env()):
        for key, value in source.items():
            if isinstance(value, dict):
                merged = dict(interfaces.get(str(key), {}))
                merged.update(value)
                interfaces[str(key)] = merged
    return interfaces


def _load_backend_interfaces_env() -> dict[str, Any]:
    raw = os.environ.get("ROBOCLAW_VLA_BACKEND_INTERFACES_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("ROBOCLAW_VLA_BACKEND_INTERFACES_JSON must be a JSON object")
    return parsed


def _load_backend_interfaces_file() -> dict[str, Any]:
    path = os.environ.get("ROBOCLAW_VLA_BACKEND_INTERFACES_FILE", "").strip()
    if not path:
        return {}
    parsed = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("ROBOCLAW_VLA_BACKEND_INTERFACES_FILE must contain a JSON object")
    return parsed


def _workflow_for_request(workflow: str, params: Mapping[str, Any]) -> str:
    requested = workflow.strip()
    backend_kind = str(params.get("backendKind") or params.get("backend") or "rlinf").strip()
    if backend_kind in {"", "rlinf"} and requested in {"", "vla_rl_backend"}:
        return "rlinf_vla"
    if requested:
        return requested
    return "rlinf_vla" if backend_kind in {"", "rlinf"} else "vla_rl_backend"


def deployability_gate(
    contract: Mapping[str, Any],
    *,
    robot_embodiment: str = "",
    observation_schema: str = "",
    action_schema: str = "",
) -> dict[str, Any]:
    checks = [
        _check_required(contract, "modelFamily"),
        _check_required(contract, "checkpointPath"),
        _check_required(contract, "artifactPath"),
    ]
    checks.append(_check_match("robotEmbodiment", contract.get("robotEmbodiment"), robot_embodiment))
    checks.append(_check_match("observationSchema", contract.get("observationSchema"), observation_schema))
    checks.append(_check_match("actionSchema", contract.get("actionSchema"), action_schema))
    blocking = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    return {
        "deployable": not blocking,
        "checks": checks,
        "blockingReasons": [check["message"] for check in blocking],
        "warnings": [check["message"] for check in warnings],
    }


def _load_contract(contract_path: str, contract: Mapping[str, Any] | None) -> dict[str, Any]:
    if contract is not None:
        return dict(contract)
    if not contract_path:
        raise ValueError("contract_path or contract is required")
    path = Path(contract_path).expanduser()
    return json.loads(path.read_text(encoding="utf-8"))


def _load_first_json_path(paths: Any) -> dict[str, Any]:
    if isinstance(paths, str):
        candidates = [paths]
    elif isinstance(paths, list):
        candidates = [str(path) for path in paths]
    else:
        candidates = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    return {}


def _deployability_hints(params: Mapping[str, Any]) -> list[str]:
    hints = []
    if not params.get("robotEmbodiment") and not params.get("robotAdapter"):
        hints.append("robot embodiment is not declared; deployment will require a robot compatibility check")
    if not params.get("observationSchema"):
        hints.append("observation schema is not declared")
    if not params.get("actionSchema"):
        hints.append("action schema is not declared")
    return hints


def _check_required(contract: Mapping[str, Any], key: str) -> dict[str, str]:
    value = str(contract.get(key) or "").strip()
    if value:
        return {"name": key, "status": "pass", "message": f"{key} is declared"}
    return {"name": key, "status": "fail", "message": f"{key} is required for deployment"}


def _check_match(name: str, expected: Any, actual: str) -> dict[str, str]:
    expected_text = str(expected or "").strip()
    actual_text = actual.strip()
    if not expected_text or not actual_text:
        return {"name": name, "status": "warn", "message": f"{name} was not fully checked"}
    if expected_text == actual_text:
        return {"name": name, "status": "pass", "message": f"{name} matches"}
    return {"name": name, "status": "fail", "message": f"{name} mismatch: expected {expected_text}, got {actual_text}"}


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
