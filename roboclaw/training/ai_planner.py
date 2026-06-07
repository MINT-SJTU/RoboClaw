"""AI-assisted training and evaluation planning."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Mapping

from roboclaw.providers.base import LLMProvider
from roboclaw.training.schema import TrainingPlanSpec


_BLOCKED_PARAM_KEYS = {
    "providerToken",
    "autodlToken",
    "autodlApiKey",
    "sshHost",
    "sshPort",
    "sshUser",
    "sshPassword",
    "sshKey",
    "sshKeyPath",
    "apiKey",
    "secretKey",
    "builtinTrainingProfile",
}

_AI_OVERRIDE_KEYS = {
    "workflow",
    "backendKind",
    "modelFamily",
    "policyType",
    "algorithm",
    "benchmark",
    "suite",
    "environmentKind",
    "environmentHint",
    "trainingMode",
    "evalEpisodes",
    "maxSteps",
}

_PLACEHOLDER_VALUE_RE = re.compile(
    r"^(?:云端工作目录|云端项目目录|云端产物目录|<[^>]+>|\$\{[^}]+}|"
    r"tbd|todo|placeholder|unknown|待填写|待确认|占位|your[-_ ]?(?:path|dir|repo|model|dataset))$",
    re.IGNORECASE,
)

_HUMAN_TEXT_PATH_RE = re.compile(r"/root/autodl-tmp/[^\s\"',;，。；、)）\]}】]+")
_HUMAN_ARTIFACT_PATH_RE = re.compile(r"/workspace/outputs[^\s\"',;，。；、)）\]}】]*")


async def generate_ai_training_plan(
    provider: LLMProvider | None,
    spec: TrainingPlanSpec,
    *,
    workflow: str,
    params: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Use the configured LLM provider to turn user intent into a structured plan.

    The LLM never starts compute. It only proposes a plan that is then sent through
    the normal EVO_Train contract path or shown to the user for confirmation.
    """

    if provider is None:
        return {
            "source": "llm_unconfigured",
            "providerModel": "unconfigured",
            "warnings": [
                "未配置 AI 规划器。请在「设置 → 模型提供商」中添加 API Key（支持 OpenAI / Anthropic / 本地 Ollama），保存后自动启用。"
            ],
        }
    model = _provider_model(provider)
    if model == "unconfigured":
        return {
            "source": "llm_unconfigured",
            "providerModel": "unconfigured",
            "warnings": [
                "未配置 AI 规划器。请在「设置 → 模型提供商」中添加 API Key（支持 OpenAI / Anthropic / 本地 Ollama），保存后自动启用。"
            ],
        }

    prompt_payload = {
        "userMessage": spec.message,
        "requestedWorkflow": spec.workflow or workflow,
        "currentParams": dict(params),
        "provider": spec.provider,
        "skuId": spec.sku_id,
        "imageId": spec.image_id,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are Evo Studio's robot training planner. Convert a user's natural "
                "language request into a safe VLA/RL training or evaluation plan. "
                "First understand the user's actual research intent, then decide whether "
                "a launchable plan can be produced. Avoid generic templates: preserve the "
                "specific model, benchmark, dataset, backend, and success criteria the "
                "user asked for. Return only JSON. Do not include markdown. Do not start jobs, do not "
                "ask for provider tokens, SSH credentials, or cloud secrets. Use the "
                "same language as the user for human-facing summary, steps, warnings, "
                "and hints. Prefer a smoke test when data, model, environment, or cost "
                "is uncertain."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a structured plan for this RoboClaw/EVO_Train request.\n"
                "Allowed top-level JSON fields:\n"
                "workflow, params, readyToStart, missingFields, warnings, humanSummary, "
                "intentUnderstanding, planSteps, evaluationPlan, resourceHints, safetyChecks, "
                "clarifyingQuestions.\n"
                "intentUnderstanding is required. It must summarize what the user actually "
                "asked for, with fields like objective, taskType, model, dataset, benchmark, "
                "backend, mode, constraints, successCriteria, confidence, and unknowns. "
                "If confidence is low or unknowns affect correctness/cost, set "
                "readyToStart=false and ask clarifyingQuestions instead of filling a template.\n"
                "Allowed workflows are rlinf_vla and vla_rl_backend. For custom Git "
                "projects, OpenVLA-OFT, public HuggingFace data, or user supplied code "
                "sources, still use workflow=vla_rl_backend and express code/model/data "
                "inside params. Do not invent workflow aliases and do not set "
                "builtinTrainingProfile; EVO_Train will match the backend profile from "
                "the structured sources, model, algorithm, and training mode.\n"
                "If required information is missing, set readyToStart=false and put the "
                "most important follow-up questions in clarifyingQuestions instead of "
                "guessing hidden defaults.\n"
                "When the request is VLA/RL training or evaluation, model the plan like "
                "an RLinf-style contract instead of a vague template name. Put these "
                "sections under params.trainingContract when relevant: sources.dataset, "
                "sources.model, runner, actor.model, rollout, env, algorithm, runtime, "
                "artifacts. Also keep compatibility flat fields such as datasetSource, "
                "modelSource, policyType, modelFamily, algorithm, trainingMode, "
                "benchmark, suite, maxSteps, evalEpisodes, gpuCount, replicas, "
                "requestedGpuTotal, repoUrl, launcherModule, scriptPath, configName, "
                "artifactPath when you know them. Do not output placeholder path strings "
                "such as 云端工作目录, <path>, TBD, or TODO. If a path is unknown, omit the "
                "field and add it to missingFields or clarifyingQuestions. Keep human-facing "
                "planSteps concise and do not include raw internal cloud paths there. "
                "If the user asks for RLinf, LIBERO-Pro, "
                "LIBERO-Plus, or OpenVLA-OFT simulation evaluation through RLinf, prefer "
                "the official RLinf repo https://github.com/RLinf/RLinf.git and express "
                "the request as an rlinf_vla contract; do not switch to a raw openvla-oft "
                "project unless the user explicitly asks to bypass RLinf.\n\n"
                f"Request context:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    attempts = _planner_attempts()
    response = None
    for attempt in range(attempts):
        try:
            response = await asyncio.wait_for(
                provider.chat_with_retry(
                    messages,
                    model=model,
                    max_tokens=1600,
                    temperature=0.1,
                ),
                timeout=_planner_timeout_seconds(),
            )
            break
        except TimeoutError:
            if attempt + 1 >= attempts:
                break
            continue
        except Exception as exc:
            return {
                "source": "llm_error",
                "providerModel": model,
                "warnings": [f"AI planner failed: {exc}"],
            }
    if response is None:
        return {
            "source": "llm_timeout",
            "providerModel": model,
            "warnings": [
                f"AI planner timed out after {attempts} attempt(s); no launchable AI plan was produced. "
                "This usually means the configured API gateway did not return in time."
            ],
        }
    try:
        finish_reason = response.finish_reason
    except Exception as exc:
        return {
            "source": "llm_error",
            "providerModel": model,
            "warnings": [f"AI planner failed: {exc}"],
        }
    if finish_reason == "error":
        return {
            "source": "llm_error",
            "providerModel": model,
            "warnings": [response.content or "AI planner call failed."],
        }

    parsed = _extract_json_object(response.content or "")
    if not isinstance(parsed, dict):
        return {
            "source": "llm_parse_error",
            "providerModel": model,
            "warnings": ["AI planner did not return a JSON object."],
            "rawContent": (response.content or "")[:1200],
        }

    proposed_params = parsed.get("params")
    if not isinstance(proposed_params, dict):
        proposed_params = {}
    parsed["params"] = _sanitize_params(proposed_params)
    parsed["workflow"] = str(parsed.get("workflow") or workflow or "rlinf_vla")
    parsed["source"] = "llm"
    parsed["providerModel"] = model
    parsed["missingFields"] = _string_list(parsed.get("missingFields"))
    parsed["warnings"] = _string_list(parsed.get("warnings"))
    parsed["planSteps"] = _string_list(parsed.get("planSteps"))
    parsed["evaluationPlan"] = _string_list(parsed.get("evaluationPlan"))
    parsed["resourceHints"] = _string_list(parsed.get("resourceHints"))
    parsed["safetyChecks"] = _string_list(parsed.get("safetyChecks"))
    parsed["clarifyingQuestions"] = _string_list(parsed.get("clarifyingQuestions"))
    parsed["intentUnderstanding"] = _sanitize_mapping(parsed.get("intentUnderstanding"))
    parsed["humanSummary"] = _clean_human_plan_text(str(parsed.get("humanSummary") or ""))
    parsed["readyToStart"] = bool(parsed.get("readyToStart"))
    return parsed


def merge_ai_plan(
    *,
    ai_plan: Mapping[str, Any] | None,
    workflow: str,
    params: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Merge AI output into deterministic planner params with local params winning."""

    merged = dict(params)
    if not ai_plan:
        return workflow, merged
    ai_params = ai_plan.get("params")
    if isinstance(ai_params, Mapping):
        for key, value in ai_params.items():
            if key in _BLOCKED_PARAM_KEYS:
                continue
            if key in _AI_OVERRIDE_KEYS or key not in merged or merged.get(key) in (None, "", [], {}):
                merged[str(key)] = value
    proposed_workflow = str(ai_plan.get("workflow") or "").strip()
    if _looks_like_vla_backend_intent(proposed_workflow, merged):
        proposed_workflow = "vla_rl_backend"
    return proposed_workflow or workflow, merged


def _looks_like_vla_backend_intent(workflow: str, params: Mapping[str, Any]) -> bool:
    if workflow in {"rlinf_vla", "vla_rl_backend", ""}:
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


def _source_uri(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("uri", "url", "repo", "repoId", "repoUrl", "gitUrl", "path"):
            item = value.get(key)
            if item not in (None, ""):
                return str(item)
        return ""
    return str(value or "")


def _provider_model(provider: LLMProvider) -> str:
    try:
        return provider.get_default_model()
    except Exception:
        return "unknown"


def _planner_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("EVO_STUDIO_AI_PLANNER_TIMEOUT_SECONDS", "45")))
    except ValueError:
        return 45.0


def _planner_attempts() -> int:
    try:
        return max(1, min(3, int(os.environ.get("EVO_STUDIO_AI_PLANNER_ATTEMPTS", "2"))))
    except ValueError:
        return 2


def _sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if str(key) in _BLOCKED_PARAM_KEYS:
            continue
        sanitized = _sanitize_plan_value(value)
        if sanitized in (None, "", [], {}):
            continue
        result[str(key)] = sanitized
    return result


def _sanitize_plan_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text or _PLACEHOLDER_VALUE_RE.fullmatch(text):
            return None
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        cleaned = [_sanitize_plan_value(item) for item in value]
        return [item for item in cleaned if item not in (None, "", [], {})]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).strip()
            if not key_text or key_text in _BLOCKED_PARAM_KEYS:
                continue
            sanitized = _sanitize_plan_value(item)
            if sanitized not in (None, "", [], {}):
                result[key_text] = sanitized
        return result
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        item = _clean_human_plan_text(value)
        return [item] if item else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            formatted = _clean_human_plan_text(_format_list_item(item))
            if formatted:
                result.append(formatted)
        return result
    item = _clean_human_plan_text(_format_list_item(value))
    return [item] if item else []


def _clean_human_plan_text(value: str) -> str:
    text = value.strip()
    if not text or _PLACEHOLDER_VALUE_RE.fullmatch(text):
        return ""
    text = _HUMAN_TEXT_PATH_RE.sub("云端项目目录", text)
    text = _HUMAN_ARTIFACT_PATH_RE.sub("产物目录", text)
    return text.replace("云端工作目录", "云端项目目录").replace("云端产物目录", "产物目录")


def _format_list_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        field = value.get("field")
        reason = value.get("reason")
        if field and reason:
            return f"{field}: {reason}"
        name = value.get("name") or value.get("title")
        details = value.get("details")
        step = value.get("step")
        actions = value.get("actions")
        if name and isinstance(actions, list):
            prefix = f"{step}. " if step else ""
            action_text = "；".join(_format_list_item(item) for item in actions if _format_list_item(item))
            return f"{prefix}{name}: {action_text}" if action_text else f"{prefix}{name}"
        if name and details:
            prefix = f"{step}. " if step else ""
            if isinstance(details, list):
                details = "；".join(_format_list_item(item) for item in details if _format_list_item(item))
            elif isinstance(details, Mapping):
                details = json.dumps(dict(details), ensure_ascii=False)
            return f"{prefix}{name}: {details}"
        pass_criteria = value.get("passCriteria")
        if name and pass_criteria:
            return f"{name}: {pass_criteria}"
        item_type = value.get("type")
        metric = value.get("primaryMetric")
        episodes = value.get("episodes")
        if item_type and metric:
            episode_text = f", {episodes} episode(s)" if episodes is not None else ""
            return f"{item_type}: primary metric {metric}{episode_text}"
        checks = value.get("checks")
        gate = value.get("successGate")
        if isinstance(checks, list) and checks:
            check_text = "；".join(_format_list_item(item) for item in checks if _format_list_item(item))
            gate_text = f"；通过门槛：{gate}" if gate else ""
            prefix = f"{item_type}: " if item_type else ""
            return f"{prefix}{check_text}{gate_text}"
        smoke_criteria = value.get("smokeCriteria")
        promote_criteria = value.get("promoteToFullRunWhen")
        if isinstance(smoke_criteria, list) and smoke_criteria:
            parts = [f"smoke 通过标准：{'；'.join(_format_list_item(item) for item in smoke_criteria if _format_list_item(item))}"]
            if isinstance(promote_criteria, list) and promote_criteria:
                parts.append(f"正式训练条件：{'；'.join(_format_list_item(item) for item in promote_criteria if _format_list_item(item))}")
            suggestion = value.get("fullRunSuggestion")
            if isinstance(suggestion, Mapping):
                suggestion_text = _format_list_item(suggestion)
                if suggestion_text:
                    parts.append(f"正式训练建议：{suggestion_text}")
            return "；".join(parts)
        goal = value.get("goal")
        protocol = value.get("protocol")
        if goal:
            parts = [f"目标：{goal}"]
            if isinstance(protocol, Mapping):
                protocol_episodes = protocol.get("episodes")
                protocol_metrics = protocol.get("metrics")
                protocol_criteria = protocol.get("passCriteria")
                if protocol_episodes is not None:
                    parts.append(f"回合数：{protocol_episodes}")
                if isinstance(protocol_metrics, list) and protocol_metrics:
                    parts.append(f"指标：{'、'.join(_format_list_item(item) for item in protocol_metrics if _format_list_item(item))}")
                if isinstance(protocol_criteria, list) and protocol_criteria:
                    parts.append(f"通过标准：{'；'.join(_format_list_item(item) for item in protocol_criteria if _format_list_item(item))}")
            failure_logging = value.get("failureLogging")
            if isinstance(failure_logging, Mapping) and failure_logging.get("enabled"):
                include = failure_logging.get("include")
                if isinstance(include, list) and include:
                    parts.append(f"失败日志：{'、'.join(_format_list_item(item) for item in include if _format_list_item(item))}")
            return "；".join(parts)
        if metric:
            secondary = value.get("secondaryMetrics")
            secondary_text = ""
            if isinstance(secondary, list) and secondary:
                secondary_text = f"，辅助指标：{'、'.join(_format_list_item(item) for item in secondary if _format_list_item(item))}"
            criteria = value.get("acceptanceCriteria")
            criteria_text = ""
            if isinstance(criteria, list) and criteria:
                criteria_text = f"，通过标准：{'；'.join(_format_list_item(item) for item in criteria if _format_list_item(item))}"
            return f"主指标：{metric}{secondary_text}{criteria_text}"
        compute = value.get("compute")
        if compute:
            time_estimate = value.get("timeEstimate")
            cost_control = value.get("costControl")
            parts = [f"算力：{compute}"]
            if time_estimate:
                parts.append(f"预计耗时：{time_estimate}")
            if isinstance(cost_control, list) and cost_control:
                parts.append(f"成本控制：{'；'.join(_format_list_item(item) for item in cost_control if _format_list_item(item))}")
            return "；".join(parts)
        recommended = value.get("recommended")
        if isinstance(recommended, list) and recommended:
            priority = value.get("priority")
            priority_text = f"优先级：{priority}；" if priority else ""
            return f"{priority_text}{'；'.join(_format_list_item(item) for item in recommended if _format_list_item(item))}"
        cost_control = value.get("costControl") or value.get("costStrategy")
        runtime_hints = value.get("runtimeHints") or value.get("runtimeStrategy")
        if isinstance(cost_control, list) or isinstance(runtime_hints, list):
            parts = []
            if isinstance(cost_control, list) and cost_control:
                parts.append(f"成本控制：{'；'.join(_format_list_item(item) for item in cost_control if _format_list_item(item))}")
            if isinstance(runtime_hints, list) and runtime_hints:
                parts.append(f"运行提示：{'；'.join(_format_list_item(item) for item in runtime_hints if _format_list_item(item))}")
            return "；".join(parts)
        return json.dumps(dict(value), ensure_ascii=False)
    return str(value).strip()


def _sanitize_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(item, str):
            text = _clean_human_plan_text(item)
            if text:
                result[key_text] = text
        elif isinstance(item, (int, float, bool)) or item is None:
            result[key_text] = item
        elif isinstance(item, list):
            entries = [_clean_human_plan_text(_format_list_item(entry)) for entry in item]
            result[key_text] = [entry for entry in entries if entry]
        elif isinstance(item, Mapping):
            nested = _sanitize_mapping(item)
            if nested:
                result[key_text] = nested
    return result


def _extract_json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
