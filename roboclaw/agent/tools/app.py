"""App-awareness tool for the in-app RoboClaw AI."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.bus.events import OutboundMessage

SendCallback = Callable[[OutboundMessage], Awaitable[None]]


def _action(
    action_id: str,
    label: str,
    tool: str,
    action: str,
    requires: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "label": label,
        "tool": tool,
        "action": action,
        "requires": requires or [],
    }


APP_PAGES: list[dict[str, Any]] = [
    {
        "id": "control",
        "route": "/control",
        "name": "控制中心",
        "description": "Live robot control for teleoperation, recording, replay, inference, and hardware readiness.",
        "state_sources": ["/api/hardware/status", "/api/session/status"],
        "actions": [
            _action("hardware.get_status", "读取硬件状态", "app", "describe_page"),
            _action("teleop.teleoperate", "启动遥操作", "teleop", "teleoperate"),
            _action("record.record", "采集数据集", "record", "record"),
            _action("replay.replay", "回放数据集 episode", "replay", "replay"),
            _action("infer.run_policy", "运行策略推理", "infer", "run_policy"),
        ],
    },
    {
        "id": "recovery",
        "route": "/recovery",
        "name": "恢复中心",
        "description": "Hardware fault recovery, recheck guidance, and dashboard restart controls.",
        "state_sources": ["/api/recovery/faults", "/api/recovery/guides"],
        "actions": [
            _action("recovery.recheck", "重新检测硬件", "app", "describe_page"),
            _action("doctor.check", "检查 embodied 环境", "doctor", "check"),
        ],
    },
    {
        "id": "training",
        "route": "/training",
        "name": "训练中心",
        "description": "Policy training dashboard with dataset/policy discovery, job start/stop, status, and loss curves.",
        "state_sources": ["/api/train/current", "/api/train/datasets", "/api/train/policies"],
        "actions": [
            _action("train.list_datasets", "列出可训练数据集", "train", "list_datasets"),
            _action("train.list_policies", "列出策略/模型", "train", "list_policies"),
            _action("train.train", "启动训练", "train", "train"),
            _action("train.job_status", "查看训练任务状态", "train", "job_status"),
        ],
    },
    {
        "id": "curation_datasets",
        "route": "/curation/datasets",
        "name": "数据集读取",
        "description": "Dataset discovery/import/preparation and episode preview for curation workflows.",
        "state_sources": ["/api/curation/datasets", "/api/explorer/summary"],
        "actions": [
            _action("pipeline.get_current_page_data", "读取当前数据集读取页面数据", "pipeline", "get_current_page_data"),
            _action("pipeline.list_datasets", "列出 Pipeline 数据集", "pipeline", "list_datasets"),
            _action("pipeline.get_explorer_summary", "读取当前数据集摘要", "pipeline", "get_explorer_summary", ["dataset"]),
            _action("pipeline.get_explorer_details", "读取当前数据集结构详情", "pipeline", "get_explorer_details", ["dataset"]),
            _action("pipeline.get_explorer_episodes", "读取 episode 列表页", "pipeline", "get_explorer_episodes", ["dataset"]),
            _action("pipeline.prepare_remote_dataset", "准备远程数据集到 Pipeline", "pipeline", "prepare_remote_dataset", ["dataset"]),
            _action("pipeline.load_remote_dataset", "加载远程数据集并同步前端", "pipeline", "load_remote_dataset", ["dataset"]),
            _action("pipeline.get_state", "读取当前数据集 Pipeline 状态", "pipeline", "get_state", ["dataset"]),
        ],
    },
    {
        "id": "curation_quality",
        "route": "/curation/quality",
        "name": "质量验证",
        "description": "Dataset-aware quality validation with metadata, timing, action, visual, depth, and trajectory checks.",
        "state_sources": [
            "/api/curation/quality-defaults",
            "/api/curation/quality-results",
            "/api/curation/state",
        ],
        "actions": [
            _action("pipeline.get_current_page_data", "读取当前质量验证页面数据", "pipeline", "get_current_page_data"),
            _action("pipeline.get_quality_defaults", "读取数据集默认质量验证参数", "pipeline", "get_quality_defaults", ["dataset"]),
            _action("pipeline.get_quality_results", "读取质量验证结果", "pipeline", "get_quality_results", ["dataset"]),
            _action("pipeline.run_quality", "运行质量验证", "pipeline", "run_quality", ["dataset"]),
            _action("pipeline.pause_quality", "暂停质量验证", "pipeline", "pause_quality", ["dataset"]),
            _action("pipeline.resume_quality", "恢复质量验证", "pipeline", "resume_quality", ["dataset"]),
        ],
    },
    {
        "id": "curation_text_alignment",
        "route": "/curation/text-alignment",
        "name": "文本对齐",
        "description": "Prototype discovery, annotation workspace, and semantic propagation for episode text alignment.",
        "state_sources": [
            "/api/curation/prototype-results",
            "/api/curation/annotation-workspace",
            "/api/curation/propagation-results",
        ],
        "actions": [
            _action("pipeline.get_current_page_data", "读取当前文本对齐页面数据", "pipeline", "get_current_page_data"),
            _action("pipeline.get_state", "读取 Pipeline 状态", "pipeline", "get_state", ["dataset"]),
            _action("pipeline.get_alignment_overview", "读取对齐总览与 DTW 结果", "pipeline", "get_alignment_overview", ["dataset"]),
            _action("pipeline.get_prototype_results", "读取原型发现结果", "pipeline", "get_prototype_results", ["dataset"]),
            _action("pipeline.get_propagation_results", "读取标注传播结果", "pipeline", "get_propagation_results", ["dataset"]),
            _action("pipeline.get_episode_workspace", "读取标注工作台 episode 数据", "pipeline", "get_episode_workspace", ["dataset", "episode_index"]),
            _action("pipeline.run_prototype", "发现原型片段", "pipeline", "run_prototype", ["dataset"]),
            _action("pipeline.run_propagation", "语义传播标注", "pipeline", "run_propagation", ["dataset", "source_episode_index"]),
        ],
    },
    {
        "id": "curation_data_overview",
        "route": "/curation/data-overview",
        "name": "数据总览",
        "description": "Quality, DTW, semantic alignment, episode video, and joint-angle overview for checked episodes.",
        "state_sources": [
            "/api/curation/alignment-overview",
            "/api/curation/quality-results",
            "/api/curation/annotation-workspace",
            "/api/curation/propagation-results",
            "/api/curation/prototype-results",
        ],
        "actions": [
            _action("pipeline.get_current_page_data", "读取当前数据总览页面数据", "pipeline", "get_current_page_data"),
            _action("pipeline.get_data_overview", "读取数据总览完整状态", "pipeline", "get_data_overview", ["dataset"]),
            _action("pipeline.get_alignment_overview", "读取数据总览结果", "pipeline", "get_alignment_overview", ["dataset"]),
            _action("pipeline.get_prototype_results", "读取原型发现结果", "pipeline", "get_prototype_results", ["dataset"]),
            _action("pipeline.get_propagation_results", "读取标注传播结果", "pipeline", "get_propagation_results", ["dataset"]),
            _action("pipeline.get_quality_results", "读取质量结果", "pipeline", "get_quality_results", ["dataset"]),
            _action("pipeline.get_episode_workspace", "读取 episode 视频与关节角度", "pipeline", "get_episode_workspace", ["dataset", "episode_index"]),
            _action("pipeline.get_state", "读取 Pipeline 状态", "pipeline", "get_state", ["dataset"]),
        ],
    },
    {
        "id": "settings",
        "route": "/settings",
        "name": "设置总览",
        "description": "Settings overview for hardware, provider, and HuggingFace integration.",
        "state_sources": ["/api/devices", "/api/system/provider-status"],
        "actions": [
            _action("app.list_pages", "查看设置相关页面", "app", "list_pages"),
        ],
    },
    {
        "id": "settings_hardware",
        "route": "/settings/hardware",
        "name": "硬件设置",
        "description": "Device catalog, hardware setup wizard, calibration, and manifest management.",
        "state_sources": ["/api/setup/session", "/api/devices", "/api/hardware/status"],
        "actions": [
            _action("setup.identify", "识别并配置硬件", "setup", "identify"),
            _action("setup.preview_cameras", "预览摄像头", "setup", "preview_cameras"),
            _action("setup.modify", "修改设备绑定", "setup", "modify"),
            _action("calibration.calibrate", "校准机械臂", "calibration", "calibrate"),
            _action("doctor.check", "检查硬件环境", "doctor", "check"),
        ],
    },
    {
        "id": "settings_provider",
        "route": "/settings/provider",
        "name": "AI Provider 设置",
        "description": "LLM provider configuration and connection testing.",
        "state_sources": ["/api/system/provider-status", "/api/system/provider-config"],
        "actions": [
            _action("app.describe_page", "解释 Provider 设置页", "app", "describe_page"),
        ],
    },
    {
        "id": "settings_hub",
        "route": "/settings/hub",
        "name": "HuggingFace 设置",
        "description": "Hub authentication and dataset/policy transfer settings.",
        "state_sources": [
            "/api/hub/datasets/push",
            "/api/hub/datasets/pull",
            "/api/hub/policies/push",
            "/api/hub/policies/pull",
        ],
        "actions": [
            _action("hub.pull_dataset", "拉取 Hub 数据集", "hub", "pull_dataset"),
            _action("hub.push_dataset", "推送数据集到 Hub", "hub", "push_dataset"),
            _action("hub.pull_policy", "拉取 Hub 策略", "hub", "pull_policy"),
            _action("hub.push_policy", "推送策略到 Hub", "hub", "push_policy"),
        ],
    },
    {
        "id": "logs",
        "route": "/logs",
        "name": "日志",
        "description": "Runtime logs and diagnostics view.",
        "state_sources": ["/api/session/logs"],
        "actions": [
            _action("doctor.check", "检查运行环境", "doctor", "check"),
        ],
    },
]


class AppTool(Tool):
    """Let RoboClaw AI understand the web app surface and current page context."""

    def __init__(self, send_callback: SendCallback | None = None):
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""
        self._context_by_session: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "app"

    @property
    def description(self) -> str:
        return (
            "Discover RoboClaw web app pages, inspect the user's current page context, "
            "list page-level capabilities, and request navigation in the Web UI."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_pages",
                        "get_current_context",
                        "describe_page",
                        "list_page_actions",
                        "resolve_route",
                        "navigate",
                    ],
                    "description": "App-awareness operation to perform.",
                },
                "page": {
                    "type": "string",
                    "description": "Page id or route. Defaults to the current page when available.",
                },
                "route": {
                    "type": "string",
                    "description": "Route to resolve or navigate to, such as /curation/quality.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def set_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        app_context = _extract_app_context(metadata or {})
        if app_context:
            self._context_by_session[self._session_key(channel, chat_id)] = app_context

    async def execute(self, action: str, page: str = "", route: str = "") -> str:
        try:
            if action == "list_pages":
                return _json({"pages": [_page_summary(item) for item in APP_PAGES]})

            if action == "get_current_context":
                context = self._current_context()
                resolved = _resolve_page(page or route or str(context.get("route") or ""))
                return _json({
                    "context": context,
                    "page": _page_payload(resolved) if resolved else None,
                    "context_available": bool(context),
                })

            if action == "describe_page":
                resolved = self._resolve_requested_page(page, route)
                if not resolved:
                    return _json({"error": "page or route is required"})
                return _json({"page": _page_payload(resolved)})

            if action == "list_page_actions":
                resolved = self._resolve_requested_page(page, route)
                if not resolved:
                    return _json({"error": "page or route is required"})
                return _json({
                    "page": _page_summary(resolved),
                    "actions": deepcopy(resolved.get("actions", [])),
                })

            if action == "resolve_route":
                resolved = self._resolve_requested_page(page, route)
                if not resolved:
                    return _json({"error": "No matching page found"})
                return _json({"page": _page_payload(resolved)})

            if action == "navigate":
                resolved = self._resolve_requested_page(page, route)
                target_route = str((resolved or {}).get("route") or route)
                if not _is_safe_route(target_route):
                    return _json({"error": f"Unsafe or unknown route: {target_route}"})
                event_sent = await self._send_navigation_event(target_route, resolved)
                return _json({
                    "status": "navigation_requested" if event_sent else "navigation_resolved",
                    "route": target_route,
                    "page": _page_summary(resolved) if resolved else None,
                    "event_sent": event_sent,
                })

            return _json({"error": f"Unknown app action: {action}"})
        except Exception as exc:
            return _json({"error": str(exc)})

    def _current_context(self) -> dict[str, Any]:
        return deepcopy(self._context_by_session.get(self._session_key(self._channel, self._chat_id), {}))

    def _resolve_requested_page(self, page: str, route: str) -> dict[str, Any] | None:
        if page or route:
            return _resolve_page(page or route)
        context = self._current_context()
        return _resolve_page(str(context.get("route") or context.get("pathname") or ""))

    async def _send_navigation_event(
        self,
        route: str,
        page: dict[str, Any] | None,
    ) -> bool:
        if not self._send_callback or self._channel != "web" or not self._chat_id:
            return False
        await self._send_callback(
            OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content="",
                metadata={
                    "app_event": {
                        "type": "app.navigate",
                        "route": route,
                        "page": _page_summary(page) if page else None,
                    }
                },
            )
        )
        return True

    @staticmethod
    def _session_key(channel: str, chat_id: str) -> str:
        return f"{channel}:{chat_id}"


def _extract_app_context(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("app_context") or metadata.get("appContext") or metadata.get("app")
    if not isinstance(raw, dict):
        return {}
    context = deepcopy(raw)
    route = str(context.get("route") or context.get("pathname") or "").strip()
    if route:
        context["route"] = _normalize_route(route)
    return context


def _resolve_page(value: str) -> dict[str, Any] | None:
    needle = _normalize_route(value)
    if not needle:
        return None
    for page in sorted(APP_PAGES, key=lambda item: len(item["route"]), reverse=True):
        page_id = str(page["id"])
        page_route = str(page["route"])
        if needle == page_id or needle == page_route:
            return page
        if needle.startswith(page_route.rstrip("/") + "/"):
            return page
    return None


def _normalize_route(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    if "://" in clean:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(clean)
            clean = parsed.path or "/"
        except Exception:
            return ""
    if "?" in clean:
        clean = clean.split("?", 1)[0]
    if "#" in clean:
        clean = clean.split("#", 1)[0]
    if not clean.startswith("/"):
        return clean
    return clean.rstrip("/") or "/"


def _is_safe_route(route: str) -> bool:
    return bool(route) and route.startswith("/") and not route.startswith("//")


def _page_summary(page: dict[str, Any] | None) -> dict[str, Any]:
    if not page:
        return {}
    return {
        "id": page["id"],
        "route": page["route"],
        "name": page["name"],
        "description": page["description"],
    }


def _page_payload(page: dict[str, Any]) -> dict[str, Any]:
    payload = _page_summary(page)
    payload["state_sources"] = list(page.get("state_sources", []))
    payload["actions"] = deepcopy(page.get("actions", []))
    return payload


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
