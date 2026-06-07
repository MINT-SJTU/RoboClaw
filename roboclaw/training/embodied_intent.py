"""Natural-language routing for embodied AI work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


EXPLAIN_PREFIXES = (
    "为什么",
    "为啥",
    "什么意思",
    "解释",
    "说明",
    "什么是",
)

ACTION_TOKENS = (
    "复现",
    "训练",
    "评测",
    "跑",
    "运行",
    "启动",
    "配置",
    "调测",
    "采集",
    "上传",
    "同步",
    "部署",
    "bringup",
    "launch",
    "smoke",
    "eval",
    "evaluate",
    "train",
    "run",
    "collect",
    "upload",
)

VLA_TOKENS = (
    "vla",
    "openvla",
    "openvla-oft",
    "starvla",
    "pi0",
    "pi0.5",
    "gr00t",
    "smolvla",
    "lerobot",
    "xlerobot",
    "rlinf",
    "libero",
    "maniskill",
)

DATA_TOKENS = (
    "数据集",
    "dataset",
    "采集",
    "teleop",
    "上传",
    "push",
    "lerobot",
    "episode",
)

ROS2_TOKENS = (
    "ros2",
    "ros 2",
    "nav2",
    "tf2",
    "rviz",
    "rosbag",
    "launch",
    "bringup",
    "topic",
    "node",
)

SLAM_TOKENS = (
    "kiss-icp",
    "kiss_icp",
    "icp",
    "slam",
    "lidar",
    "激光雷达",
    "odometry",
    "里程计",
    "定位",
    "建图",
)

FRAMEWORK_TOKENS = {
    "xlerobot": ("xlerobot", "xle robot"),
    "starvla": ("starvla", "star vla"),
    "lerobot": ("lerobot",),
    "rlinf": ("rlinf",),
    "ros2": ("ros2", "ros 2"),
    "kiss_icp": ("kiss-icp", "kiss_icp"),
}


@dataclass(frozen=True)
class EmbodiedIntent:
    """Structured task route inferred from a user utterance."""

    route: str
    domains: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    workflow: str = ""
    action: str = ""
    explain_only: bool = False
    should_delegate: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "domains": list(self.domains),
            "frameworks": list(self.frameworks),
            "workflow": self.workflow,
            "action": self.action,
            "explainOnly": self.explain_only,
            "shouldDelegate": self.should_delegate,
            "params": dict(self.params),
        }


def classify_embodied_intent(content: str, metadata: Mapping[str, Any] | None = None) -> EmbodiedIntent:
    text = (content or "").strip()
    lowered = text.lower()
    explain_only = lowered.startswith(EXPLAIN_PREFIXES)
    action = _first_action(lowered)
    domains = _domains(lowered)
    frameworks = _frameworks(lowered)
    route = _route(domains, action=action, explain_only=explain_only)
    params = _params_for_route(route, lowered, frameworks)
    return EmbodiedIntent(
        route=route,
        domains=tuple(domains),
        frameworks=tuple(frameworks),
        workflow=str(params.get("workflow") or ""),
        action=action,
        explain_only=explain_only,
        should_delegate=bool(route != "none" and not explain_only and action),
        params=params,
    )


def _first_action(lowered: str) -> str:
    for token in ACTION_TOKENS:
        if token in lowered:
            return token
    return ""


def _domains(lowered: str) -> list[str]:
    domains: list[str] = []
    if _contains_any(lowered, VLA_TOKENS):
        domains.append("vla")
    if _contains_any(lowered, DATA_TOKENS):
        domains.append("data")
    if _contains_any(lowered, ROS2_TOKENS):
        domains.append("ros2")
    if _contains_any(lowered, SLAM_TOKENS):
        domains.append("slam")
    return domains


def _frameworks(lowered: str) -> list[str]:
    result: list[str] = []
    for name, tokens in FRAMEWORK_TOKENS.items():
        if _contains_any(lowered, tokens):
            result.append(name)
    return result


def _route(domains: list[str], *, action: str, explain_only: bool) -> str:
    if explain_only or not action:
        return "none"
    if "vla" in domains:
        return "cloud_training"
    if "data" in domains:
        return "dataset_pipeline"
    if "ros2" in domains or "slam" in domains:
        return "robotics_runtime"
    return "none"


def _params_for_route(route: str, lowered: str, frameworks: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if route == "cloud_training":
        params["workflow"] = "vla_rl_backend"
        if "starvla" in frameworks:
            params.setdefault("modelFamily", "starvla")
            params.setdefault("builtinTrainingProfile", "roboclaw_rlinf_backend")
        if "xlerobot" in frameworks:
            params.setdefault("robotAdapter", "xlerobot")
            params.setdefault("robotEmbodiment", "xlerobot")
        if "lerobot" in frameworks:
            params.setdefault("datasetFormat", "lerobot")
    elif route == "dataset_pipeline":
        params["pipeline"] = "dataset_push"
    elif route == "robotics_runtime":
        params["runtimeKind"] = "robotics_runtime"
        if "kiss_icp" in frameworks or _contains_any(lowered, SLAM_TOKENS):
            params["capability"] = "slam_odometry"
            params["recommendedPackage"] = "kiss-icp"
        if "ros2" in frameworks or _contains_any(lowered, ROS2_TOKENS):
            params["middleware"] = "ros2"
    return params


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)
