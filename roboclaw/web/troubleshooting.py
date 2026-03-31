"""Troubleshooting decision map and fault snapshot generator.

Provides user-facing troubleshooting steps for each hardware fault type,
plus a snapshot generator for tech support reports.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from roboclaw.embodied.hardware_monitor import HardwareFault


@dataclass
class TroubleshootEntry:
    title: str
    description: str
    steps: list[str]
    can_recheck: bool


TROUBLESHOOT_MAP: dict[str, TroubleshootEntry] = {
    "arm_disconnected": TroubleshootEntry(
        title="机械臂断开连接",
        description="检测到机械臂 USB 连接中断",
        steps=[
            "检查 USB 线缆是否松动，重新插紧",
            "等待 10 秒",
            "点击下方「重新检测」",
            "如仍未恢复：拔掉 USB 线，等待 5 秒，重新插入",
            "再次点击「重新检测」",
            "如多次尝试仍失败，点击「联系技术支持」",
        ],
        can_recheck=True,
    ),
    "arm_timeout": TroubleshootEntry(
        title="机械臂通信超时",
        description="机械臂已连接但无法正常通信",
        steps=[
            "关闭机械臂电源，等待 5 秒后重新开启",
            "重新插拔 USB 线缆",
            "点击「重新检测」",
            "如反复出现，联系技术支持",
        ],
        can_recheck=True,
    ),
    "arm_not_calibrated": TroubleshootEntry(
        title="机械臂未校准",
        description="机械臂需要校准后才能采集数据",
        steps=[
            "请联系部署人员执行校准操作",
        ],
        can_recheck=False,
    ),
    "camera_disconnected": TroubleshootEntry(
        title="摄像头断开连接",
        description="检测到摄像头 USB 连接中断",
        steps=[
            "检查摄像头 USB 线缆是否松动",
            "尝试更换 USB 端口",
            "点击「重新检测」",
            "如仍未恢复，联系技术支持",
        ],
        can_recheck=True,
    ),
    "camera_frame_drop": TroubleshootEntry(
        title="摄像头画面丢失",
        description="摄像头已连接但无法获取画面",
        steps=[
            "检查摄像头镜头是否被遮挡",
            "重新插拔 USB 线缆",
            "点击「重新检测」",
        ],
        can_recheck=True,
    ),
    "record_crashed": TroubleshootEntry(
        title="采集进程异常退出",
        description="数据采集进程意外终止",
        steps=[
            "点击「开始新采集」重新开始",
            "如反复崩溃，点击「联系技术支持」生成故障报告",
        ],
        can_recheck=False,
    ),
}


def get_troubleshoot_map_json() -> dict[str, dict[str, Any]]:
    """Return the troubleshooting map as a JSON-serializable dict."""
    return {key: asdict(entry) for key, entry in TROUBLESHOOT_MAP.items()}


def generate_fault_snapshot(
    setup: dict[str, Any],
    faults: list[HardwareFault],
    stderr_tail: str,
) -> dict[str, Any]:
    """Generate a fault snapshot for tech support.

    Includes: setup.json content, active faults, last 50 lines of stderr, timestamp.
    """
    return {
        "timestamp": time.time(),
        "setup": setup,
        "faults": [f.to_dict() for f in faults],
        "stderr_tail": stderr_tail,
    }
