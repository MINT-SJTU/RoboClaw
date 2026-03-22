"""Helper utilities extracted from onboarding controller."""

from __future__ import annotations

import os
import re
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboclaw.config.paths import resolve_serial_by_id_path
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import get_ros2_profile
from roboclaw.embodied.localization import localize_text
from roboclaw.embodied.onboarding.model import SetupOnboardingState
from roboclaw.embodied.onboarding.ros2_install import select_ros2_recipe


def select_serial_device_by_id(output: str) -> str | None:
    for line in output.splitlines():
        candidate = line.strip()
        if candidate.startswith("/dev/serial/by-id/"):
            if "->" in candidate:
                candidate = candidate.split("->", 1)[0].strip()
            return candidate
    return None


def normalize_serial_device_by_id(device_path: str) -> str | None:
    candidate = device_path.strip()
    if not candidate:
        return None
    serial_by_id = resolve_serial_by_id_path(candidate)
    if serial_by_id is None:
        return None
    return str(serial_by_id)


def clear_serial_probe_facts(facts: dict[str, Any]) -> None:
    for key in ("serial_device_by_id", "serial_device_unstable", "serial_device_unresponsive", "serial_probe_error"):
        facts.pop(key, None)


def set_serial_device_by_id(facts: dict[str, Any], serial_by_id: str) -> None:
    clear_serial_probe_facts(facts)
    facts["serial_device_by_id"] = serial_by_id


def set_unstable_serial_device(facts: dict[str, Any]) -> None:
    clear_serial_probe_facts(facts)
    facts["serial_device_unstable"] = True


def set_unresponsive_serial_device(facts: dict[str, Any], detail: str) -> None:
    clear_serial_probe_facts(facts)
    facts["serial_device_unresponsive"] = True
    facts["serial_probe_error"] = detail


def mount_frame(mount: str) -> str:
    if mount == "wrist":
        return "tool0"
    return "world"


def sensor_topic(sensor: dict[str, Any]) -> str:
    if sensor["mount"] == "wrist":
        return "/wrist_camera/image_raw"
    if sensor["mount"] == "overhead":
        return "/overhead_camera/image_raw"
    return f"/{sensor['attachment_id']}/image_raw"


def extend_unique(items: list[str], value: str) -> list[str]:
    if value not in items:
        items.append(value)
    return items


def component_summary(state: SetupOnboardingState) -> str:
    robots = ", ".join(item["robot_id"] for item in state.robot_attachments) or "no robot yet"
    sensors = ", ".join(f"{item['sensor_id']}@{item['mount']}" for item in state.sensor_attachments) or "no sensor yet"
    return f"robots=[{robots}] sensors=[{sensors}]"


def profile_id(state: SetupOnboardingState) -> str | None:
    profile = primary_profile(state)
    return profile.id if profile is not None else None


def primary_profile(state: SetupOnboardingState) -> Any:
    if not state.robot_attachments:
        return None
    primary_robot = state.robot_attachments[0]["robot_id"]
    return get_ros2_profile(primary_robot)


def resolved_ros2_distro(state: SetupOnboardingState) -> str | None:
    facts = state.detected_facts
    distro = str(facts.get("ros2_distro") or "").strip()
    if distro:
        return distro
    installed = facts.get("ros2_installed_distros")
    if isinstance(installed, list) and installed:
        value = str(installed[0]).strip()
        if value:
            return value
    recipe = select_ros2_recipe(facts)
    if recipe is not None:
        return recipe.distro
    for distro in ("jazzy", "humble", "iron", "rolling", "foxy"):
        if Path(f"/opt/ros/{distro}/setup.bash").exists() or Path(f"/opt/ros/{distro}/setup.zsh").exists():
            return distro
    return None


def launch_command(state: SetupOnboardingState) -> str | None:
    facts = state.detected_facts
    if facts.get("simulation_requested") is True:
        model_path = str(facts.get("sim_model_path") or "").strip()
        if not model_path:
            return None
        namespace = ros2_namespace(state)
        joint_mapping = json.dumps(facts.get("sim_joint_mapping", {}) or {}, ensure_ascii=True, sort_keys=True)
        return " ".join(
            [
                "source /opt/ros/*/setup.bash 2>/dev/null;",
                "/usr/bin/python3 -m roboclaw.embodied.simulation.mujoco_ros2_node",
                f"--model-path {shlex.quote(model_path)}",
                f"--namespace {shlex.quote(namespace)}",
                f"--joint-mapping {shlex.quote(joint_mapping)}",
                "--viewer-port 9878",
            ]
        )
    if not state.robot_attachments:
        return None
    primary_robot = state.robot_attachments[0]["robot_id"]
    profile = primary_profile(state)
    device_by_id = str(facts.get("serial_device_by_id") or "").strip()
    if profile is None or not device_by_id:
        return None
    namespace = ros2_namespace(state)
    return profile.control_launch_command(
        namespace=namespace,
        robot_id=primary_robot,
        device_by_id=device_by_id,
    )


def ros2_namespace(state: SetupOnboardingState) -> str:
    prefix = str(os.environ.get("ROBOCLAW_ROS2_NAMESPACE_PREFIX") or "/roboclaw").strip() or "/roboclaw"
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    prefix = re.sub(r"[^A-Za-z0-9_/]", "_", prefix).rstrip("/") or "/roboclaw"
    target = "sim" if state.detected_facts.get("simulation_requested") is True else "real"
    return f"{prefix}/{state.assembly_id}/{target}"


def asset_summary(state: SetupOnboardingState) -> str:
    return ", ".join(f"{key}={Path(value).name}" for key, value in sorted(state.generated_assets.items()))


def generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_ids(current_setup_id: str, robots: list[dict[str, Any]]) -> tuple[str, str, str, str, str]:
    if not robots:
        setup_id = current_setup_id
    elif current_setup_id.startswith("embodied_setup"):
        primary_robot_id = robots[0]["robot_id"]
        setup_id = f"{primary_robot_id}_setup"
    else:
        setup_id = current_setup_id
    return (
        setup_id,
        setup_id,
        setup_id,
        f"{setup_id}_real_local",
        f"{setup_id}_ros2_local",
    )


def sensor_attachment_id(base: str, existing: list[Any], index: int) -> str:
    del existing
    candidate = {
        "wrist": "wrist_camera",
        "overhead": "overhead_camera",
        "external": "external_camera",
    }.get(base, base if base in {"wrist_camera", "overhead_camera", "external_camera", "camera"} else "camera")
    return candidate if index == 0 or candidate != "camera" else f"{candidate}_{index + 1}"


def simulation_options_message(language: str | None, options: str) -> str:
    return localize_text(language, en=f"I can set up simulation for: {options}.\nTell me which robot you want to try.", zh=f"我可以为这些机器人准备仿真：{options}。\n告诉我你想试哪个机器人。")


def simulation_ready_message(language: str | None) -> str:
    return localize_text(language, en="Your simulation environment is ready!\nTry saying `open gripper` or `go home`.", zh="你的仿真环境已经准备好了！\n你可以试试说：`open gripper` 或 `go home`。")


def request_robot_message(language: str | None, example_robot: str) -> str:
    return localize_text(
        language,
        en="Let's get your robot connected!\nFirst, tell me what robot you have.\nFor example: `{}`, or `{} with a wrist camera`.".format(example_robot, example_robot),
        zh="让我们来连接你的机器人！\n先告诉我你有什么机器人。\n例如：`{}`，或 `{} 加腕部摄像头`。".format(example_robot, example_robot),
    )


def unknown_robot_message(language: str | None, robot_id: str, supported_models: str) -> str:
    return localize_text(
        language,
        en=f"I don't recognize the robot model '{robot_id}'.\nCurrently supported: {supported_models}.\nPlease check the name and try again.\nTechnical detail: RoboClaw does not have a framework ROS2 control surface profile for this model yet.",
        zh=f"我不认识机器人型号 '{robot_id}'。\n目前支持的型号：{supported_models}。\n请检查名称后再试一次。",
    )


def connection_confirmation_message(language: str | None, state: SetupOnboardingState) -> str:
    summary = component_summary(state)
    return localize_text(
        language,
        en=f"I saved what you told me about this setup: {summary}.\nOne quick question: are these devices already connected to this machine?\nYou can answer naturally, for example: `connected`, `not connected`, `已经接好了`, or `还没连接`.",
        zh=f"我已经记下你刚才告诉我的 setup 信息：{summary}。\n还有一个简单问题：这些设备现在是否已经接到这台机器上？\n你可以自然回答，例如：`connected`、`not connected`、`已经接好了`，或者 `还没连接`。",
    )


def calibration_ready_message(language: str | None, state: SetupOnboardingState) -> str:
    summary = component_summary(state)
    assets = asset_summary(state)
    return localize_text(
        language,
        en=f"This setup is now ready: {summary}.\nCalibration is done, and RoboClaw has already checked the setup files.\nYou can now connect, calibrate, move, debug, or reset.\nCreated files: {assets}",
        zh=f"这个 setup 现在已经就绪：{summary}。\n标定已经完成，RoboClaw 也已经检查过这些 setup 文件。\n现在你可以继续连接、标定、移动、排查问题或重置。\n已生成的文件：{assets}",
    )


def calibration_missing_message(language: str | None, robot_id: str, expected_path: str | None) -> str:
    return localize_text(
        language,
        en=f"This `{robot_id}` robot needs calibration before you can use it.\nCalibration file location: `{expected_path}`.\nThere is no calibration file available in this environment yet.\nYou can start calibration in natural language, for example: `calibrate` or `help me calibrate`.\nTechnical detail: this setup requires framework-managed calibration before execution.",
        zh=f"这个 `{robot_id}` 机器人在使用前需要先完成标定。\n标定文件位置：`{expected_path}`。\n当前这个环境里还没有可用的标定文件。\n直接告诉我开始标定即可，例如：`calibrate`、`帮我标定` 或 `开始校准`。\n技术说明：这个 setup 在执行前需要 RoboClaw 管理的标定。",
    )


def serial_unresponsive_message(language: str | None, detail: str) -> str:
    return localize_text(
        language,
        en=f"I found a stable `/dev/serial/by-id/...` device, but it did not answer the registered embodiment probe.\nProbe result: `{detail}`.\nConnect the actual controller or expose the correct stable by-id device, then reply again.",
        zh=f"我找到了稳定的 `/dev/serial/by-id/...` 设备，但它没有回应当前本体注册的探测。\n探测结果：`{detail}`。\n请接上真正的控制器，或者暴露正确、稳定的 by-id 设备路径，然后再回复我。",
    )


def unstable_serial_message(language: str | None) -> str:
    return localize_text(
        language,
        en="I found a serial device, but I will not persist an unstable tty node.\nPlease expose a stable `/dev/serial/by-id/...` mapping for this controller, then reply again.",
        zh="我找到了串口设备，但不会把不稳定的 tty 节点写进配置。\n请为这个控制器提供稳定的 `/dev/serial/by-id/...` 映射，然后再回复我。",
    )


def ros2_partial_install_message(language: str | None, guide_summary: str, recipe_summary: str, shell_repair: str) -> str:
    return localize_text(
        language,
        en=f"Local probing is complete. This setup needs ROS2, and RoboClaw found a partial install on this machine.\nI also read the workspace ROS2 install guide: {guide_summary}.\nSelected install path: {recipe_summary}.\n{shell_repair}",
        zh=f"本地探测已经完成。这个 setup 需要 ROS2，而 RoboClaw 在这台机器上发现了部分安装。\n我也已经读过工作区里的 ROS2 安装指南：{guide_summary}。\n当前选择的安装路径：{recipe_summary}。\n{shell_repair}",
    )


def ros2_missing_message(language: str | None, guide_summary: str, recipe_summary: str) -> str:
    return localize_text(
        language,
        en=f"Local probing is complete. This setup needs ROS2, but ROS2 is not available on this machine yet.\nI also read the workspace ROS2 install guide: {guide_summary}.\nSelected install path: {recipe_summary}.\nTell me to start ROS2 install in natural language and I will prepare the guided flow.\nIf you want GUI tools such as RViz, say `need desktop tools` before starting the install.",
        zh=f"本地探测已经完成。这个 setup 需要 ROS2，但这台机器上目前还没有可用的 ROS2。\n我也已经读过工作区里的 ROS2 安装指南：{guide_summary}。\n当前选择的安装路径：{recipe_summary}。\n直接自然地告诉我开始安装 ROS2，我就会准备 RoboClaw 引导的安装流程。\n如果你需要 RViz 之类的 GUI 工具，可以在开始安装前告诉我 `need desktop tools`。",
    )


def ros2_recheck_failed_message(language: str | None) -> str:
    return localize_text(
        language,
        en="I re-checked this machine after your update, but `ros2` is still not available in the shell yet.\nContinue with the guided install steps, open a fresh shell if needed, then tell me that ROS2 is installed.",
        zh="我在你更新之后重新检查了这台机器，但当前 shell 里还是还不能使用 `ros2`。\n请继续完成引导式安装步骤；如果需要，打开一个新的 shell，然后告诉我 ROS2 已经装好了。",
    )


def ros2_waiting_message(language: str | None) -> str:
    return localize_text(
        language,
        en="This setup is still waiting in the ROS2 prerequisite stage.\nTell me to start ROS2 install and I will prepare or run the guided install flow.",
        zh="这个 setup 仍然停留在 ROS2 前置条件阶段。\n直接告诉我开始安装 ROS2，我就会继续准备或执行引导式安装流程。",
    )


def ros2_install_complete_message(language: str | None) -> str:
    return localize_text(
        language,
        en="The guided ROS2 install steps are complete.\nFinish the commands in your shell, then tell me that ROS2 is installed and I will verify the environment before generating the setup assets.",
        zh="引导式 ROS2 安装步骤已经全部给完了。\n请先在你的 shell 里把命令执行完，然后告诉我 ROS2 已经装好了，我会在生成 setup 资产之前先验证环境。",
    )


def validation_failed_message(language: str | None, issues: str, *, simulation: bool = False) -> str:
    if simulation:
        return localize_text(language, en=f"The simulation setup files were written, but validation is still failing:\n{issues}", zh=f"仿真 setup 文件已经写出，但校验仍然失败：\n{issues}")
    return localize_text(language, en=f"The setup assets were written, but validation is still failing:\n{issues}", zh=f"setup 资产已经写出，但校验仍然失败：\n{issues}")


def final_ready_message(language: str | None, state: SetupOnboardingState) -> str:
    summary = component_summary(state)
    assets = asset_summary(state)
    return localize_text(
        language,
        en=f"This setup is now ready: {summary}.\nI wrote the assembly, deployment, and adapter into the workspace. You can keep refining setup details in chat, or continue with connect / calibrate / move / debug / reset.\nGenerated assets: {assets}",
        zh=f"这个 setup 现在已经就绪：{summary}。\n我已经把 assembly、deployment 和 adapter 写入工作区。你可以继续在对话里细化 setup 细节，或者继续执行 connect / calibrate / move / debug / reset。\n生成的资产：{assets}",
    )
