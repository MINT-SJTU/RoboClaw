"""SO101 built-in embodiment declaration."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from roboclaw.embodied.builtins.model import BuiltinEmbodiment
from roboclaw.embodied.builtins.registry import register_builtin_embodiment
from roboclaw.embodied.definition.components.robots import SO101_ROBOT
from roboclaw.embodied.definition.foundation.schema import CapabilityFamily
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import (
    PrimitiveAliasSpec,
    PrimitiveServiceSpec,
    Ros2EmbodimentProfile,
)
from roboclaw.embodied.execution.integration.control_surfaces.ros2.so101_feetech import (
    So101CalibrationMonitor,
    So101FeetechRuntime,
)
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind
from roboclaw.embodied.execution.orchestration.skills import SkillSpec, SkillStep
from roboclaw.embodied.execution.orchestration.runtime.calibration import CalibrationDriver
from roboclaw.embodied.execution.orchestration.runtime.model import CalibrationPhase, RuntimeStatus
from roboclaw.embodied.localization import localize_text
from roboclaw.embodied.probes import ProbeProvider, ProbeResult

if TYPE_CHECKING:
    from roboclaw.embodied.execution.orchestration.runtime.executor import (
        ExecutionContext,
        ProcedureExecutionResult,
    )

SO101_DOC_PATH = "roboclaw/vendor/scservo_sdk"
SO101_SERIAL_PROBE_MODULE = "roboclaw.embodied.execution.integration.control_surfaces.ros2.scservo_probe"


def _result(
    *,
    procedure: ProcedureKind,
    ok: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> "ProcedureExecutionResult":
    from roboclaw.embodied.execution.orchestration.runtime.executor import ProcedureExecutionResult

    return ProcedureExecutionResult(
        procedure=procedure,
        ok=ok,
        message=message,
        details=details or {},
    )


@dataclass
class So101CalibrationFlow:
    """One in-memory calibration interaction for a setup runtime."""

    monitor: Any
    calibration_path: Path
    phase: CalibrationPhase
    interval_s: float
    heartbeat_s: float
    sample_limit: int | None
    stop_event: asyncio.Event | None = None
    task: asyncio.Task[None] | None = None
    last_error: str | None = None
    next_sample_idx: int = 1
    overwrite_existing: bool = False


class So101CalibrationDriver(CalibrationDriver):
    """SO101-specific manual calibration flow."""

    id = "so101_manual_calibration"

    def __init__(self) -> None:
        self._flows: dict[str, So101CalibrationFlow] = {}

    async def begin(
        self,
        context: "ExecutionContext",
        *,
        on_progress: Any | None = None,
    ) -> "ProcedureExecutionResult":
        del on_progress
        calibration_path = self._calibration_path(context)
        if calibration_path is None:
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en=f"Setup `{context.setup_id}` is not ready for calibration yet.",
                    zh=f"setup `{context.setup_id}` 还没有准备好开始标定。",
                ),
            )
        existing = self._flows.get(context.runtime.id)
        if existing is not None:
            return self.describe(context)

        overwrite_existing = calibration_path.exists()
        monitor: Any | None = None
        try:
            monitor = self._build_monitor(context)
            monitor.connect()
            monitor.prepare_manual_calibration()
            interval_s, heartbeat_s, sample_limit = self._stream_settings()
            self._flows[context.runtime.id] = So101CalibrationFlow(
                monitor=monitor,
                calibration_path=calibration_path,
                phase=CalibrationPhase.AWAIT_MID_POSE_ACK,
                interval_s=interval_s,
                heartbeat_s=heartbeat_s,
                sample_limit=sample_limit,
                overwrite_existing=overwrite_existing,
            )
        except Exception as exc:
            if monitor is not None:
                try:
                    monitor.disconnect()
                except Exception:
                    pass
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = str(exc)
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=self._friendly_error(
                    action="start the live SO101 calibration guide",
                    raw_error=exc,
                    reconnect_allowed=overwrite_existing,
                    language=self._preferred_language(context),
                ),
                details={"raw_error": str(exc), "calibration_path": str(calibration_path)},
            )

        context.runtime.status = RuntimeStatus.DISCONNECTED
        context.runtime.last_error = None
        return self.describe(context)

    async def advance(
        self,
        context: "ExecutionContext",
        user_input: str | None = None,
        *,
        on_progress: Any | None = None,
    ) -> "ProcedureExecutionResult":
        del user_input
        flow = self._flows.get(context.runtime.id)
        if flow is None:
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en="There is no pending calibration interaction for this setup. Tell me to calibrate first.",
                    zh="这个 setup 当前没有待继续的标定流程。请先告诉我开始标定。",
                ),
            )
        if flow.phase == CalibrationPhase.AWAIT_MID_POSE_ACK:
            return await self._start_stream(context, flow=flow, on_progress=on_progress)
        if flow.phase == CalibrationPhase.STREAMING:
            return await self._finish(context, flow=flow)
        self.cleanup(context.runtime.id)
        return _result(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message=localize_text(
                self._preferred_language(context),
                en="The pending calibration interaction is in an unknown state. Tell me to calibrate to start again.",
                zh="当前待继续的标定流程处于未知状态。请重新告诉我开始标定。",
            ),
        )

    def describe(self, context: "ExecutionContext") -> "ProcedureExecutionResult":
        flow = self._flows.get(context.runtime.id)
        if flow is None:
            calibration_path = self._calibration_path(context)
            expected_path = str(calibration_path) if calibration_path is not None else None
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en=(
                        f"Setup `{context.setup_id}` needs calibration before execution."
                        + (f" Expected canonical path: `{expected_path}`." if expected_path else "")
                        + " Tell me to calibrate after the hardware is ready so RoboClaw can walk you through it."
                    ),
                    zh=(
                        f"setup `{context.setup_id}` 在执行前需要先完成标定。"
                        + (f" 标准标定文件路径应为：`{expected_path}`。" if expected_path else "")
                        + " 等硬件准备好后，直接告诉我开始标定，我就会带你走完整个流程。"
                    ),
                ),
            )
        return self._phase_message(
            context,
            phase=flow.phase,
            calibration_path=flow.calibration_path,
            overwrite_existing=flow.overwrite_existing,
        )

    def phase(self, context: "ExecutionContext") -> str | None:
        flow = self._flows.get(context.runtime.id)
        return flow.phase if flow is not None else None

    def cleanup(self, runtime_id: str) -> None:
        flow = self._flows.pop(runtime_id, None)
        if flow is None:
            return
        if flow.stop_event is not None:
            flow.stop_event.set()
        if flow.task is not None:
            flow.task.cancel()
        try:
            flow.monitor.disconnect()
        except Exception:
            pass

    @staticmethod
    def _preferred_language(context: Any) -> str:
        return str(getattr(context, "preferred_language", "en") or "en")

    @staticmethod
    def _calibration_path(context: "ExecutionContext") -> Path | None:
        profile = context.profile
        if profile is None or not getattr(profile, "requires_calibration", False):
            return None
        return profile.canonical_calibration_path()

    @staticmethod
    def _serial_device_by_id(context: "ExecutionContext") -> str | None:
        device = str(context.deployment.connection.get("serial_device_by_id") or "").strip()
        if device:
            return device
        for robot_config in context.deployment.robots.values():
            candidate = str(robot_config.get("serial_device_by_id") or "").strip()
            if candidate:
                return candidate
        return None

    def _build_monitor(self, context: "ExecutionContext") -> So101CalibrationMonitor:
        device_by_id = self._serial_device_by_id(context)
        if not device_by_id:
            raise RuntimeError("No stable `/dev/serial/by-id/...` device is configured for this setup yet.")
        return So101CalibrationMonitor(device_by_id=device_by_id)

    @staticmethod
    def _is_serial_transport_error(message: str) -> bool:
        lower = message.lower()
        return any(
            token in lower
            for token in (
                "there is no status packet",
                "incorrect status packet",
                "failed to open servo device",
                "resource busy",
                "permission denied",
                "no such file or directory",
            )
        )

    @classmethod
    def _friendly_error(
        cls,
        *,
        action: str,
        raw_error: Exception,
        reconnect_allowed: bool,
        language: str = "en",
    ) -> str:
        message = str(raw_error).strip()
        lower = message.lower()
        reconnect_hint = (
            localize_text(
                language,
                en=" If this setup was already working earlier, tell me to reconnect it first, then tell me to start calibration again.",
                zh=" 如果这个 setup 之前是能工作的，先告诉我重新连接它，然后再告诉我重新开始标定。",
            )
            if reconnect_allowed
            else ""
        )
        if "stable `/dev/serial/by-id/" in message or "/dev/serial/by-id/" in message and "configured" in lower:
            return localize_text(
                language,
                en=(
                    f"I could not find a stable SO101 USB/serial device, so {action}."
                    " Plug the controller into this machine and make sure the `/dev/serial/by-id/...` path is available,"
                    " then tell me to start calibration again."
                ),
                zh=(
                    f"我没有找到稳定的 SO101 USB/串口设备，因此无法{action}。"
                    " 请把控制器接到这台机器上，并确认 `/dev/serial/by-id/...` 路径可用，"
                    " 然后再告诉我开始标定。"
                ),
            )
        if "failed to open servo device" in lower or "resource busy" in lower or "permission denied" in lower:
            return localize_text(
                language,
                en=(
                    f"I found the SO101 USB device, but I could not open it, so {action}."
                    " Please make sure the arm power is on, the USB/serial cable is firmly connected,"
                    " and no other program is using the arm."
                    " Then tell me to start calibration again."
                    + reconnect_hint
                ),
                zh=(
                    f"我找到了 SO101 的 USB 设备，但无法打开它，因此无法{action}。"
                    " 请确认机械臂已经上电、USB/串口线连接牢靠，而且没有其他程序正在占用这台机械臂。"
                    " 然后再告诉我开始标定。"
                    + reconnect_hint
                ),
            )
        if cls._is_serial_transport_error(message):
            return localize_text(
                language,
                en=(
                    f"I could talk to the SO101 USB device, but the arm did not answer, so {action}."
                    " Please make sure the arm power is on and the USB/serial cable is firmly connected."
                    " If you just replugged the cable or powered the arm back on, tell me to start calibration again."
                    + reconnect_hint
                ),
                zh=(
                    f"我可以访问 SO101 的 USB 设备，但机械臂没有响应，因此无法{action}。"
                    " 请确认机械臂已经上电，并且 USB/串口线连接牢靠。"
                    " 如果你刚重新插了线或者重新上电，请再告诉我开始标定。"
                    + reconnect_hint
                ),
            )
        return localize_text(
            language,
            en=f"RoboClaw could not {action}. {message}",
            zh=f"RoboClaw 无法{action}。{message}",
        )

    @staticmethod
    def _stream_settings() -> tuple[float, float, int | None]:
        interval_s = 0.2
        heartbeat_s = 1.0
        raw_limit = os.environ.get("ROBOCLAW_SO101_CALIBRATION_SAMPLE_LIMIT", "").strip()
        sample_limit = int(raw_limit) if raw_limit else None
        return interval_s, heartbeat_s, sample_limit

    def _phase_message(
        self,
        context: "ExecutionContext",
        *,
        phase: str | CalibrationPhase,
        calibration_path: Path,
        overwrite_existing: bool = False,
    ) -> "ProcedureExecutionResult":
        expected_path = str(calibration_path)
        if phase == CalibrationPhase.AWAIT_MID_POSE_ACK:
            save_notice = (
                localize_text(
                    self._preferred_language(context),
                    en=f" RoboClaw will overwrite the existing calibration file at `{expected_path}` when you press Enter again.",
                    zh=f" 当你再次按 Enter 时，RoboClaw 会覆盖现有的标定文件 `{expected_path}`。",
                )
                if overwrite_existing
                else localize_text(
                    self._preferred_language(context),
                    en=f" RoboClaw will save the calibration file to `{expected_path}` when you press Enter again.",
                    zh=f" 当你再次按 Enter 时，RoboClaw 会把标定文件保存到 `{expected_path}`。",
                )
            )
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en=(
                        f"SO101 calibration is ready for setup `{context.setup_id}`."
                        " Move the arm to a middle pose first, then press Enter to start live calibration."
                        + save_notice
                    ),
                    zh=(
                        f"setup `{context.setup_id}` 的 SO101 标定已经准备好了。"
                        " 先把机械臂移到中间位姿，然后按 Enter 开始实时标定。"
                        + save_notice
                    ),
                ),
                details={"calibration_path": expected_path, "calibration_phase": phase},
            )
        if phase == CalibrationPhase.STREAMING:
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en=(
                        f"SO101 live calibration is already running for setup `{context.setup_id}`."
                        " Keep moving every joint through its full range of motion, then press Enter again to stop and save."
                        f" It will be saved to `{expected_path}`."
                    ),
                    zh=(
                        f"setup `{context.setup_id}` 的 SO101 实时标定已经在运行。"
                        " 继续把每个关节跑完整量程，然后再次按 Enter 停止并保存。"
                        f" 标定文件会保存到 `{expected_path}`。"
                    ),
                ),
                details={"calibration_path": expected_path, "calibration_phase": phase},
            )
        return _result(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message=localize_text(
                self._preferred_language(context),
                en="SO101 calibration is in an unknown state. Tell me to calibrate to restart it.",
                zh="SO101 标定当前处于未知状态。请重新告诉我开始标定。",
            ),
            details={"calibration_phase": phase},
        )

    async def _start_stream(
        self,
        context: "ExecutionContext",
        *,
        flow: So101CalibrationFlow,
        on_progress: Any | None,
    ) -> "ProcedureExecutionResult":
        try:
            mid_pose = flow.monitor.capture_mid_pose()
            flow.monitor.apply_half_turn_homings(mid_pose)
            initial_snapshot = flow.monitor.start_observation()
            flow.stop_event = asyncio.Event()
            flow.phase = CalibrationPhase.STREAMING
            if on_progress is not None:
                await on_progress(self._format_snapshot(initial_snapshot, sample_idx=1))
                flow.next_sample_idx = 2
            flow.task = asyncio.create_task(self._run_stream(flow, on_progress=on_progress))
        except Exception as exc:
            self.cleanup(context.runtime.id)
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = str(exc)
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=self._friendly_error(
                    action="start live SO101 calibration",
                    raw_error=exc,
                    reconnect_allowed=flow.overwrite_existing,
                    language=self._preferred_language(context),
                ),
                details={"raw_error": str(exc), "calibration_path": str(flow.calibration_path)},
            )

        context.runtime.status = RuntimeStatus.DISCONNECTED
        context.runtime.last_error = None
        return _result(
            procedure=ProcedureKind.CALIBRATE,
            ok=False,
            message=localize_text(
                self._preferred_language(context),
                en=(
                    f"SO101 live calibration started for setup `{context.setup_id}`."
                    " RoboClaw is now streaming a LeRobot-style `MIN | POS | MAX` table above."
                    " Move every joint through its full range of motion, then press Enter again to stop and save."
                    f" It will be saved to `{flow.calibration_path}`."
                ),
                zh=(
                    f"setup `{context.setup_id}` 的 SO101 实时标定已经开始。"
                    " RoboClaw 现在正在上方持续输出 LeRobot 风格的 `MIN | POS | MAX` 表格。"
                    " 请把每个关节都跑完整量程，然后再次按 Enter 停止并保存。"
                    f" 标定文件会保存到 `{flow.calibration_path}`。"
                ),
            ),
            details={"calibration_path": str(flow.calibration_path), "calibration_phase": "streaming"},
        )

    async def _finish(
        self,
        context: "ExecutionContext",
        *,
        flow: So101CalibrationFlow,
    ) -> "ProcedureExecutionResult":
        try:
            if flow.stop_event is not None:
                flow.stop_event.set()
            if flow.task is not None:
                await flow.task
            if flow.last_error:
                raise RuntimeError(flow.last_error)
            flow.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            payload = flow.monitor.export_calibration_payload()
            flow.calibration_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            self.cleanup(context.runtime.id)
            context.runtime.status = RuntimeStatus.ERROR
            context.runtime.last_error = str(exc)
            return _result(
                procedure=ProcedureKind.CALIBRATE,
                ok=False,
                message=localize_text(
                    self._preferred_language(context),
                    en=f"RoboClaw could not save the SO101 calibration file. {exc}",
                    zh=f"RoboClaw 无法保存 SO101 标定文件。{exc}",
                ),
            )

        self.cleanup(context.runtime.id)
        context.runtime.status = RuntimeStatus.DISCONNECTED
        context.runtime.last_error = None
        return _result(
            procedure=ProcedureKind.CALIBRATE,
            ok=True,
            message=localize_text(
                self._preferred_language(context),
                en=(
                    "Calibration complete! You can now control the arm. "
                    f"Calibration file saved to `{flow.calibration_path}`."
                ),
                zh=(
                    "标定完成！现在你可以控制机械臂了。"
                    f" 标定文件已保存到 `{flow.calibration_path}`。"
                ),
            ),
            details={"calibration_path": str(flow.calibration_path), "calibration_phase": "completed"},
        )

    async def _run_stream(
        self,
        flow: So101CalibrationFlow,
        *,
        on_progress: Any | None,
    ) -> None:
        sample_idx = flow.next_sample_idx - 1
        last_payload = ""
        last_emit = 0.0
        try:
            while True:
                if flow.stop_event is not None and flow.stop_event.is_set():
                    return
                sample_idx += 1
                snapshot = flow.monitor.snapshot_observed()
                payload = self._format_snapshot(snapshot, sample_idx=sample_idx)
                now = asyncio.get_running_loop().time()
                if on_progress is not None and (payload != last_payload or (now - last_emit) >= flow.heartbeat_s):
                    await on_progress(payload)
                    last_emit = now
                    last_payload = payload
                if flow.sample_limit is not None and sample_idx >= flow.sample_limit:
                    return
                await asyncio.sleep(flow.interval_s)
        except Exception as exc:
            flow.last_error = str(exc)
            if on_progress is not None:
                await on_progress(f"SO101 calibration stream stopped: {exc}")

    @staticmethod
    def _format_snapshot(snapshot: Any, *, sample_idx: int) -> str:
        del sample_idx

        def _cell(value: int | None) -> str:
            return "?" if value is None else str(value)

        lines = [
            f"SO101 calibration live view on `{snapshot.resolved_device or snapshot.device_by_id}`",
            "```text",
            "JOINT            ID     MIN    POS    MAX",
        ]
        for row in snapshot.rows:
            lines.append(
                f"{row.joint_name:<16} {row.servo_id:>2} { _cell(row.range_min_raw):>7} { _cell(row.position_raw):>6} { _cell(row.range_max_raw):>6}"
            )
        lines.append("```")
        return "\n".join(lines)


class So101SerialProbeProvider(ProbeProvider):
    """SO101 onboarding serial probe provider."""

    id = "so101_serial_probe"

    async def probe_serial_device(
        self,
        serial_by_id: str,
        *,
        run_tool: Any,
        on_progress: Any | None = None,
    ) -> ProbeResult:
        probe = await run_tool(
            "exec",
            {
                "command": (
                    "bash -lc 'PY_BIN=\"$(command -v python3 || command -v python || true)\"; "
                    "if [ -z \"$PY_BIN\" ]; then printf \"ROBOCLAW_SO101_SERIAL_PYTHON_MISSING\\n\"; exit 0; fi; "
                    f"\"$PY_BIN\" -m {SO101_SERIAL_PROBE_MODULE} {shlex.quote(serial_by_id)}'"
                )
            },
            on_progress=on_progress,
        )
        if "ROBOCLAW_SO101_SERIAL_OK" in probe:
            return ProbeResult(ok=True)
        lines = [line.strip() for line in probe.splitlines() if line.strip()]
        detail = lines[-1] if lines else "SO101 serial probe failed"
        return ProbeResult(ok=False, detail=detail)


def build_so101_runtime(
    *,
    device_by_id: str,
    robot_id: str,
    calibration_path: str | None,
    calibration_id: str,
) -> So101FeetechRuntime:
    """Build the SO101 runtime for the control-surface server."""

    return So101FeetechRuntime(
        device_by_id=device_by_id,
        robot_name=robot_id,
        calibration_path=calibration_path,
        calibration_id=calibration_id,
    )


SO101_ROS2_PROFILE = Ros2EmbodimentProfile(
    id="so101_ros2_standard",
    robot_id="so101",
    primitive_aliases=(
        PrimitiveAliasSpec(
            primitive_name="gripper_open",
            aliases=("打开夹爪", "张开夹爪", "open gripper", "open the gripper"),
        ),
        PrimitiveAliasSpec(
            primitive_name="gripper_close",
            aliases=("闭合夹爪", "关闭夹爪", "夹住", "close gripper", "close the gripper"),
        ),
        PrimitiveAliasSpec(
            primitive_name="go_named_pose",
            aliases=("回到 home", "回到原点", "回到初始位", "go home", "go to home"),
            default_args={"name": "home"},
        ),
    ),
    primitive_services=(
        PrimitiveServiceSpec(primitive_name="gripper_open", service_name="primitive_gripper_open"),
        PrimitiveServiceSpec(primitive_name="gripper_close", service_name="primitive_gripper_close"),
        PrimitiveServiceSpec(primitive_name="go_named_pose", service_name="primitive_go_home"),
    ),
    auto_probe_serial=True,
    control_surface_server_module="roboclaw.embodied.execution.integration.control_surfaces.ros2.control_surface",
    calibration_robot_name="so101",
    control_default_calibration_id="so101_real",
    calibration_driver_id=So101CalibrationDriver.id,
    probe_provider_id=So101SerialProbeProvider.id,
    notes=(
        "Control-surface profile for a ROS2-backed SO101 setup.",
        "Natural-language aliases stay in framework code so workspace assets remain setup-specific only.",
        "Primitive execution can fall back to profile-declared ROS2 services when no generic action surface exists.",
    ),
)


SO101_BUILTIN = BuiltinEmbodiment(
    id="so101",
    robot=SO101_ROBOT,
    ros2_profile=SO101_ROS2_PROFILE,
    calibration_driver_id=So101CalibrationDriver.id,
    probe_provider_id=So101SerialProbeProvider.id,
    onboarding_aliases=("so101", "so 101", "so-101", "so_101"),
    skills=(
        SkillSpec(
            name="pick_and_place",
            description="Open, home, close, home, and release.",
            steps=(
                SkillStep("gripper_open"),
                SkillStep("go_named_pose", {"name": "home"}),
                SkillStep("gripper_close"),
                SkillStep("go_named_pose", {"name": "home"}),
                SkillStep("gripper_open"),
            ),
            required_capabilities=(CapabilityFamily.END_EFFECTOR, CapabilityFamily.NAMED_POSE),
        ),
        SkillSpec(
            name="reset_arm",
            description="Return to home and open the gripper.",
            steps=(SkillStep("go_named_pose", {"name": "home"}), SkillStep("gripper_open")),
            required_capabilities=(CapabilityFamily.NAMED_POSE,),
        ),
    ),
    control_surface_runtime_factory=build_so101_runtime,
)


register_builtin_embodiment(
    SO101_BUILTIN,
    calibration_driver=So101CalibrationDriver(),
    probe_provider=So101SerialProbeProvider(),
)


__all__ = [
    "SO101_BUILTIN",
    "SO101_DOC_PATH",
    "SO101_ROS2_PROFILE",
    "SO101_SERIAL_PROBE_MODULE",
    "So101CalibrationDriver",
    "So101CalibrationFlow",
    "So101SerialProbeProvider",
    "build_so101_runtime",
]
