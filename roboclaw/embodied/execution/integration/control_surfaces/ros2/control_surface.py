"""ROS2 control-surface server for framework-owned embodied execution."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from typing import Any

from roboclaw.config.paths import resolve_active_serial_device_path
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import (
    SO101_ROS2_PROFILE,
    get_ros2_profile,
)
from roboclaw.embodied.execution.integration.control_surfaces.ros2.so101_feetech import So101FeetechRuntime


class Ros2ControlSurfaceServer:
    """Expose a minimal ROS2 lifecycle and primitive surface."""

    def __init__(
        self,
        *,
        namespace: str,
        profile_id: str,
        robot_id: str,
        device_by_id: str,
        calibration_path: str | None,
        calibration_id: str,
        state_rate_hz: float,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self._rclpy = rclpy
        self._string = String
        self._namespace = namespace.rstrip("/")
        self._profile_id = profile_id
        self._robot_id = robot_id
        self._lock = threading.RLock()
        self._runtime = self._build_runtime(
            profile_id=profile_id,
            robot_id=robot_id,
            device_by_id=device_by_id,
            calibration_path=calibration_path,
            calibration_id=calibration_id,
        )
        self._last_primitive: str | None = None
        self._last_error: str | None = None
        self._last_result: dict[str, Any] = {}

        class ControlSurfaceNode(Node):
            pass

        self.node = ControlSurfaceNode("roboclaw_control_surface")
        self._state_publisher = self.node.create_publisher(String, f"{self._namespace}/state", 10)
        self._health_publisher = self.node.create_publisher(String, f"{self._namespace}/health", 10)
        self._events_publisher = self.node.create_publisher(String, f"{self._namespace}/events", 10)
        self.node.create_timer(max(0.25, 1.0 / max(state_rate_hz, 0.1)), self._publish_status)

        self.node.create_service(Trigger, f"{self._namespace}/connect", self._handle_connect)
        self.node.create_service(Trigger, f"{self._namespace}/disconnect", self._handle_disconnect)
        self.node.create_service(Trigger, f"{self._namespace}/stop", self._handle_stop)
        self.node.create_service(Trigger, f"{self._namespace}/reset", self._handle_reset)
        self.node.create_service(Trigger, f"{self._namespace}/recover", self._handle_recover)
        self.node.create_service(Trigger, f"{self._namespace}/debug_snapshot", self._handle_debug_snapshot)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_gripper_open", self._handle_gripper_open)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_gripper_close", self._handle_gripper_close)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_go_home", self._handle_go_home)

    def _build_runtime(
        self,
        *,
        profile_id: str,
        robot_id: str,
        device_by_id: str,
        calibration_path: str | None,
        calibration_id: str,
    ) -> Any:
        profile = get_ros2_profile(profile_id)
        robot_profile = get_ros2_profile(robot_id)
        if profile is None:
            raise ValueError(f"Unknown control-surface ROS2 profile '{profile_id}'.")
        if robot_profile is None:
            raise ValueError(f"Unknown control-surface ROS2 robot '{robot_id}'.")
        if profile.id != robot_profile.id:
            raise ValueError(
                f"Control-surface ROS2 profile/robot mismatch: profile='{profile.id}' robot='{robot_id}'."
            )
        if profile.id != SO101_ROS2_PROFILE.id:
            raise ValueError(
                f"Control-surface ROS2 server does not support profile '{profile.id}' yet."
            )
        return So101FeetechRuntime(
            device_by_id=device_by_id,
            robot_name=robot_id,
            calibration_path=calibration_path,
            calibration_id=calibration_id,
        )

    def spin(self) -> None:
        self._rclpy.spin(self.node)

    def shutdown(self) -> None:
        try:
            with self._lock:
                self._runtime.disconnect()
        except Exception:
            pass
        self.node.destroy_node()
        try:
            self._rclpy.shutdown()
        except Exception:
            pass

    def _status_payload(self) -> dict[str, Any]:
        with self._lock:
            try:
                snapshot = self._runtime.snapshot()
            except Exception as exc:
                snapshot = {
                    "connected": self._runtime.connected,
                    "snapshot_error": str(exc),
                }
        return {
            "profile_id": self._profile_id,
            "robot_id": self._robot_id,
            "connected": snapshot.get("connected"),
            "device_by_id": snapshot.get("device_by_id"),
            "resolved_device": snapshot.get("resolved_device"),
            "gripper_present_raw": snapshot.get("gripper_present_raw"),
            "gripper_percent": snapshot.get("gripper_percent"),
            "last_primitive": self._last_primitive,
            "last_error": self._last_error,
            "snapshot_error": snapshot.get("snapshot_error"),
        }

    def _publish_status(self) -> None:
        try:
            status = json.dumps(self._status_payload(), ensure_ascii=False, sort_keys=True)
            self._state_publisher.publish(self._string(data=status))
            health = json.dumps({"ok": self._last_error is None, "last_error": self._last_error}, ensure_ascii=False)
            self._health_publisher.publish(self._string(data=health))
            event = json.dumps({"primitive": self._last_primitive, "error": self._last_error}, ensure_ascii=False)
            self._events_publisher.publish(self._string(data=event))
        except Exception as exc:
            self._last_error = str(exc)

    def _snapshot_locked(self) -> dict[str, Any]:
        with self._lock:
            return self._runtime.snapshot()

    def _success(self, response: Any, message: str, result: dict[str, Any] | None = None) -> Any:
        self._last_error = None
        self._last_result = result or {}
        response.success = True
        response.message = message
        return response

    @staticmethod
    def _is_recoverable_runtime_error(exc: Exception) -> bool:
        lower = str(exc).lower()
        return any(
            token in lower
            for token in (
                "port is in use",
                "there is no status packet",
                "incorrect status packet",
            )
        )

    @classmethod
    def _friendly_runtime_error_message(cls, exc: Exception) -> str:
        lower = str(exc).lower()
        if "port is in use" in lower:
            return (
                "The SO101 serial port is still busy. RoboClaw already tried to clear the current robot connection automatically, "
                "but something else is still using the arm. Close the other program and reply `connect` again."
            )
        if "there is no status packet" in lower or "incorrect status packet" in lower:
            return (
                "RoboClaw could reach the SO101 USB device, but the arm did not answer. "
                "Please make sure the arm power is on and the USB/serial cable is firmly connected, then reply `connect` again."
            )
        return str(exc)

    def _runtime_device_candidates(self) -> tuple[str, ...]:
        candidates: set[str] = set()
        resolved = getattr(self._runtime, "_resolved_device_path", None)
        if resolved is not None:
            candidates.add(os.path.realpath(str(resolved)))
        device_by_id = str(getattr(self._runtime, "device_by_id", "") or "").strip()
        if device_by_id:
            try:
                candidates.add(os.path.realpath(str(resolve_active_serial_device_path(device_by_id))))
            except Exception:
                candidates.add(os.path.realpath(device_by_id))
        return tuple(sorted(item for item in candidates if item))

    @staticmethod
    def _pids_using_device(device_path: str) -> tuple[int, ...]:
        current_pid = os.getpid()
        owners: list[int] = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == current_pid:
                continue
            fd_dir = os.path.join(entry.path, "fd")
            try:
                fd_entries = os.scandir(fd_dir)
            except OSError:
                continue
            with fd_entries as iterator:
                for fd_entry in iterator:
                    try:
                        target = os.readlink(fd_entry.path)
                    except OSError:
                        continue
                    if os.path.realpath(target) == device_path:
                        owners.append(pid)
                        break
        return tuple(sorted(set(owners)))

    def _terminate_competing_device_users_locked(self) -> None:
        current_pid = os.getpid()
        owner_pids: set[int] = set()
        for device_path in self._runtime_device_candidates():
            owner_pids.update(self._pids_using_device(device_path))
        owner_pids.discard(current_pid)
        if not owner_pids:
            return
        for pid in owner_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            alive = {pid for pid in owner_pids if os.path.exists(f"/proc/{pid}")}
            if not alive:
                return
            time.sleep(0.05)
            owner_pids = alive
        for pid in owner_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue

    def _recover_runtime_locked(self) -> None:
        try:
            self._runtime.disconnect()
        except Exception:
            pass
        self._runtime.connect()

    def _call_with_runtime_recovery(self, action: Any) -> Any:
        try:
            return action()
        except Exception as exc:
            if not self._is_recoverable_runtime_error(exc):
                raise
            self._terminate_competing_device_users_locked()
        self._recover_runtime_locked()
        return action()

    def _failure(self, response: Any, exc: Exception) -> Any:
        friendly = self._friendly_runtime_error_message(exc)
        self._last_error = friendly
        response.success = False
        response.message = friendly
        return response

    def _handle_connect(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                result = self._call_with_runtime_recovery(
                    lambda: (
                        self._runtime.connect(),
                        self._runtime.snapshot(),
                    )[1]
                )
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "connected", result)

    def _handle_disconnect(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                self._runtime.disconnect()
                result = self._runtime.snapshot()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "disconnected", result)

    def _handle_stop(self, request: Any, response: Any) -> Any:
        del request
        return self._success(response, "stop acknowledged", self._snapshot_locked())

    def _handle_reset(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                def run() -> Any:
                    if not self._runtime.connected:
                        self._runtime.connect()
                    self._last_primitive = "go_named_pose"
                    return self._runtime.go_home()

                result = self._call_with_runtime_recovery(run)
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "reset to home", result)

    def _handle_recover(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                result = self._call_with_runtime_recovery(
                    lambda: (
                        self._runtime.disconnect(),
                        self._runtime.connect(),
                        self._runtime.snapshot(),
                    )[2]
                )
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "recovered", result)

    def _handle_debug_snapshot(self, request: Any, response: Any) -> Any:
        del request
        payload = self._status_payload()
        return self._success(response, json.dumps(payload, ensure_ascii=False, sort_keys=True), payload)

    def _handle_gripper_open(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                def run() -> Any:
                    if not self._runtime.connected:
                        self._runtime.connect()
                    self._last_primitive = "gripper_open"
                    return self._runtime.open_gripper()

                result = self._call_with_runtime_recovery(run)
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "gripper opened", result)

    def _handle_gripper_close(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                def run() -> Any:
                    if not self._runtime.connected:
                        self._runtime.connect()
                    self._last_primitive = "gripper_close"
                    return self._runtime.close_gripper()

                result = self._call_with_runtime_recovery(run)
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "gripper closed", result)

    def _handle_go_home(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                def run() -> Any:
                    if not self._runtime.connected:
                        self._runtime.connect()
                    self._last_primitive = "go_named_pose"
                    return self._runtime.go_home()

                result = self._call_with_runtime_recovery(run)
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "home pose reached", result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboClaw ROS2 control-surface server.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--device-by-id", required=True)
    parser.add_argument("--calibration-path")
    parser.add_argument("--calibration-id", default="so101_real")
    parser.add_argument("--state-rate-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    import rclpy

    args = _parse_args()
    rclpy.init()
    server = Ros2ControlSurfaceServer(
        namespace=args.namespace,
        profile_id=args.profile_id,
        robot_id=args.robot_id,
        device_by_id=args.device_by_id,
        calibration_path=args.calibration_path,
        calibration_id=args.calibration_id,
        state_rate_hz=args.state_rate_hz,
    )
    try:
        server.spin()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
